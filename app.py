
import os
import time
import hmac
import threading

import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template_string
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# ==========================================
# SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "waste_final_v1.tflite"
)

# Match the model training order
CLASS_NAMES = ["plastic", "paper", "metal", "glass"]

CONF_THRESHOLD = 0.65
NMS_THRESHOLD = 0.45

DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", "")

model_lock = threading.Lock()

latest_result = {
    "detected": False,
    "class": "none",
    "confidence": 0.0,
    "boxes": [],
    "timestamp": 0
}

result_lock = threading.Lock()

# ==========================================
# LOAD MODEL
# ==========================================

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=2
)

interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

print("MODEL LOADED", flush=True)
print("Input:", input_info["shape"], flush=True)
print("Output:", output_info["shape"], flush=True)


# ==========================================
# AUTHENTICATION
# ==========================================

def authorized():
    token = request.headers.get("X-Device-Token", "")

    return bool(DEVICE_TOKEN) and hmac.compare_digest(
        token, DEVICE_TOKEN
    )


# ==========================================
# DETECTION
# ==========================================

def detect_objects(frame):

    original_h, original_w = frame.shape[:2]

    shape = input_info["shape"]
    model_h = int(shape[1])
    model_w = int(shape[2])

    image = cv2.resize(frame, (model_w, model_h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dtype = input_info["dtype"]

    if np.issubdtype(dtype, np.floating):
        tensor = image.astype(dtype) / 255.0
    else:
        scale, zero = input_info["quantization"]

        if scale > 0:
            tensor = np.round(
                (image.astype(np.float32) / 255.0)
                / scale + zero
            )
            info = np.iinfo(dtype)
            tensor = np.clip(
                tensor, info.min, info.max
            ).astype(dtype)
        else:
            tensor = image.astype(dtype)

    tensor = np.expand_dims(tensor, axis=0)

    with model_lock:
        interpreter.set_tensor(input_info["index"], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])

    scale, zero = output_info["quantization"]

    if scale > 0:
        output = (
            output.astype(np.float32) - zero
        ) * scale

    predictions = np.squeeze(output)

    channels = 4 + len(CLASS_NAMES)

    if predictions.ndim != 2:
        raise ValueError("Unsupported output shape")

    if predictions.shape[0] == channels:
        predictions = predictions.T

    if predictions.shape[1] != channels:
        raise ValueError(
            "Unexpected output: " + str(output.shape)
        )

    boxes = []
    confidences = []
    class_ids = []

    scale_x = original_w / model_w
    scale_y = original_h / model_h

    for prediction in predictions:

        scores = prediction[4:]

        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, w, h = prediction[:4]

        # Raw YOLOv8 coordinates in model pixels
        x1 = int((cx - w / 2) * scale_x)
        y1 = int((cy - h / 2) * scale_y)
        x2 = int((cx + w / 2) * scale_x)
        y2 = int((cy + h / 2) * scale_y)

        x1 = max(0, min(x1, original_w - 1))
        y1 = max(0, min(y1, original_h - 1))
        x2 = max(0, min(x2, original_w))
        y2 = max(0, min(y2, original_h))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1, y1, x2 - x1, y2 - y1
        ])

        confidences.append(confidence)
        class_ids.append(class_id)

    # Non-Maximum Suppression, separately per class
    detections = []

    for class_id in sorted(set(class_ids)):
        positions = [
            i for i, c in enumerate(class_ids)
            if c == class_id
        ]

        group_boxes = [boxes[i] for i in positions]
        group_scores = [confidences[i] for i in positions]

        indices = cv2.dnn.NMSBoxes(
            group_boxes,
            group_scores,
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        for index in np.array(indices).reshape(-1):
            i = positions[int(index)]
            x, y, w, h = boxes[i]

            detections.append({
                "class": CLASS_NAMES[class_ids[i]],
                "confidence": round(confidences[i], 3),
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

    detections.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    return detections[:10]


# ==========================================
# LIVE CAMERA WEB PAGE
# ==========================================

CAMERA_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport"
 content="width=device-width, initial-scale=1">

<title>Bottle AI Live Detection</title>

<style>
body {
    margin: 0;
    padding: 16px;
    background: #101827;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

h2 { margin-bottom: 8px; }

canvas {
    width: 100%;
    max-width: 640px;
    border: 2px solid #22c55e;
    border-radius: 12px;
    background: black;
}

input, button {
    padding: 12px;
    margin: 6px;
    border-radius: 8px;
    font-size: 15px;
}

button {
    background: #22c55e;
    border: none;
    font-weight: bold;
}

#status {
    margin: 14px;
    color: #cbd5e1;
}

#result {
    font-size: 20px;
    font-weight: bold;
    color: #86efac;
}

video { display: none; }
</style>
</head>

<body>

<h2>Plastic & Glass Bottle AI</h2>

<p>Live AI Detection - Render Cloud</p>

<input id="token" type="password"
 placeholder="Device Token">

<br>

<button onclick="startCamera()">
Start Camera
</button>

<button onclick="stopCamera()">
Stop Camera
</button>

<p id="status">Camera stopped</p>

<video id="video" autoplay playsinline muted></video>
<canvas id="view" width="640" height="480"></canvas>

<p id="result">Waiting for detection...</p>

<script>

const video = document.getElementById("video");
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

const statusText = document.getElementById("status");
const resultText = document.getElementById("result");

let stream = null;
let running = false;
let detections = [];

function draw() {

    if (running && video.readyState >= 2) {

        ctx.drawImage(
            video, 0, 0,
            canvas.width, canvas.height
        );

        detections.forEach(d => {

            const isPlastic = d.class === "plastic";

            const color = isPlastic
                ? "#22c55e"
                : "#f59e0b";

            ctx.strokeStyle = color;
            ctx.lineWidth = 4;

            ctx.strokeRect(
                d.x, d.y, d.w, d.h
            );

            const label =
                d.class + " " +
                Math.round(d.confidence * 100) + "%";

            ctx.font = "bold 16px Arial";

            const labelWidth =
                ctx.measureText(label).width + 12;

            const labelY = Math.max(22, d.y);

            ctx.fillStyle = color;

            ctx.fillRect(
                d.x,
                labelY - 22,
                labelWidth,
                24
            );

            ctx.fillStyle = "#101827";

            ctx.fillText(
                label,
                d.x + 6,
                labelY - 5
            );
        });
    }

    if (running) {
        requestAnimationFrame(draw);
    }
}

async function startCamera() {

    if (running) return;

    const token = document.getElementById("token").value;

    if (!token) {
        statusText.textContent = "Enter Device Token";
        return;
    }

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        running = true;

        statusText.textContent = "Camera running";

        draw();
        detectLoop();

    } catch (error) {
        statusText.textContent =
            "Camera error: " + error.message;
    }
}

function stopCamera() {

    running = false;
    detections = [];

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    statusText.textContent = "Camera stopped";
}

async function detectLoop() {

    while (running) {

        const capture = document.createElement("canvas");
        capture.width = 640;
        capture.height = 480;

        const c = capture.getContext("2d");

        c.drawImage(video, 0, 0, 640, 480);

        const blob = await new Promise(resolve =>
            capture.toBlob(resolve, "image/jpeg", 0.65)
        );

        if (!blob || !running) break;

        const form = new FormData();
        form.append("image", blob, "frame.jpg");

        const token =
            document.getElementById("token").value;

        try {

            const response = await fetch("/detect", {
                method: "POST",
                headers: {
                    "X-Device-Token": token
                },
                body: form
            });

            const data = await response.json();

            if (!response.ok) {
                detections = [];
                statusText.textContent =
                    "Error: " +
                    (data.error || response.status);
                running = false;
                break;
            }

            detections = data.boxes || [];

            if (detections.length > 0) {

                const best = detections[0];

                resultText.textContent =
                    best.class.toUpperCase() +
                    " - " +
                    Math.round(best.confidence * 100) +
                    "%";

            } else {
                resultText.textContent =
                    "No bottle detected";
            }

            statusText.textContent =
                "AI detection active";

        } catch (error) {

            detections = [];
            statusText.textContent =
                "Network error: " + error.message;
        }

        await new Promise(resolve =>
            setTimeout(resolve, 800)
        );
    }
}

</script>
</body>
</html>
"""


@app.get("/camera")
def camera_page():
    return render_template_string(CAMERA_HTML)


# ==========================================
# DETECT FRAME
# ==========================================

@app.post("/detect")
def detect_frame():

    global latest_result

    if not authorized():
        return jsonify({"error": "Unauthorized"}), 401

    uploaded = request.files.get("image")

    if not uploaded:
        return jsonify({"error": "No image"}), 400

    image_bytes = uploaded.read()

    image_array = np.frombuffer(
        image_bytes, dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        boxes = detect_objects(frame)

    except Exception:
        app.logger.exception("Inference failed")
        return jsonify({"error": "Inference failed"}), 500

    best = boxes[0] if boxes else None

    result = {
        "detected": bool(boxes),
        "class": best["class"] if best else "none",
        "confidence": best["confidence"] if best else 0.0,
        "boxes": boxes,
        "timestamp": time.time()
    }

    with result_lock:
        latest_result = result

    return jsonify(result)


# ==========================================
# STATUS
# ==========================================

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Bottle AI Live Detection",
        "camera": "/camera"
    })


@app.get("/status")
def status():
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "input_shape": input_info["shape"].tolist(),
        "output_shape": output_info["shape"].tolist()
    })


# Future ESP32 integration (not used yet)
@app.get("/result")
def result():

    if not authorized():
        return jsonify({"error": "Unauthorized"}), 401

    with result_lock:
        data = latest_result.copy()

    if time.time() - data["timestamp"] > 5:
        return jsonify({
            "detected": False,
            "class": "none",
            "confidence": 0.0,
            "boxes": [],
            "timestamp": data["timestamp"]
        })

    return jsonify(data)


# ==========================================
# START
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        debug=False
    )
