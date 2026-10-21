
import os
import time
import hmac
import threading

import cv2
import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string
)

from ai_edge_litert.interpreter import Interpreter

# =========================================
# FLASK SETTINGS
# =========================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "waste_final_v1.tflite"
)

# Original YOLO training class order
CLASS_NAMES = [
    "plastic",
    "paper",
    "metal",
    "glass"
]

# Detect ONLY these classes
ALLOWED_CLASSES = {"plastic", "glass"}

CONF_THRESHOLD = 0.65
NMS_THRESHOLD = 0.45

DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", "")

model_lock = threading.Lock()
result_lock = threading.Lock()

latest_result = {
    "detected": False,
    "class": "none",
    "confidence": 0.0,
    "boxes": [],
    "timestamp": 0
}

# =========================================
# LOAD TFLITE MODEL
# =========================================

print("Loading TFLite model...", flush=True)

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=2
)

interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

print("MODEL LOADED!", flush=True)
print("Input:", input_info["shape"], flush=True)
print("Output:", output_info["shape"], flush=True)

# =========================================
# AUTHENTICATION
# =========================================

def authorized():
    token = request.headers.get(
        "X-Device-Token", ""
    )

    return (
        bool(DEVICE_TOKEN)
        and hmac.compare_digest(token, DEVICE_TOKEN)
    )

# =========================================
# IMAGE PREPROCESSING
# =========================================

def preprocess(frame):

    model_h = int(input_info["shape"][1])
    model_w = int(input_info["shape"][2])

    image = cv2.resize(
        frame, (model_w, model_h)
    )

    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    dtype = input_info["dtype"]

    if np.issubdtype(dtype, np.floating):
        tensor = image.astype(dtype) / 255.0

    else:
        scale, zero = input_info["quantization"]

        if scale <= 0:
            raise ValueError(
                "Unsupported input quantization"
            )

        tensor = np.round(
            (image.astype(np.float32) / 255.0)
            / scale + zero
        )

        info = np.iinfo(dtype)

        tensor = np.clip(
            tensor, info.min, info.max
        ).astype(dtype)

    return np.expand_dims(tensor, axis=0)

# =========================================
# DETECT PLASTIC AND GLASS ONLY
# =========================================

def detect_objects(frame):

    original_h, original_w = frame.shape[:2]

    model_h = int(input_info["shape"][1])
    model_w = int(input_info["shape"][2])

    tensor = preprocess(frame)

    with model_lock:

        interpreter.set_tensor(
            input_info["index"], tensor
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_info["index"]
        )

    scale, zero = output_info["quantization"]

    if scale > 0:
        output = (
            output.astype(np.float32) - zero
        ) * scale

    predictions = np.squeeze(output)

    channels = 4 + len(CLASS_NAMES)

    if predictions.ndim != 2:
        raise ValueError(
            "Unexpected output shape: "
            + str(output.shape)
        )

    if predictions.shape[0] == channels:
        predictions = predictions.T

    if predictions.shape[1] != channels:
        raise ValueError(
            "Model classes do not match: "
            + str(output.shape)
        )

    boxes = []
    confidences = []
    class_ids = []

    scale_x = original_w / model_w
    scale_y = original_h / model_h

    for pred in predictions:

        scores = pred[4:]

        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        label = CLASS_NAMES[class_id]

        # Ignore paper and metal
        if label not in ALLOWED_CLASSES:
            continue

        cx, cy, w, h = pred[:4]

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
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        confidences.append(confidence)
        class_ids.append(class_id)

    detections = []

    # NMS separately for each class
    for class_id in set(class_ids):

        selected = [
            i for i, c in enumerate(class_ids)
            if c == class_id
        ]

        group_boxes = [
            boxes[i] for i in selected
        ]

        group_scores = [
            confidences[i] for i in selected
        ]

        indices = cv2.dnn.NMSBoxes(
            group_boxes,
            group_scores,
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        for index in np.array(indices).reshape(-1):

            i = selected[int(index)]
            x, y, w, h = boxes[i]

            detections.append({
                "class": CLASS_NAMES[class_id],
                "confidence": round(
                    confidences[i], 3
                ),
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

# =========================================
# LIVE CAMERA WEBSITE
# =========================================

CAMERA_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>Plastic Glass Bottle AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 12px;
    font-family: Arial, sans-serif;
    background: #0f172a;
    color: white;
    text-align: center;
}

h2 {
    margin: 10px 0;
}

#token {
    width: 90%;
    max-width: 340px;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
}

#viewer {
    width: 100%;
    max-width: 720px;
    margin: 15px auto;
    border: 2px solid #22c55e;
    border-radius: 12px;
    overflow: hidden;
    background: black;
}

#canvas {
    display: block;
    width: 100%;
    height: auto;
}

#video {
    display: none;
}

#status {
    color: #cbd5e1;
    padding: 10px;
}

#result {
    font-size: 23px;
    font-weight: bold;
    color: #86efac;
    padding: 12px;
}

#retry {
    display: none;
    padding: 12px 20px;
    background: #22c55e;
    border: none;
    border-radius: 8px;
    font-size: 16px;
}

</style>

</head>

<body>

<h2>Plastic & Glass Bottle AI</h2>

<p>Live Camera - Render Cloud</p>

<input
id="token"
type="password"
placeholder="Enter Device Token"
autocomplete="off">

<p id="status">Opening camera...</p>

<button id="retry" onclick="startCamera()">
Enable Camera
</button>

<div id="viewer">

<video id="video"
autoplay
muted
playsinline>
</video>

<canvas id="canvas"
width="640"
height="480">
</canvas>

</div>

<div id="result">
Waiting for AI detection...
</div>

<script>

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const tokenEl = document.getElementById("token");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const retryEl = document.getElementById("retry");

let stream = null;
let running = false;
let detections = [];
let lastDetectionTime = 0;

const DETECTION_DELAY = 700;
const BOX_MAX_AGE = 3000;

function sleep(ms) {
    return new Promise(resolve =>
        setTimeout(resolve, ms)
    );
}

// ======================================
// START CAMERA AUTOMATICALLY
// ======================================

async function startCamera() {

    if (running) return;

    statusEl.textContent = "Opening camera...";
    retryEl.style.display = "none";

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: {
                    ideal: "environment"
                },
                width: {
                    ideal: 640
                },
                height: {
                    ideal: 480
                }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        running = true;

        statusEl.textContent = "CAMERA LIVE";

        drawLoop();
        detectLoop();

    } catch (error) {

        statusEl.textContent =
            "Camera error: " + error.message;

        retryEl.style.display = "inline-block";
    }
}

// ======================================
// DRAW VIDEO AND DETECTION BOXES
// ======================================

function drawLoop() {

    if (!running) return;

    if (video.readyState >= 2) {

        ctx.drawImage(
            video,
            0, 0,
            canvas.width,
            canvas.height
        );

        const boxesFresh =
            Date.now() - lastDetectionTime
            < BOX_MAX_AGE;

        if (boxesFresh) {

            for (const d of detections) {

                let color = "#22c55e";

                if (d.class === "glass") {
                    color = "#38bdf8";
                }

                ctx.strokeStyle = color;
                ctx.lineWidth = 4;

                ctx.strokeRect(
                    d.x,
                    d.y,
                    d.w,
                    d.h
                );

                const label =
                    d.class.toUpperCase()
                    + " "
                    + Math.round(
                        d.confidence * 100
                    )
                    + "%";

                ctx.font = "bold 18px Arial";

                const labelWidth =
                    ctx.measureText(label).width + 12;

                const labelY = Math.max(d.y, 26);

                ctx.fillStyle = color;

                ctx.fillRect(
                    d.x,
                    labelY - 25,
                    labelWidth,
                    26
                );

                ctx.fillStyle = "#000";

                ctx.fillText(
                    label,
                    d.x + 6,
                    labelY - 5
                );
            }
        }
    }

    requestAnimationFrame(drawLoop);
}

// ======================================
// SEND LIVE FRAMES TO RENDER
// ======================================

async function detectLoop() {

    while (running) {

        const token = tokenEl.value.trim();

        if (!token) {

            detections = [];

            resultEl.textContent =
                "Enter Device Token to enable AI";

            await sleep(500);
            continue;
        }

        if (video.readyState < 2) {
            await sleep(300);
            continue;
        }

        const capture = document.createElement("canvas");

        capture.width = canvas.width;
        capture.height = canvas.height;

        const c = capture.getContext("2d");

        c.drawImage(
            video,
            0, 0,
            capture.width,
            capture.height
        );

        const blob = await new Promise(resolve =>
            capture.toBlob(
                resolve,
                "image/jpeg",
                0.65
            )
        );

        if (!blob) {
            await sleep(500);
            continue;
        }

        const form = new FormData();

        form.append(
            "image",
            blob,
            "frame.jpg"
        );

        try {

            statusEl.textContent =
                "CAMERA LIVE - AI ANALYZING";

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
                lastDetectionTime = 0;

                resultEl.textContent =
                    data.error || "AI error";

                if (response.status === 401) {

                    statusEl.textContent =
                        "Incorrect Device Token";

                    tokenEl.value = "";
                }

                await sleep(1000);
                continue;
            }

            detections = data.boxes || [];
            lastDetectionTime = Date.now();

            if (detections.length > 0) {

                const best = detections[0];

                resultEl.textContent =
                    best.class.toUpperCase()
                    + " - "
                    + Math.round(
                        best.confidence * 100
                    )
                    + "%";

            } else {

                resultEl.textContent =
                    "No plastic or glass detected";
            }

            statusEl.textContent =
                "CAMERA LIVE - AI CONNECTED";

        } catch (error) {

            detections = [];
            lastDetectionTime = 0;

            statusEl.textContent =
                "Network error: " + error.message;
        }

        await sleep(DETECTION_DELAY);
    }
}

// ======================================
// AUTO START
// ======================================

window.addEventListener("load", () => {
    startCamera();
});

window.addEventListener("pagehide", () => {

    running = false;

    if (stream) {

        stream.getTracks().forEach(
            track => track.stop()
        );
    }
});

</script>

</body>
</html>
"""

# =========================================
# WEB ROUTES
# =========================================

@app.get("/")
def home():

    return jsonify({
        "status": "online",
        "service": "Plastic Glass Bottle AI",
        "camera": "/camera",
        "allowed_classes": [
            "plastic",
            "glass"
        ]
    })


@app.get("/camera")
def camera_page():

    response = app.make_response(
        render_template_string(CAMERA_HTML)
    )

    response.headers["Cache-Control"] = "no-store"

    return response


# =========================================
# DETECT API
# =========================================

@app.post("/detect")
def detect_frame():

    global latest_result

    if not authorized():

        return jsonify({
            "error": "Unauthorized"
        }), 401

    uploaded = request.files.get("image")

    if uploaded is None:

        return jsonify({
            "error": "No image received"
        }), 400

    image_bytes = uploaded.read()

    if not image_bytes:

        return jsonify({
            "error": "Empty image"
        }), 400

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({
            "error": "Invalid image"
        }), 400

    try:
        boxes = detect_objects(frame)

    except Exception:
        app.logger.exception("Inference error")

        return jsonify({
            "error": "Model inference failed"
        }), 500

    best = boxes[0] if boxes else None

    result = {
        "detected": bool(boxes),
        "class": (
            best["class"] if best else "none"
        ),
        "confidence": (
            best["confidence"] if best else 0.0
        ),
        "boxes": boxes,
        "timestamp": time.time()
    }

    with result_lock:
        latest_result = result

    return jsonify(result)


# =========================================
# STATUS API
# =========================================

@app.get("/status")
def status():

    return jsonify({
        "status": "online",
        "model_loaded": True,
        "input_shape": input_info["shape"].tolist(),
        "output_shape": output_info["shape"].tolist(),
        "allowed_classes": [
            "plastic",
            "glass"
        ]
    })


# =========================================
# FUTURE ESP32 API
# =========================================

@app.get("/result")
def get_result():

    if not authorized():

        return jsonify({
            "error": "Unauthorized"
        }), 401

    with result_lock:
        result = latest_result.copy()

    if time.time() - result["timestamp"] > 5:

        result["detected"] = False
        result["class"] = "none"
        result["confidence"] = 0.0
        result["boxes"] = []

    return jsonify(result)


# =========================================
# START SERVER
# =========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        debug=False
    )
