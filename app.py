
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


# ==========================================
# CONFIGURATION
# ==========================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "waste_final_v1.tflite"
)

# IMPORTANT:
# Must match your training class order.
CLASS_NAMES = [
    "plastic",
    "paper",
    "metal",
    "glass"
]

CONF_THRESHOLD = 0.65

DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "")

RESULT_MAX_AGE = 5

lock = threading.Lock()
model_lock = threading.Lock()

latest_result = {
    "detected": False,
    "class": "none",
    "confidence": 0.0,
    "box": None,
    "timestamp": 0
}


# ==========================================
# LOAD TFLITE MODEL
# ==========================================

print("Loading TFLite model...", flush=True)

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=2
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Model loaded!", flush=True)
print("Input:", input_details["shape"], flush=True)
print("Output:", output_details["shape"], flush=True)


# ==========================================
# AUTHENTICATION
# ==========================================

def authorized():
    if not DEVICE_TOKEN:
        return False

    token = (
        request.headers.get("X-Device-Token")
        or request.form.get("token", "")
    )

    return hmac.compare_digest(token, DEVICE_TOKEN)


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess(frame):
    shape = input_details["shape"]

    height = int(shape[1])
    width = int(shape[2])

    image = cv2.resize(
        frame,
        (width, height)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    dtype = input_details["dtype"]

    if np.issubdtype(dtype, np.floating):
        tensor = image.astype(dtype) / 255.0

    else:
        scale, zero = input_details["quantization"]

        if scale > 0:
            tensor = np.round(
                (image.astype(np.float32) / 255.0)
                / scale + zero
            )

            info = np.iinfo(dtype)

            tensor = np.clip(
                tensor,
                info.min,
                info.max
            ).astype(dtype)
        else:
            tensor = image.astype(dtype)

    return np.expand_dims(tensor, axis=0)


# ==========================================
# MODEL INFERENCE
# ==========================================

def detect(frame):

    tensor = preprocess(frame)

    with model_lock:
        interpreter.set_tensor(
            input_details["index"],
            tensor
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

    scale, zero = output_details["quantization"]

    if scale > 0:
        output = (
            output.astype(np.float32) - zero
        ) * scale

    predictions = np.squeeze(output)

    if predictions.ndim != 2:
        raise ValueError(
            "Unsupported model output: "
            + str(output.shape)
        )

    channels = 4 + len(CLASS_NAMES)

    # YOLOv8 output shape:
    # [channels, boxes] or [boxes, channels]
    if predictions.shape[0] == channels:
        predictions = predictions.T

    if predictions.shape[1] != channels:
        raise ValueError(
            "Model output does not match classes: "
            + str(output.shape)
        )

    scores = predictions[:, 4:]

    if scores.shape[0] == 0:
        return "none", 0.0, None

    best_classes = np.argmax(scores, axis=1)
    best_scores = np.max(scores, axis=1)

    index = int(np.argmax(best_scores))
    confidence = float(best_scores[index])

    if confidence < CONF_THRESHOLD:
        return "none", confidence, None

    class_index = int(best_classes[index])
    label = CLASS_NAMES[class_index]

    # Raw YOLO bounding box:
    # center_x, center_y, width, height
    box = predictions[index, :4].tolist()

    return label, confidence, box


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Plastic Glass Bottle AI",
        "model": "waste_final_v1.tflite",
        "camera_page": "/camera",
        "result_api": "/result"
    })


# ==========================================
# CAMERA PAGE
# ==========================================

CAMERA_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>Bottle AI Camera</title>

<style>
body {
    font-family: Arial, sans-serif;
    text-align: center;
    padding: 20px;
    background: #f5f5f5;
}

input, button {
    margin: 12px;
    padding: 12px;
    font-size: 16px;
    max-width: 95%;
}

button {
    background: #166b45;
    color: white;
    border: none;
    border-radius: 8px;
}

pre {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    text-align: left;
}
</style>

</head>

<body>

<h2>Plastic & Glass Bottle AI</h2>

<p>Capture a bottle image and send it to AI.</p>

<input
    type="password"
    id="token"
    placeholder="Device Token">

<br>

<input
    type="file"
    id="photo"
    accept="image/*"
    capture="environment">

<br>

<button onclick="sendImage()">
Detect Bottle
</button>

<pre id="result">Waiting for image...</pre>

<script>
async function sendImage() {

    const file = document.getElementById("photo").files[0];
    const token = document.getElementById("token").value;
    const result = document.getElementById("result");

    if (!file || !token) {
        result.textContent = "Select image and enter token.";
        return;
    }

    const form = new FormData();
    form.append("image", file);

    result.textContent = "Analyzing...";

    try {
        const response = await fetch("/detect", {
            method: "POST",
            headers: {
                "X-Device-Token": token
            },
            body: form
        });

        const data = await response.json();

        result.textContent =
            JSON.stringify(data, null, 2);

    } catch (error) {
        result.textContent = "Error: " + error;
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
# DETECTION API
# ==========================================

@app.post("/detect")
def detect_image():

    global latest_result

    if not authorized():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    uploaded = request.files.get("image")

    if uploaded is None:
        return jsonify({
            "error": "No image provided"
        }), 400

    try:
        raw = uploaded.read()

        if not raw:
            return jsonify({
                "error": "Empty image"
            }), 400

        array = np.frombuffer(raw, dtype=np.uint8)

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Invalid image"
            }), 400

        label, confidence, box = detect(frame)

        result = {
            "detected": label != "none",
            "class": label,
            "confidence": round(confidence, 3),
            "box": box,
            "timestamp": time.time()
        }

        with lock:
            latest_result = result

        return jsonify(result)

    except Exception as e:
        app.logger.exception("Detection failed")

        return jsonify({
            "error": "Detection failed",
            "type": type(e).__name__
        }), 500


# ==========================================
# ESP32 RESULT API
# ==========================================

@app.get("/result")
def result():

    if not authorized():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    with lock:
        data = latest_result.copy()

    if time.time() - data["timestamp"] > RESULT_MAX_AGE:
        data["detected"] = False
        data["class"] = "none"
        data["confidence"] = 0.0
        data["box"] = None

    return jsonify(data)


# ==========================================
# STATUS API
# ==========================================

@app.get("/status")
def status():
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "classes": CLASS_NAMES
    })


# ==========================================
# LOCAL START
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
