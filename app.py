
import os
import time
import hmac
import threading

import cv2
import numpy as np

from flask import Flask, request, jsonify, Response
from ultralytics import YOLO


# =========================================
# SETTINGS
# =========================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "waste_final_v1.tflite"
)

# Original class order
CLASS_NAMES = {
    0: "plastic",
    1: "paper",
    2: "metal",
    3: "glass"
}

# Detect ONLY plastic and glass
ALLOWED_CLASSES = {"plastic", "glass"}

SELECTED_CLASS_IDS = [
    i for i, name in CLASS_NAMES.items()
    if name in ALLOWED_CLASSES
]

CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.45

# Same sorting zone as the PC project
ARRIVAL_ROI = (
    0.25,
    0.208333,
    0.78125,
    0.875
)

DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "")

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
# LOAD ULTRALYTICS MODEL
# =========================================

print("Loading YOLO TFLite model...", flush=True)

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print("YOLO model initialized!", flush=True)


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


@app.post("/auth")
def auth():
    if not authorized():
        return jsonify({
            "authorized": False,
            "error": "Invalid token"
        }), 401

    return jsonify({
        "authorized": True
    })


# =========================================
# DETECT BOTTLES USING ULTRALYTICS
# =========================================

def detect_bottles(frame):

    height, width = frame.shape[:2]

    # Sorting zone from original PC code
    left, top, right, bottom = ARRIVAL_ROI

    rx1 = int(left * width)
    ry1 = int(top * height)
    rx2 = int(right * width)
    ry2 = int(bottom * height)

    roi_image = frame[ry1:ry2, rx1:rx2].copy()

    if roi_image.size == 0:
        raise ValueError("Invalid sorting zone")

    # Same YOLO detection method as PC
    with model_lock:
        result = model.predict(
            source=roi_image,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=SELECTED_CLASS_IDS,
            imgsz=640,
            verbose=False
        )[0]

    detections = []

    if result.boxes is not None:

        coordinates = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for bounds, class_id, confidence in zip(
            coordinates,
            class_ids,
            confidences
        ):

            if class_id not in SELECTED_CLASS_IDS:
                continue

            label = CLASS_NAMES.get(
                int(class_id),
                "unknown"
            )

            x1, y1, x2, y2 = map(int, bounds)

            # Convert ROI coordinates to full frame
            x1 += rx1
            x2 += rx1
            y1 += ry1
            y2 += ry1

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))

            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class": label,
                "confidence": round(
                    float(confidence), 3
                ),
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1
            })

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    sorting_zone = [
        rx1,
        ry1,
        rx2 - rx1,
        ry2 - ry1
    ]

    return detections[:10], sorting_zone


# =========================================
# LIVE CAMERA WEBSITE
# =========================================

CAMERA_HTML = r"""
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
    padding: 14px;
    background: #0f172a;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

h2 {
    margin: 12px 0;
}

#login {
    max-width: 400px;
    margin: 12vh auto;
    padding: 24px;
    background: #1e293b;
    border-radius: 16px;
}

input {
    width: 100%;
    padding: 14px;
    border: 0;
    border-radius: 8px;
    font-size: 17px;
    margin: 12px 0;
}

button {
    padding: 13px 24px;
    margin: 8px;
    background: #22c55e;
    color: #08111e;
    font-size: 16px;
    font-weight: bold;
    border: 0;
    border-radius: 8px;
}

button:disabled {
    opacity: 0.5;
}

#cameraPage {
    display: none;
}

#video {
    display: none;
}

#canvas {
    display: block;
    width: 100%;
    max-width: 720px;
    height: auto;
    margin: 12px auto;
    border: 2px solid #22c55e;
    border-radius: 12px;
    background: black;
}

#status {
    color: #cbd5e1;
    margin: 12px;
}

#result {
    color: #86efac;
    font-size: 23px;
    font-weight: bold;
    margin: 14px;
}

#error {
    color: #fca5a5;
}

#enable {
    display: none;
}
</style>
</head>

<body>

<div id="login">
    <h2>Plastic & Glass Bottle AI</h2>

    <p>Enter password to start</p>

    <input
        id="password"
        type="password"
        placeholder="Device Token"
        autocomplete="off">

    <button id="connect">
        Connect
    </button>

    <p id="error"></p>
</div>


<div id="cameraPage">
    <h2>Live Bottle Detection</h2>

    <p id="status">Opening camera...</p>

    <button id="enable">
        Enable Camera
    </button>

    <video id="video" autoplay muted playsinline></video>

    <canvas
        id="canvas"
        width="640"
        height="480">
    </canvas>

    <p id="result">Waiting for AI...</p>

    <button id="disconnect">
        Disconnect
    </button>
</div>


<script>
const $ = id => document.getElementById(id);

const video = $("video");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

let token = "";
let stream = null;
let running = false;
let session = 0;

let detections = [];
let roi = null;
let detectionTime = 0;

const DETECTION_DELAY = 700;
const BOX_MAX_AGE = 3000;

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


// =====================================
// LOGIN
// =====================================

$("connect").onclick = async function() {

    const candidate = $("password").value.trim();

    if (!candidate) {
        $("error").textContent = "Enter password";
        return;
    }

    $("connect").disabled = true;
    $("error").textContent = "Checking password...";

    try {
        const response = await fetch("/auth", {
            method: "POST",
            headers: {
                "X-Device-Token": candidate
            }
        });

        if (!response.ok) {
            $("error").textContent = "Incorrect password";
            return;
        }

        token = candidate;
        $("password").value = "";

        $("login").style.display = "none";
        $("cameraPage").style.display = "block";

        await startCamera();

    } catch (error) {
        $("error").textContent =
            "Connection error: " + error.message;
    } finally {
        $("connect").disabled = false;
    }
};

$("password").addEventListener("keydown", event => {
    if (event.key === "Enter") {
        $("connect").click();
    }
});


// =====================================
// CAMERA
// =====================================

async function startCamera() {

    if (running) return;

    $("status").textContent = "Opening camera...";

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: {ideal: "environment"},
                width: {ideal: 640},
                height: {ideal: 480}
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        running = true;
        session++;

        const mySession = session;

        $("enable").style.display = "none";
        $("status").textContent = "Camera LIVE";

        drawLoop(mySession);
        detectLoop(mySession);

    } catch (error) {
        $("status").textContent =
            "Camera error: " + error.message;

        $("enable").style.display = "inline-block";
    }
}

$("enable").onclick = startCamera;


// =====================================
// DRAW LIVE VIDEO AND BOXES
// =====================================

function drawLoop(mySession) {

    if (!running || session !== mySession) {
        return;
    }

    if (video.readyState >= 2) {
        ctx.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        // Sorting zone
        if (roi) {
            ctx.strokeStyle = "#00d5ff";
            ctx.lineWidth = 2;
            ctx.strokeRect(...roi);

            ctx.fillStyle = "#00d5ff";
            ctx.font = "bold 16px Arial";

            ctx.fillText(
                "SORTING ZONE",
                roi[0],
                Math.max(18, roi[1] - 8)
            );
        }

        // Bounding boxes
        if (Date.now() - detectionTime < BOX_MAX_AGE) {

            for (const d of detections) {

                const color = d.class === "plastic"
                    ? "#22c55e"
                    : "#ffad22";

                ctx.strokeStyle = color;
                ctx.lineWidth = 4;

                ctx.strokeRect(
                    d.x,
                    d.y,
                    d.w,
                    d.h
                );

                const label =
                    d.class.toUpperCase() + " " +
                    Math.round(d.confidence * 100) + "%";

                ctx.font = "bold 17px Arial";

                const labelY = Math.max(25, d.y);
                const textWidth =
                    ctx.measureText(label).width + 12;

                ctx.fillStyle = color;

                ctx.fillRect(
                    d.x,
                    labelY - 24,
                    textWidth,
                    25
                );

                ctx.fillStyle = "#111111";

                ctx.fillText(
                    label,
                    d.x + 6,
                    labelY - 6
                );
            }
        }
    }

    requestAnimationFrame(() => drawLoop(mySession));
}


// =====================================
// CONTINUOUS CLOUD INFERENCE
// =====================================

async function detectLoop(mySession) {

    while (running && session === mySession) {

        if (video.readyState < 2) {
            await sleep(300);
            continue;
        }

        const capture = document.createElement("canvas");

        capture.width = canvas.width;
        capture.height = canvas.height;

        capture.getContext("2d").drawImage(
            video,
            0,
            0,
            capture.width,
            capture.height
        );

        const blob = await new Promise(resolve => {
            capture.toBlob(
                resolve,
                "image/jpeg",
                0.65
            );
        });

        if (!blob) {
            await sleep(300);
            continue;
        }

        const form = new FormData();
        form.append("image", blob, "frame.jpg");

        try {
            $("status").textContent =
                "Camera LIVE - AI analyzing...";

            const response = await fetch("/detect", {
                method: "POST",
                headers: {
                    "X-Device-Token": token
                },
                body: form
            });

            const data = await response.json();

            if (!running || session !== mySession) {
                break;
            }

            if (response.status === 401) {
                stopCamera();
                $("error").textContent = "Invalid token";
                break;
            }

            if (!response.ok) {
                detections = [];
                detectionTime = 0;

                $("status").textContent =
                    data.error || "Detection failed";

                await sleep(1500);
                continue;
            }

            detections = data.boxes || [];
            roi = data.roi;
            detectionTime = Date.now();

            if (detections.length) {
                const best = detections[0];

                $("result").textContent =
                    best.class.toUpperCase() + " - " +
                    Math.round(best.confidence * 100) + "%";

            } else {
                $("result").textContent =
                    "No plastic or glass detected";
            }

            $("status").textContent =
                "Camera LIVE - AI connected";

        } catch (error) {
            detections = [];
            detectionTime = 0;

            $("status").textContent =
                "Network error: " + error.message;
        }

        await sleep(DETECTION_DELAY);
    }
}


// =====================================
// DISCONNECT
// =====================================

function stopCamera() {

    running = false;
    session++;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;

    detections = [];
    roi = null;
    token = "";

    $("password").value = "";
    $("cameraPage").style.display = "none";
    $("login").style.display = "block";
}

$("disconnect").onclick = stopCamera;

window.addEventListener("pagehide", () => {
    running = false;
    session++;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
</script>

</body>
</html>
"""


# =========================================
# ROUTES
# =========================================

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "model": "waste_final_v1.tflite",
        "engine": "Ultralytics YOLO",
        "camera": "/camera"
    })


@app.get("/camera")
def camera_page():
    return Response(
        CAMERA_HTML,
        mimetype="text/html",
        headers={
            "Cache-Control": "no-store"
        }
    )


@app.post("/detect")
def detect():
    global latest_result

    if not authorized():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    uploaded = request.files.get("image")

    if uploaded is None:
        return jsonify({
            "error": "No image"
        }), 400

    raw = uploaded.read()

    if not raw:
        return jsonify({
            "error": "Empty image"
        }), 400

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({
            "error": "Invalid image"
        }), 400

    try:
        boxes, roi = detect_bottles(image)

    except Exception:
        app.logger.exception("YOLO inference error")

        return jsonify({
            "error": "YOLO inference failed"
        }), 500

    best = boxes[0] if boxes else None

    result = {
        "detected": bool(boxes),
        "class": best["class"] if best else "none",
        "confidence": (
            best["confidence"] if best else 0.0
        ),
        "boxes": boxes,
        "roi": roi,
        "timestamp": time.time()
    }

    with result_lock:
        latest_result = result

    return jsonify(result)


@app.get("/status")
def status():
    return jsonify({
        "status": "online",
        "engine": "Ultralytics YOLO",
        "model": "waste_final_v1.tflite",
        "allowed_classes": [
            "plastic",
            "glass"
        ]
    })


# Future ESP32 integration
@app.get("/result")
def get_result():

    if not authorized():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    with result_lock:
        result = latest_result.copy()

    if time.time() - result["timestamp"] > 3:
        result.update({
            "detected": False,
            "class": "none",
            "confidence": 0.0,
            "boxes": []
        })

    return jsonify(result)


# =========================================
# START
# =========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False
    )
