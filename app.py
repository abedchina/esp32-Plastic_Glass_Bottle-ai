
import os
import time
import hmac
import threading

import cv2
import numpy as np

from flask import Flask, request, jsonify, Response

# Light TFLite runtime: no PyTorch, so the server fits in Render's free 512MB RAM
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


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

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.65"))
MIN_CONF, MAX_CONF = 0.2, 0.95   # range allowed from the sensitivity slider
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
# LOAD YOLO TFLITE MODEL
# =========================================

class LiteYOLO:
    """YOLOv8 TFLite detector with numpy pre/post-processing.

    Gives the same boxes as Ultralytics: letterbox to the model size,
    then confidence filter and per-class NMS.
    """

    def __init__(self, path):
        self.interpreter = Interpreter(model_path=path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.size = int(self.input["shape"][1])

    def predict(self, bgr, conf, iou, classes):
        h, w = bgr.shape[:2]
        size = self.size

        # Letterbox: keep aspect ratio, pad with grey (114)
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (size - nh) // 2, (size - nw) // 2

        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(
            bgr, (nw, nh), interpolation=cv2.INTER_LINEAR
        )

        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32)[None] / 255.0

        self.interpreter.set_tensor(self.input["index"], x)
        self.interpreter.invoke()

        # (1, 4 + classes, 8400) -> (8400, 4 + classes)
        pred = self.interpreter.get_tensor(self.output["index"])[0].T
        boxes, scores = pred[:, :4], pred[:, 4:]

        # TFLite export gives 0..1 coordinates
        if boxes.max() <= 2.0:
            boxes = boxes * size

        class_ids = scores.argmax(1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        keep = (confidences >= conf) & np.isin(class_ids, classes)
        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        if not len(confidences):
            return []

        # center xywh -> corners, back to original image pixels
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - left) / r
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - top) / r
        x2 = (boxes[:, 0] + boxes[:, 2] / 2 - left) / r
        y2 = (boxes[:, 1] + boxes[:, 3] / 2 - top) / r

        rects = [
            [float(a), float(b), float(c - a), float(d - b)]
            for a, b, c, d in zip(x1, y1, x2, y2)
        ]

        picked = cv2.dnn.NMSBoxesBatched(
            rects,
            confidences.tolist(),
            class_ids.tolist(),
            conf,
            iou
        )

        return [
            (
                int(class_ids[i]),
                float(confidences[i]),
                (x1[i], y1[i], x2[i], y2[i])
            )
            for i in np.array(picked).flatten()
        ]


print("Loading YOLO TFLite model...", flush=True)

model = LiteYOLO(MODEL_PATH)

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
# DETECT BOTTLES
# =========================================

def detect_bottles(frame, conf=CONF_THRESHOLD):

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
        found = model.predict(
            roi_image,
            conf=conf,
            iou=IOU_THRESHOLD,
            classes=SELECTED_CLASS_IDS
        )

    detections = []

    for class_id, confidence, bounds in found:

        label = CLASS_NAMES.get(class_id, "unknown")

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
            "confidence": round(confidence, 3),
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

/* kept on the page (not display:none) so phones keep decoding frames */
#video {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
}

.tools {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    align-items: center;
    max-width: 720px;
    margin: 8px auto;
}

select {
    padding: 10px;
    border-radius: 8px;
    font-size: 15px;
    max-width: 100%;
}

.tools label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #cbd5e1;
}

.tools input[type=range] {
    width: 160px;
    margin: 0;
    padding: 0;
}

#warn {
    color: #fcd34d;
    max-width: 720px;
    margin: 8px auto;
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

    <div class="tools">
        <select id="cameraSelect" aria-label="Camera"></select>
        <button id="restart">Restart camera</button>
        <label>Sensitivity
            <input id="sens" type="range" min="20" max="90" value="65">
            <b id="sensVal">65%</b>
        </label>
    </div>

    <p id="warn"></p>

    <video id="video" autoplay muted playsinline></video>

    <canvas
        id="canvas"
        width="640"
        height="480">
    </canvas>

    <p id="result">Waiting for AI...</p>
    <p style="color:#94a3b8;font-size:14px">Lower sensitivity % = detects more (also weak or blurry items). Keep the bottle inside SORTING ZONE.</p>

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

let cameraId = "";

async function listCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === "videoinput");
        const sel = $("cameraSelect");
        sel.innerHTML = "";
        cams.forEach((c, i) => {
            const o = document.createElement("option");
            o.value = c.deviceId;
            o.textContent = c.label || ("Camera " + (i + 1));
            sel.appendChild(o);
        });
        const current = stream && stream.getVideoTracks()[0].getSettings().deviceId;
        if (current) sel.value = current;
        sel.style.display = cams.length > 1 ? "inline-block" : "none";
    } catch (e) {}
}

async function startCamera() {

    if (running) return;

    $("status").textContent = "Opening camera...";
    $("warn").textContent = "";

    try {
        // same 640x480 framing the sorting zone was tuned for
        const videoConstraints = cameraId
            ? {deviceId: {exact: cameraId}}
            : {facingMode: {ideal: "environment"}};
        videoConstraints.width = {ideal: 640};
        videoConstraints.height = {ideal: 480};

        stream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false
        });

        // ask for continuous autofocus where the camera supports it
        const track = stream.getVideoTracks()[0];
        try {
            const caps = track.getCapabilities ? track.getCapabilities() : {};
            if (caps.focusMode && caps.focusMode.includes("continuous")) {
                await track.applyConstraints({advanced: [{focusMode: "continuous"}]});
            }
        } catch (e) {}

        video.srcObject = stream;
        await video.play();

        // wait until the camera really delivers frames
        for (let i = 0; i < 30 && !video.videoWidth; i++) {
            await sleep(100);
        }

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        running = true;
        session++;

        const mySession = session;

        $("enable").style.display = "none";
        $("status").textContent = "Camera LIVE";

        listCameras();
        drawLoop(mySession);
        detectLoop(mySession);

    } catch (error) {
        $("status").textContent =
            "Camera error: " + error.message;
        $("warn").textContent =
            "Close other apps or tabs that use the camera, then press Enable Camera.";

        $("enable").style.display = "inline-block";
    }
}

function stopStream() {
    running = false;
    session++;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    video.srcObject = null;
}

async function restartCamera() {
    stopStream();
    await sleep(300);
    await startCamera();
}

$("restart").onclick = restartCamera;

$("cameraSelect").onchange = () => {
    cameraId = $("cameraSelect").value;
    restartCamera();
};

$("sens").oninput = () => {
    $("sensVal").textContent = $("sens").value + "%";
};

// a black picture usually means another tab or app holds the camera
let darkSince = 0;

function checkDark() {
    const probe = document.createElement("canvas");
    probe.width = 32;
    probe.height = 24;
    const p = probe.getContext("2d");
    p.drawImage(video, 0, 0, 32, 24);
    const px = p.getImageData(0, 0, 32, 24).data;
    let sum = 0;
    for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
    const dark = sum / (px.length / 4) / 3 < 8;
    if (!dark) { darkSince = 0; $("warn").textContent = ""; return; }
    if (!darkSince) darkSince = Date.now();
    if (Date.now() - darkSince > 3000) {
        $("warn").textContent =
            "The camera picture is black. Close other tabs/apps that use the camera, then press Restart camera (or pick another camera).";
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
        if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }
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
                0.85
            );
        });

        if (!blob) {
            await sleep(300);
            continue;
        }

        const form = new FormData();
        form.append("image", blob, "frame.jpg");
        form.append("conf", String($("sens").value / 100));

        checkDark();

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

            let data;

            try {
                data = await response.json();
            } catch {
                // Render sends an HTML page while the server wakes up or restarts
                detections = [];
                detectionTime = 0;
                $("status").textContent =
                    "Server is starting... retrying (" + response.status + ")";
                await sleep(2000);
                continue;
            }

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

                // also show the other material's best score, so close calls are visible
                const other = detections.find(d => d.class !== best.class);

                $("result").textContent =
                    best.class.toUpperCase() + " - " +
                    Math.round(best.confidence * 100) + "%" +
                    (other
                        ? "  (" + other.class + " " +
                          Math.round(other.confidence * 100) + "%)"
                        : "");

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

    stopStream();

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
        "engine": "YOLO TFLite (LiteRT)",
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
        conf = float(request.form.get("conf", CONF_THRESHOLD))
    except ValueError:
        conf = CONF_THRESHOLD
    conf = min(MAX_CONF, max(MIN_CONF, conf))

    try:
        boxes, roi = detect_bottles(image, conf)

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
        "engine": "YOLO TFLite (LiteRT)",
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
