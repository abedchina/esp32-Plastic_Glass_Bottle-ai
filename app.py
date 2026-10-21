
import os
import time
import hmac
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'waste_final_v1.tflite'
)

CLASS_NAMES = ['plastic', 'paper', 'metal', 'glass']
ALLOWED_CLASSES = {'plastic', 'glass'}

CONF_THRESHOLD = 0.65
NMS_THRESHOLD = 0.45

# Same sorting area as the original PC code
ARRIVAL_ROI = (0.25, 0.208333, 0.78125, 0.875)

DEVICE_TOKEN = os.getenv('DEVICE_TOKEN', '')

model_lock = threading.Lock()
result_lock = threading.Lock()

latest_result = {
    'detected': False,
    'class': 'none',
    'confidence': 0.0,
    'boxes': [],
    'timestamp': 0
}

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

print(
    'MODEL LOADED',
    input_info['shape'],
    output_info['shape'],
    flush=True
)


# ==========================================
# AUTHENTICATION
# ==========================================

def authorized():
    value = request.headers.get('X-Device-Token', '')

    return (
        bool(DEVICE_TOKEN)
        and hmac.compare_digest(value, DEVICE_TOKEN)
    )


# ==========================================
# LETTERBOX PREPROCESSING
# ==========================================

def letterbox(image, size):
    """
    Keep aspect ratio and add gray padding,
    similar to Ultralytics YOLO.
    """

    src_h, src_w = image.shape[:2]
    dst_w, dst_h = size

    scale = min(
        dst_w / src_w,
        dst_h / src_h
    )

    resized_w = round(src_w * scale)
    resized_h = round(src_h * scale)

    dw = (dst_w - resized_w) / 2
    dh = (dst_h - resized_h) / 2

    image = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR
    )

    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return image, scale, left, top


# ==========================================
# DETECTION
# ==========================================

def detection_on_frame(frame):

    frame_h, frame_w = frame.shape[:2]

    rx1 = int(ARRIVAL_ROI[0] * frame_w)
    ry1 = int(ARRIVAL_ROI[1] * frame_h)
    rx2 = int(ARRIVAL_ROI[2] * frame_w)
    ry2 = int(ARRIVAL_ROI[3] * frame_h)

    crop = frame[ry1:ry2, rx1:rx2]

    if crop.size == 0:
        raise ValueError('ROI empty')

    input_shape = input_info['shape']

    mh = int(input_shape[1])
    mw = int(input_shape[2])

    image, ratio, pad_x, pad_y = letterbox(
        crop,
        (mw, mh)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    dtype = input_info['dtype']

    if np.issubdtype(dtype, np.floating):

        tensor = image.astype(dtype) / 255.0

    else:

        scale, zero = input_info['quantization']

        if scale <= 0:
            raise ValueError(
                'Unsupported input quantization'
            )

        tensor = np.rint(
            (image.astype(np.float32) / 255.0)
            / scale + zero
        )

        info = np.iinfo(dtype)

        tensor = np.clip(
            tensor,
            info.min,
            info.max
        ).astype(dtype)

    tensor = np.expand_dims(
        tensor,
        axis=0
    )

    # ======================================
    # RUN TFLITE MODEL
    # ======================================

    with model_lock:

        interpreter.set_tensor(
            input_info['index'],
            tensor
        )

        interpreter.invoke()

        raw = interpreter.get_tensor(
            output_info['index']
        )

    scale, zero = output_info['quantization']

    if scale > 0:

        raw = (
            raw.astype(np.float32) - zero
        ) * scale

    predictions = np.squeeze(raw)

    channels = 4 + len(CLASS_NAMES)

    if predictions.ndim != 2:
        raise ValueError(
            'Unexpected output shape: {}'.format(
                raw.shape
            )
        )

    if predictions.shape[0] == channels:
        predictions = predictions.T

    if predictions.shape[1] != channels:
        raise ValueError(
            'Expected {} channels but got {}'.format(
                channels,
                raw.shape
            )
        )

    # ======================================
    # FILTER PLASTIC AND GLASS
    # ======================================

    classes = predictions[:, 4:]

    class_ids = np.argmax(
        classes,
        axis=1
    )

    confidence = np.max(
        classes,
        axis=1
    )

    accepted_ids = [
        i for i, name in enumerate(CLASS_NAMES)
        if name in ALLOWED_CLASSES
    ]

    keep = (
        (confidence >= CONF_THRESHOLD)
        & np.isin(class_ids, accepted_ids)
    )

    selected = predictions[keep]
    selected_class_ids = class_ids[keep]
    selected_confidence = confidence[keep]

    boxes = []
    scores = []
    ids = []

    crop_h, crop_w = crop.shape[:2]

    # ======================================
    # CONVERT BOX COORDINATES
    # ======================================

    for prediction, cls_id, conf in zip(
        selected,
        selected_class_ids,
        selected_confidence
    ):

        cx, cy, bw, bh = map(
            float,
            prediction[:4]
        )

        x1 = (cx - bw / 2 - pad_x) / ratio
        y1 = (cy - bh / 2 - pad_y) / ratio

        x2 = (cx + bw / 2 - pad_x) / ratio
        y2 = (cy + bh / 2 - pad_y) / ratio

        x1 = max(0, min(crop_w, x1))
        y1 = max(0, min(crop_h, y1))

        x2 = max(0, min(crop_w, x2))
        y2 = max(0, min(crop_h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            int(x1 + rx1),
            int(y1 + ry1),
            int(x2 - x1),
            int(y2 - y1)
        ])

        scores.append(float(conf))
        ids.append(int(cls_id))

    # ======================================
    # NON MAXIMUM SUPPRESSION
    # ======================================

    objects = []

    for cls_id in set(ids):

        positions = [
            i for i, c in enumerate(ids)
            if c == cls_id
        ]

        grouped = [
            boxes[i] for i in positions
        ]

        group_scores = [
            scores[i] for i in positions
        ]

        indices = cv2.dnn.NMSBoxes(
            grouped,
            group_scores,
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        for item in np.array(indices).reshape(-1):

            j = positions[int(item)]

            x, y, w, h = boxes[j]

            objects.append({
                'class': CLASS_NAMES[cls_id],
                'confidence': round(scores[j], 3),
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })

    objects.sort(
        key=lambda d: d['confidence'],
        reverse=True
    )

    roi = [
        rx1,
        ry1,
        rx2 - rx1,
        ry2 - ry1
    ]

    return objects[:10], roi


# ==========================================
# LIVE CAMERA WEBSITE
# ==========================================

CAMERA_HTML = r"""
<!doctype html>
<html lang="en">

<head>
<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>Plastic & Glass Bottle AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    background: #0f172a;
    color: #fff;
    text-align: center;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 14px;
}

#login {
    max-width: 380px;
    margin: 12vh auto;
    background: #1e293b;
    padding: 24px;
    border-radius: 14px;
}

input, button {
    padding: 13px;
    font-size: 17px;
    border-radius: 8px;
    border: 0;
    margin: 8px 0;
}

input {
    width: 100%;
}

button {
    background: #22c55e;
    font-weight: bold;
    cursor: pointer;
}

#camera {
    display: none;
}

canvas {
    width: 100%;
    max-width: 720px;
    height: auto;
    background: #000;
    border: 2px solid #22c55e;
    border-radius: 10px;
}

video {
    display: none;
}

#result {
    font-size: 23px;
    font-weight: bold;
    color: #86efac;
}

#status {
    color: #cbd5e1;
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

    <p>Enter password before opening the camera</p>

    <input
        id="password"
        type="password"
        placeholder="Device Token"
        autocomplete="off"
    >

    <button id="connect">
        Connect
    </button>

    <p id="error"></p>

</div>


<div id="camera">

    <h2>Live Bottle Detection</h2>

    <p id="status">Opening camera...</p>

    <button id="enable">
        Enable Camera
    </button>

    <video
        id="video"
        autoplay
        muted
        playsinline>
    </video>

    <canvas
        id="canvas"
        width="640"
        height="480">
    </canvas>

    <p id="result">
        Waiting for AI...
    </p>

    <button id="disconnect">
        Disconnect
    </button>

</div>


<script>

const $ = id => document.getElementById(id);

const video = $('video');
const canvas = $('canvas');

const ctx = canvas.getContext('2d');

let token = '';
let stream = null;
let running = false;

let current = 0;
let boxes = [];
let roi = null;
let updated = 0;

const sleep = ms =>
    new Promise(r => setTimeout(r, ms));


// ======================================
// LOGIN
// ======================================

$('connect').onclick = async () => {

    const candidate = $('password').value.trim();

    if (!candidate) {
        $('error').textContent = 'Enter Device Token';
        return;
    }

    $('connect').disabled = true;

    $('error').textContent = 'Checking password...';

    try {

        const response = await fetch('/auth', {
            method: 'POST',
            headers: {
                'X-Device-Token': candidate
            }
        });

        if (!response.ok) {
            $('error').textContent = 'Incorrect password';
            return;
        }

        token = candidate;

        $('password').value = '';

        $('login').style.display = 'none';
        $('camera').style.display = 'block';

        await startCamera();

    } catch (e) {

        $('error').textContent =
            'Connection problem: ' + e.message;

    } finally {

        $('connect').disabled = false;
    }
};


$('password').addEventListener(
    'keydown',
    e => {
        if (e.key === 'Enter') {
            $('connect').click();
        }
    }
);


// ======================================
// START LIVE CAMERA
// ======================================

async function startCamera() {

    if (running) return;

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: {
                    ideal: 'environment'
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

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        running = true;
        current++;

        $('enable').style.display = 'none';
        $('status').textContent = 'Live camera';

        drawLoop(current);
        detectLoop(current);

    } catch (e) {

        $('status').textContent =
            'Camera permission required: ' + e.message;

        $('enable').style.display = 'inline-block';
    }
}

$('enable').onclick = startCamera;


// ======================================
// DRAW VIDEO + SORTING ZONE + BOXES
// ======================================

function drawLoop(id) {

    if (!running || id !== current) {
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

        // Sorting zone rectangle
        if (roi) {

            ctx.strokeStyle = '#00d5ff';
            ctx.lineWidth = 2;

            ctx.strokeRect(...roi);

            ctx.font = 'bold 14px Arial';
            ctx.fillStyle = '#00d5ff';

            ctx.fillText(
                'SORTING ZONE',
                roi[0],
                Math.max(16, roi[1] - 7)
            );
        }

        // Detection boxes
        if (Date.now() - updated < 2500) {

            for (const d of boxes) {

                const color =
                    d.class === 'plastic'
                        ? '#22c55e'
                        : '#ffad22';

                ctx.lineWidth = 4;
                ctx.strokeStyle = color;

                ctx.strokeRect(
                    d.x,
                    d.y,
                    d.w,
                    d.h
                );

                const label =
                    d.class + ' ' +
                    Math.round(d.confidence * 100) + '%';

                ctx.font = 'bold 17px Arial';

                const y = Math.max(23, d.y);

                const tw =
                    ctx.measureText(label).width + 12;

                ctx.fillStyle = color;

                ctx.fillRect(
                    d.x,
                    y - 23,
                    tw,
                    24
                );

                ctx.fillStyle = '#111';

                ctx.fillText(
                    label,
                    d.x + 6,
                    y - 6
                );
            }
        }
    }

    requestAnimationFrame(
        () => drawLoop(id)
    );
}


// ======================================
// CONTINUOUS CLOUD DETECTION
// ======================================

async function detectLoop(id) {

    while (running && id === current) {

        const shot = document.createElement('canvas');

        shot.width = canvas.width;
        shot.height = canvas.height;

        shot.getContext('2d').drawImage(
            video,
            0,
            0,
            shot.width,
            shot.height
        );

        const blob = await new Promise(
            r => shot.toBlob(
                r,
                'image/jpeg',
                0.65
            )
        );

        if (!blob) {
            await sleep(300);
            continue;
        }

        const form = new FormData();

        form.append(
            'image',
            blob,
            'frame.jpg'
        );

        try {

            $('status').textContent =
                'Camera LIVE - analyzing...';

            const resp = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'X-Device-Token': token
                },
                body: form
            });

            const data = await resp.json();

            if (!running || id !== current) {
                break;
            }

            if (resp.status === 401) {

                stop();

                $('error').textContent =
                    'Incorrect or expired token';

                break;
            }

            if (!resp.ok) {

                boxes = [];

                $('status').textContent =
                    data.error || ('Error ' + resp.status);

                await sleep(1200);
                continue;
            }

            boxes = data.boxes || [];
            roi = data.roi;

            updated = Date.now();

            if (boxes.length) {

                $('result').textContent =
                    boxes[0].class.toUpperCase()
                    + ' - '
                    + Math.round(
                        boxes[0].confidence * 100
                    )
                    + '%';

            } else {

                $('result').textContent =
                    'No plastic or glass detected';
            }

            $('status').textContent =
                'Camera LIVE - AI connected';

        } catch (e) {

            boxes = [];

            $('status').textContent =
                'Connection error: ' + e.message;
        }

        await sleep(700);
    }
}


// ======================================
// DISCONNECT
// ======================================

function stop() {

    running = false;
    current++;

    if (stream) {

        stream.getTracks().forEach(
            t => t.stop()
        );

        stream = null;
    }

    video.srcObject = null;

    boxes = [];
    roi = null;
    token = '';

    $('camera').style.display = 'none';
    $('login').style.display = 'block';

    $('password').value = '';
}

$('disconnect').onclick = stop;

window.addEventListener(
    'pagehide',
    () => {
        if (stream) {
            stream.getTracks().forEach(
                t => t.stop()
            );
        }
    }
);

</script>

</body>
</html>
"""


# ==========================================
# FLASK ROUTES
# ==========================================

@app.get('/')
def home():

    return jsonify({
        'status': 'online',
        'camera': '/camera',
        'model': 'waste_final_v1.tflite'
    })


@app.get('/camera')
def camera_page():

    return Response(
        CAMERA_HTML,
        mimetype='text/html',
        headers={
            'Cache-Control': 'no-store'
        }
    )


@app.post('/auth')
def auth():

    if not authorized():

        return jsonify({
            'authorized': False
        }), 401

    return jsonify({
        'authorized': True
    })


@app.post('/detect')
def detect():

    global latest_result

    if not authorized():

        return jsonify({
            'error': 'Unauthorized'
        }), 401

    picture = request.files.get('image')

    if picture is None:

        return jsonify({
            'error': 'No image'
        }), 400

    data = np.frombuffer(
        picture.read(),
        np.uint8
    )

    frame = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({
            'error': 'Invalid image'
        }), 400

    try:

        boxes, roi = detection_on_frame(frame)

    except Exception:

        app.logger.exception(
            'Detection failed'
        )

        return jsonify({
            'error': 'Detection failed - check Render Logs'
        }), 500

    best = boxes[0] if boxes else None

    result = {
        'detected': bool(boxes),
        'class': (
            best['class'] if best else 'none'
        ),
        'confidence': (
            best['confidence'] if best else 0.0
        ),
        'boxes': boxes,
        'roi': roi,
        'timestamp': time.time()
    }

    with result_lock:
        latest_result = result

    return jsonify(result)


@app.get('/status')
def status():

    return jsonify({
        'online': True,
        'input_shape': input_info['shape'].tolist(),
        'output_shape': output_info['shape'].tolist(),
        'allowed_classes': [
            'plastic',
            'glass'
        ]
    })


@app.get('/result')
def get_result():

    if not authorized():

        return jsonify({
            'error': 'Unauthorized'
        }), 401

    with result_lock:
        data = latest_result.copy()

    if time.time() - data['timestamp'] > 3:

        data.update({
            'detected': False,
            'class': 'none',
            'confidence': 0.0,
            'boxes': []
        })

    return jsonify(data)


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', '10000')),
        debug=False
    )
