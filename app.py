
import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, jsonify
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

# =====================================
# SETTINGS
# =====================================

MODEL_PATH = "waste_final_v1.tflite"

CLASS_NAMES = ["plastic", "glass"]
CONF_THRESHOLD = 0.65

CAMERA_URL = os.getenv(
    "CAMERA_URL",
    "rtsp://127.0.0.1:8554/live"
)

DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", "")

latest_result = {
    "detected": False,
    "class": "none",
    "confidence": 0.0,
    "timestamp": 0
}

lock = threading.Lock()

# =====================================
# LOAD MODEL
# =====================================

interpreter = None
input_details = None
output_details = None


def load_model():
    global interpreter, input_details, output_details

    interpreter = Interpreter(
        model_path=MODEL_PATH,
        num_threads=2
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("TFLite model loaded")
    print("Input:", input_details["shape"])
    print("Output:", output_details["shape"])


# =====================================
# DETECTION
# =====================================

def detect(frame):
    shape = input_details["shape"]
    height = int(shape[1])
    width = int(shape[2])

    image = cv2.resize(frame, (width, height))
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    dtype = input_details["dtype"]

    if dtype == np.float32:
        tensor = (
            image.astype(np.float32) / 255.0
        )
    else:
        scale, zero = input_details["quantization"]

        if scale > 0:
            tensor = np.round(
                (image / 255.0) / scale + zero
            )
            tensor = np.clip(
                tensor,
                np.iinfo(dtype).min,
                np.iinfo(dtype).max
            ).astype(dtype)
        else:
            tensor = image.astype(dtype)

    tensor = np.expand_dims(tensor, axis=0)

    interpreter.set_tensor(
        input_details["index"], tensor
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details["index"]
    )

    # Dequantize output if needed
    scale, zero = output_details["quantization"]
    if scale > 0:
        output = (
            output.astype(np.float32) - zero
        ) * scale

    # Standard raw YOLOv8 output
    predictions = np.squeeze(output)

    if predictions.ndim != 2:
        raise ValueError(
            "Unsupported output shape: " + str(output.shape)
        )

    num_channels = 4 + len(CLASS_NAMES)

    if predictions.shape[0] == num_channels:
        predictions = predictions.T

    if predictions.shape[1] != num_channels:
        raise ValueError(
            "Model output does not match class count: "
            + str(output.shape)
        )

    scores = predictions[:, 4:]
    best_indices = np.argmax(scores, axis=1)
    best_scores = np.max(scores, axis=1)

    if len(best_scores) == 0:
        return "none", 0.0

    idx = int(np.argmax(best_scores))
    confidence = float(best_scores[idx])

    if confidence < CONF_THRESHOLD:
        return "none", confidence

    label = CLASS_NAMES[int(best_indices[idx])]
    return label, confidence


# =====================================
# CAMERA WORKER
# =====================================

def camera_worker():
    global latest_result

    while True:
        print("Connecting to camera...")

        cap = cv2.VideoCapture(
            CAMERA_URL,
            cv2.CAP_FFMPEG
        )

        if not cap.isOpened():
            print("Camera unavailable")
            time.sleep(5)
            continue

        print("Camera connected")

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            try:
                label, confidence = detect(frame)

                with lock:
                    latest_result = {
                        "detected": label != "none",
                        "class": label,
                        "confidence": round(confidence, 3),
                        "timestamp": time.time()
                    }

            except Exception as e:
                print("Detection error:", e)
                time.sleep(2)
                break

            time.sleep(0.3)

        cap.release()
        time.sleep(3)


# =====================================
# API
# =====================================

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Plastic Glass Bottle AI"
    })


@app.get("/status")
def status():
    with lock:
        return jsonify({
            "model_loaded": interpreter is not None,
            "last_detection": latest_result["timestamp"]
        })


@app.get("/result")
def result():
    from flask import request

    token = request.headers.get("X-Device-Token", "")

    if not DEVICE_TOKEN or token != DEVICE_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    with lock:
        data = latest_result.copy()

    if time.time() - data["timestamp"] > 3:
        data["detected"] = False
        data["class"] = "none"
        data["confidence"] = 0.0

    return jsonify(data)


# =====================================
# START
# =====================================

if __name__ == "__main__":
    load_model()

    thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )
    thread.start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
