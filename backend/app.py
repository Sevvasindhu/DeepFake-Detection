from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import uuid
import os
import cv2
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Conv2D

# ---------------- CONFIG ----------------
IMG_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.h5")
STATIC_DIR = os.path.join(BASE_DIR, "static")
HEATMAP_DIR = os.path.join(STATIC_DIR, "heatmaps")

os.makedirs(HEATMAP_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- APP ----------------
app = FastAPI(title="DeepFake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------- UTIL ----------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer
    return None

# ---------------- HEATMAP ONLY ----------------
def generate_heatmap_only(img_array, save_path):
    last_conv = get_last_conv_layer(model)
    if last_conv is None:
        return False

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) if np.max(heatmap) != 0 else 1)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.imwrite(save_path, heatmap)
    return True

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"message": "DeepFake API Running"}

# 🔹 Prediction (REAL-TIME)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(HEATMAP_DIR, filename)

    try:
        with open(image_path, "wb") as f:
            f.write(await file.read())

        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]

        if pred > 0.5:
            label = "Fake"
            confidence = pred * 100
        else:
            label = "Real"
            confidence = (1 - pred) * 100

        return {
            "prediction": label,
            "confidence": round(float(confidence), 2),
            "image_id": filename
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🔹 Heatmap ONLY (ON-DEMAND)
@app.get("/heatmap-only/{image_id}")
def heatmap_only(image_id: str):
    image_path = os.path.join(HEATMAP_DIR, image_id)

    try:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        generate_heatmap_only(img, image_path)

        return {"heatmap_url": f"/static/heatmaps/{image_id}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
