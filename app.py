from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import json
import os
import base64
import cv2
from PIL import Image
import io

app = Flask(__name__)

# ── paths ── update these after downloading from Google Drive ──
MODEL_PATH   = os.path.expanduser("~/bird_classification/models/best_model.keras")
CLASSES_PATH = os.path.expanduser("~/bird_classification/models/class_names.json")

model       = None
class_names = None

# ──────────────────────────────────────────────
def load_model():
    global model, class_names
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH} — run in demo mode")

    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            class_names = json.load(f)
        print(f"✅ {len(class_names)} classes loaded")
    else:
        # fallback demo class names
        class_names = [f"Species_{i}" for i in range(525)]


def preprocess_image(pil_img):
    """Resize + EfficientNet preprocessing"""
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    # EfficientNet built-in preprocessing (same as training)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def get_gradcam_heatmap(img_array):
    """Generate Grad-CAM heatmap using the last conv layer of EfficientNetB3"""
    try:
        # Find the last conv layer inside the base model
        base_model = model.layers[1]          # EfficientNetB3 base
        last_conv  = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [base_model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index   = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads       = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0)
        heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def overlay_heatmap(pil_img, heatmap):
    """Blend original image with Grad-CAM heatmap, return base64 PNG"""
    img = np.array(pil_img.convert("RGB").resize((224, 224)))
    h   = cv2.resize(heatmap, (224, 224))
    h   = np.uint8(255 * h)
    h   = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h   = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 0.55, h, 0.45, 0)
    out = Image.fromarray(blended)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── routes ────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file    = request.files["file"]
    pil_img = Image.open(file.stream)
    arr     = preprocess_image(pil_img)

    # Demo mode — model not loaded yet
    if model is None:
        demo_results = [
            {"rank": 1, "species": "American Goldfinch",    "confidence": 0.87, "family": "Finch"},
            {"rank": 2, "species": "Yellow Warbler",         "confidence": 0.07, "family": "Warbler"},
            {"rank": 3, "species": "Common Yellowthroat",    "confidence": 0.03, "family": "Warbler"},
            {"rank": 4, "species": "Wilson's Warbler",       "confidence": 0.02, "family": "Warbler"},
            {"rank": 5, "species": "Pine Warbler",           "confidence": 0.01, "family": "Warbler"},
        ]
        return jsonify({"predictions": demo_results, "gradcam": None, "demo": True})

    preds   = model.predict(arr, verbose=0)[0]
    top5    = np.argsort(preds)[-5:][::-1]

    results = []
    for rank, idx in enumerate(top5, 1):
        raw_name = class_names[idx] if idx < len(class_names) else str(idx)
        # strip leading "001." style prefix if present
        species  = raw_name.split(".")[-1].replace("_", " ").title()
        results.append({
            "rank":       rank,
            "species":    species,
            "confidence": float(preds[idx]),
        })

    # Grad-CAM
    gradcam_b64 = None
    heatmap = get_gradcam_heatmap(arr)
    if heatmap is not None:
        gradcam_b64 = overlay_heatmap(pil_img, heatmap)

    return jsonify({"predictions": results, "gradcam": gradcam_b64, "demo": False})


@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": model is not None,
        "num_classes": len(class_names) if class_names else 0,
    })


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5001)