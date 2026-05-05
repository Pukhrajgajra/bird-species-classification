import os
import io
import json
import base64

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B3_Weights
from PIL import Image
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── PATHS — update if needed ──────────────────────────────
MODEL_PATH   = os.path.expanduser("~/bird_classification/models/best_model.pth")
CLASSES_PATH = os.path.expanduser("~/bird_classification/models/class_names.json")
# MODEL_PATH   = r'D:\bird525_model\best_model.pth'
# CLASSES_PATH = r'D:\bird525_model\class_names.json'

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model       = None
class_names = None

# ── BUILD MODEL ARCHITECTURE ──────────────────────────────
def build_model(num_classes):
    m = models.efficientnet_b3(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Identity(),
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m

# ── LOAD MODEL ────────────────────────────────────────────
def load_model():
    global model, class_names

    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            class_names = json.load(f)
        print(f" {len(class_names)} classes loaded")
    else:
        class_names = [f"Species_{i}" for i in range(525)]
        print(" class_names.json not found — using fallback names")

    if os.path.exists(MODEL_PATH):
        model = build_model(len(class_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f" Model loaded from {MODEL_PATH} on {DEVICE}")
    else:
        print(f" Model not found at {MODEL_PATH} — running in demo mode")

# ── PREPROCESSING ─────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_image(pil_img):
    img = pil_img.convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    return tensor

# ── GRAD-CAM ──────────────────────────────────────────────
def get_gradcam_heatmap(tensor):
    """Generate Grad-CAM heatmap from the last conv block of EfficientNetB3."""
    try:
        target_layer = model.features[-1]  # last feature block
        activations, gradients = [], []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        output = model(tensor)
        pred_class = output.argmax(dim=1).item()
        model.zero_grad()
        output[0, pred_class].backward()

        fh.remove()
        bh.remove()

        act  = activations[0].squeeze(0)   # [C, H, W]
        grad = gradients[0].squeeze(0)     # [C, H, W]
        weights = grad.mean(dim=(1, 2))    # [C]
        heatmap = (weights[:, None, None] * act).sum(dim=0)
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def overlay_heatmap(pil_img, heatmap):
    """Blend original image with Grad-CAM heatmap, return base64 PNG."""
    img = np.array(pil_img.convert('RGB').resize((300, 300)))
    h   = cv2.resize(heatmap, (300, 300))
    h   = np.uint8(255 * h)
    h   = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h   = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 0.55, h, 0.45, 0)
    out = Image.fromarray(blended)
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ── ROUTES ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file    = request.files['file']
    pil_img = Image.open(file.stream)

    # Demo mode
    if model is None:
        demo_results = [
            {'rank': 1, 'species': 'American Goldfinch', 'confidence': 0.87},
            {'rank': 2, 'species': 'Yellow Warbler',      'confidence': 0.07},
            {'rank': 3, 'species': 'Common Yellowthroat', 'confidence': 0.03},
            {'rank': 4, 'species': "Wilson's Warbler",    'confidence': 0.02},
            {'rank': 5, 'species': 'Pine Warbler',        'confidence': 0.01},
        ]
        return jsonify({'predictions': demo_results, 'gradcam': None, 'demo': True})

    tensor = preprocess_image(pil_img)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    top5_probs, top5_indices = probs.topk(5)

    results = []
    for rank, (idx, prob) in enumerate(zip(top5_indices.tolist(), top5_probs.tolist()), 1):
        raw_name = class_names[idx] if idx < len(class_names) else str(idx)
        species  = raw_name.split('.')[-1].replace('_', ' ').title()
        results.append({
            'rank':       rank,
            'species':    species,
            'confidence': round(prob, 4),
        })

    # Grad-CAM (needs grad so run separately)
    gradcam_b64 = None
    try:
        tensor_grad = preprocess_image(pil_img)
        heatmap = get_gradcam_heatmap(tensor_grad)
        if heatmap is not None:
            gradcam_b64 = overlay_heatmap(pil_img, heatmap)
    except Exception as e:
        print(f"Grad-CAM skipped: {e}")

    return jsonify({'predictions': results, 'gradcam': gradcam_b64, 'demo': False})

@app.route('/health')
def health():
    return jsonify({
        'status':      'ok',
        'model_ready': model is not None,
        'device':      str(DEVICE),
        'num_classes': len(class_names) if class_names else 0,
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5001)