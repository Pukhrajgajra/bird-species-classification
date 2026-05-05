#  Bird Species Classification — Deep Learning

A production-grade bird species classifier built with **PyTorch** and **EfficientNetB3**,
trained on **84,635 images** across **525 species**.

## Results
| Metric | Score |
|--------|-------|
| Test Accuracy | 85% |
| Top-5 Accuracy | 97% |
| Species Classified | 525 |
| Training Images | 84,635 |
| Architecture | EfficientNetB3 |
| Framework | PyTorch |

## What It Does
- Identifies 525 bird species from a single uploaded photo
- Returns top-5 predictions with confidence scores
- Generates **Grad-CAM heatmaps** showing which part of the bird the model focused on
- Finds visually similar species using cosine similarity on learned embeddings
- t-SNE visualization of all 84K image embeddings
- Flask web app — upload any bird photo and get instant predictions

## Tech Stack

Python · PyTorch · EfficientNetB3 · Flask · OpenCV · scikit-learn · torchvision · NumPy

## How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the web app:**
```bash
python app.py
```
Open **http://localhost:5001** in your browser, upload any bird photo.

**3. Train from scratch:**
```bash
python src/train.py
```

## Model Architecture

Input Image (300×300×3)

↓

EfficientNetB3 Backbone (ImageNet pretrained)

↓

GlobalAveragePooling

↓

Dropout(0.5) → Linear(in_features → 1024) → BatchNorm1d → ReLU

↓

Dropout(0.4) → Linear(1024 → 512) → ReLU

↓

Dropout(0.3) → Linear(512 → 525)

↓

Softmax → Top-5 Predictions

## Training Strategy
- **Phase 1** — Freeze EfficientNetB3 backbone, train only classifier head
- **Phase 2** — Unfreeze top layers, fine-tune with low learning rate (2e-6)
- Early stopping with patience=8 to prevent overfitting
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Data augmentation: random rotation, flips, zoom, brightness, shear

## Key Technical Decisions
- **EfficientNetB3** over ResNet50 — better accuracy per parameter
- **Grad-CAM** for explainability — shows exactly which pixels drove the prediction
- **Top-5 accuracy 97%** — correct species is always in the top 5 predictions
- **Two-phase training** — prevents catastrophic forgetting of ImageNet features
- **PyTorch** — flexible, production-ready, industry standard

## Requirements

torch>=2.0.0


torchvision>=0.15.0

flask>=3.0.0

pillow>=10.0.0

numpy>=1.24.0

opencv-python>=4.8.0

scikit-learn>=1.3.0

matplotlib>=3.7.0

seaborn>=0.12.0

## Sample Predictions
Upload any bird photo → model returns:

American Robin        92.3%  ████████████████████

Hermit Thrush          4.1%  ████

Swainson's Thrush      1.8%  ██

Wood Thrush            1.2%  █

Veery                  0.6%  █

Plus a Grad-CAM heatmap showing which features the model used.