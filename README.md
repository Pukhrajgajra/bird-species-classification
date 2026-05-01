# 🦅 Bird Species Classification — 525 Species, 84K Images

A deep learning pipeline achieving **94.4% validation accuracy** on 525 bird species using transfer learning with EfficientNetB3, trained on 84,000+ images.

## Results
| Metric | Score |
| Validation Accuracy | 94.4% |
| Top-5 Accuracy | 99.5% |
| Species Classified | 525 |
| Training Images | 84,000+ |

## Features
- **EfficientNetB3** transfer learning with fine-tuning
- **Grad-CAM** heatmaps for explainability — visualizes which parts of the bird the model focuses on
- **t-SNE** visualization of 84K image embeddings
- **Cosine similarity** engine to find visually similar species
- **Flask REST API** — upload any bird photo and get top-5 predictions
- Multi-model benchmark: EfficientNetB3 vs ResNet50 vs MobileNetV2

## Tech Stack
Python · TensorFlow · EfficientNetB3 · OpenCV · scikit-learn · Flask · Google Colab (A100 GPU)

## Project Structure
