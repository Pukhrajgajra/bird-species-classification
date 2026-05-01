# Bird Species Classification — Deep Learning

A production-grade bird species classifier built with EfficientNetB3 and 
trained on 84,635 images across 525 species using the iNaturalist dataset.

## Results
| Metric | Score |
|--------|-------|
| Test Accuracy | 94.4% |
| Top-5 Accuracy | 99.39% |
| Species | 525 |
| Training Images | 84,635 |

## Architecture
- **Model**: EfficientNetB3 (transfer learning from ImageNet)
- **Dataset**: 525 Bird Species (84K images)
- **Training**: A100 GPU, 21 epochs, early stopping
- **Framework**: TensorFlow / Keras

## Features
- Real-time bird species identification via Flask web app
- Top-5 predictions with confidence scores
- Grad-CAM explainability heatmaps
- Cosine similarity engine for finding related species
- t-SNE visualization of learned embeddings

## Tech Stack
Python · TensorFlow · EfficientNetB3 · Flask · OpenCV · scikit-learn

## Project Structure