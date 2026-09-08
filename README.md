# LiteBrainNet: MRI Brain Tumor Classifier

A professional, lightweight deep learning project for classifying brain MRI scans into four classes:

- glioma
- meningioma
- notumor
- pituitary

This project combines:
- a compact custom CNN model
- knowledge distillation from a stronger teacher model
- explainability outputs (Grad-CAM++ and LIME)
- a FastAPI web application for practical inference

## Project Highlights

- Lightweight student model (`CustomCNN5_Brain`) for efficient deployment
- Distillation-ready pipeline for accuracy vs. model-size tradeoff
- Grad-CAM++ heatmap generation for visual explanation
- LIME explanation overlay for interpretable predictions
- Simple web interface served by FastAPI
- Production-friendly setup (`render.yaml`)

## Tech Stack

- Python
- PyTorch
- FastAPI
- OpenCV
- NumPy
- Pillow
- LIME / scikit-image

## Repository Structure

```text
MRI-Brain-Tumor-Classifier/
├── main.py                         # FastAPI app and routes
├── inference.py                    # Model, preprocessing, Grad-CAM++, LIME
├── KD_T6.0_a0.8_latest.pth         # Trained model weights
├── requirements.txt                # Python dependencies
├── render.yaml                     # Deployment configuration
├── static/
│   ├── index.html                  # Frontend page
│   ├── style.css                   # Frontend styles
│   └── script.js                   # Frontend logic
└── README.md
```

## Installation

```bash
git clone https://github.com/imtiazdeepto/MRI-Brain-Tumor-Classifier.git
cd MRI-Brain-Tumor-Classifier

python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- App: `http://localhost:8000/`
- API docs: `http://localhost:8000/api/docs`

## API Endpoints

### `GET /health`
Returns service and model readiness.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Upload one image (`multipart/form-data`, field name: `file`) and receive prediction with explainability images.

Example response:

```json
{
  "predicted_class": "meningioma",
  "heatmap_image": "<base64_png>",
  "lime_image": "<base64_png>"
}
```

## How It Works

1. Upload image
2. Preprocess (decode → RGB → contour-based crop → resize to `240x240` → normalize)
3. Predict class using `CustomCNN5_Brain`
4. Generate Grad-CAM++ and LIME explanations
5. Return result to frontend/API client

## Dataset

- Brain Tumor MRI Dataset:  
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Notes

- Designed for research and educational use.
- Not a replacement for clinical diagnosis.

## License

This project is released under the MIT License.
