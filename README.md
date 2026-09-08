# NeuroScan: MRI Brain Tumor Classifier

Live App: **https://mri-brain-tumor-classifier.onrender.com/**

NeuroScan is a FastAPI-based web app that classifies brain MRI scans into 4 classes and returns visual explanations.

## What it does
- Upload a brain MRI image
- Predict one class: `glioma`, `meningioma`, `pituitary`, or `notumor`
- Generate explainability outputs:
  - Grad-CAM++ overlay
  - LIME explanation image

## Live Usage
1. Open the hosted app: https://mri-brain-tumor-classifier.onrender.com/
2. Upload an MRI image (JPEG/PNG/BMP/TIFF/WebP)
3. Click **Analyze Scan**
4. View the diagnosis and explanation images

## API
Base URL (hosted): `https://mri-brain-tumor-classifier.onrender.com`

- `GET /health` — service and model readiness
- `POST /predict` — upload image using `multipart/form-data` with field name `file`
- Swagger docs: `/api/docs`

## Run Locally
```bash
git clone https://github.com/imtiazdeepto/MRI-Brain-Tumor-Classifier.git
cd MRI-Brain-Tumor-Classifier
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Local URLs:
- App: `http://localhost:8000/`
- API Docs: `http://localhost:8000/api/docs`

## Project Structure
```text
MRI-Brain-Tumor-Classifier/
├── main.py
├── inference.py
├── KD_T6.0_a0.8_latest.pth
├── requirements.txt
├── render.yaml
└── static/
```

## Dataset
- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Disclaimer
This tool is for research and educational use only and is not a clinical diagnostic system.
