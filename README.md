# 🧠 LiteBrainNet — Lightweight Brain Tumor MRI Classification via Knowledge Distillation

> A lightweight, explainable AI system for four-class brain tumor MRI classification, compressed via Knowledge Distillation from an EfficientNet-B1 teacher to a custom CNN student, with INT8 quantization and clinical web deployment.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Research Background](#-research-background)
- [Citation](#-citation)
- [License](#-license)

---

## Overview

This project addresses the gap between high-accuracy deep learning models for brain tumor classification and their deployability in low-resource clinical settings. We distill knowledge from a heavyweight **EfficientNet-B1** teacher into a compact **CustomCNN5_Brain** student (~500K parameters), achieving competitive accuracy with a **~16× parameter reduction**. The distilled model is further compressed via **INT8 post-training dynamic quantization** for CPU-only deployment.

A **FastAPI**-based web application provides an accessible, browser-based diagnostic interface where clinicians can upload brain MRI scans, receive classification results with confidence scores, and view **Grad-CAM++** attention overlays explaining the model's decision.

**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

**Classes:**
- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

---

## 🌟 Key Features

- **Knowledge Distillation (KD):** Transfers dark knowledge from an ImageNet-pretrained EfficientNet-B1 teacher into a lightweight custom CNN student.
- **Hyperparameter Ablation:** Systematic evaluation across temperature `T ∈ {3.0, 5.0, 6.0}` and distillation weight `α ∈ {0.5, 0.7, 0.8}`.
- **Architecture Ablation:** Comparative analysis of CustomCNN3, CustomCNN4, and CustomCNN5 student depths.
- **INT8 Quantization:** Post-training dynamic quantization targeting `nn.Linear` layers for reduced model size and faster CPU inference.
- **Explainable AI (XAI):**
  - **Grad-CAM++:** Pixel-level heatmap highlighting tumor regions from the last convolutional layer.
  - **LIME:** Superpixel-based boundary explanations for model-agnostic interpretability.
- **Clinical Web Interface:** FastAPI backend with a responsive HTML/CSS frontend for real-time diagnosis.
- **Comprehensive Benchmarking:** FLOPs, parameter count, RAM usage, CPU/GPU latency, and throughput measurements.
- **Baseline Comparisons:** No-KD CNN5, MobileNetV2, and ShuffleNetV2 trained on the same split for rigorous evaluation.

---

## 🏗️ Architecture

### Teacher Network: EfficientNet-B1
| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B1 (ImageNet pretrained) |
| Parameters | ~7.8M |
| Input Size | 240 × 240 × 3 |
| Output | 4 classes |
| Training Loss | Class-weighted Cross-Entropy |
| Optimizer | Adam (lr = 0.001) |
| Scheduler | ReduceLROnPlateau (factor = 0.1, patience = 5) |

### Student Network: CustomCNN5_Brain (LiteBrainNet)
| Property | Value |
|----------|-------|
| Architecture | 5-block custom CNN |
| Parameters | ~500K |
| Input Size | 240 × 240 × 3 |
| Conv Blocks | 5 × [Conv3×3 → ReLU → MaxPool2×2] |
| Channels | 3 → 8 → 16 → 32 → 64 → 128 |
| Spatial | 240 → 120 → 60 → 30 → 15 → 7 |
| Classifier | Flatten → FC(6272→64) → ReLU → Dropout(0.3) → FC(64→4) |
| Training Loss | `α · KL(softmax(Teacher/T) ‖ softmax(Student/T)) + (1−α) · CE` |

### Distillation Loss
```
L_KD = α · KL[ softmax(z^T / T) ‖ softmax(z^S / T) ] + (1 − α) · CE(y_true, z^S)
```
- `z^T`: Teacher logits
- `z^S`: Student logits
- `T`: Temperature (softens probability distributions)
- `α`: Balances soft distillation loss vs. hard label loss

### Quantization
- **Method:** Post-training dynamic quantization (`torch.quantization.quantize_dynamic`)
- **Target:** `nn.Linear` layers
- **Dtype:** `torch.qint8`
- **Result:** ~4× model size reduction with preserved accuracy

---

## 📂 Project Structure

```


Brain-Tumor-Classifier/
├── brain_tumor_kd_research_notebook.py   # Full research notebook (Colab)
├── Brain_Tumor_KD_Research.pdf           # Research paper (PDF)
├── main.py                     # FastAPI backend server & routing
├── inference.py                # Model architecture, preprocessing, Grad-CAM++ & LIME logic
├── lKD_T6.0_a0.8_best.pth     # Pretrained distilled weights (Student model)
├── static/
│   ├── index.html              # Single-page clinical UI
│   ├── style.css               # Clinical design system (Teal/White theme)
│   └── script.js               # Frontend upload, fetch, and rendering logic
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/imtiazdeepto/brain-tumor-kd-research.git
cd brain-tumor-kd-research
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `torch` + `torchvision`
- `fastapi` + `uvicorn` + `python-multipart`
- `opencv-python-headless`
- `pillow`, `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`
- `lime`, `scikit-image`
- `thop`, `psutil`

### 4. Download Model Weights
Place your trained `.pth` checkpoint (e.g., `KD_T6.0_a0.8_best.pth`) 

> **Note:** The model weights are generated by running the research notebook. If you do not have the weights, train the teacher and student using the provided notebook, or contact the authors.

---

## 🖥️ Usage

### Run the Web Application
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser:
- **Web UI:** [https://mri-brain-tumor-classifier.onrender.com](https://mri-brain-tumor-classifier.onrender.com)

### Using the Web Interface
1. Navigate to the homepage.
2. Drag and drop or click to upload a brain MRI image (JPG/PNG).
3. Click **Analyze Scan**.
4. View the predicted class, confidence score, probability distribution, and Grad-CAM++ overlay.

---

## 📡 API Documentation

### `GET /health`
Check server status and model load state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "CustomCNN5_Brain"
}
```

### `POST /predict`
Classify an uploaded MRI image.

**Request:** `multipart/form-data`
- `file`: MRI image (JPG/PNG, max 10MB)

**Response:**
```json
{
  "success": true,
  "predicted_class": "meningioma",
  "confidence": 0.9432,
  "all_probabilities": {
    "glioma": 0.0200,
    "meningioma": 0.9432,
    "notumor": 0.0010,
    "pituitary": 0.0358
  },
  "gradcam_image": "data:image/png;base64,iVBORw0KGgo...",
  "original_image": "data:image/png;base64,iVBORw0KGgo...",
  "processing_time_ms": 145.3,
  "model_info": {
    "name": "LiteBrainNet",
    "type": "KD Student (INT8 Quantized)",
    "input_size": "240x240"
  }
}
```

---

## 🔬 Research Background

This repository accompanies the research paper *"LiteBrainNet: A Lightweight Knowledge-Distilled Neural Network for Brain Tumor MRI Classification with Explainable AI and INT8 Quantization"*.

### Experimental Design

| Experiment | Description |
|------------|-------------|
| **Teacher Training** | EfficientNet-B1 trained with class-weighted CE for 40 epochs (early stopping patience = 10) |
| **KD Hyperparameter Ablation** | CustomCNN5 trained with `T = {3.0, 5.0, 6.0}` and `α = {0.5, 0.7, 0.8}` |
| **Architecture Ablation** | CNN3, CNN4, CNN5 trained with the best `(T, α)` from the hyperparameter ablation |
| **Baseline Models** | No-KD CNN5, MobileNetV2, ShuffleNetV2 trained on the identical train/val/test split |
| **Quantization** | Best KD student compressed with INT8 dynamic quantization on `nn.Linear` layers |
| **Benchmarking** | FLOPs, parameters, model size, RAM, CPU/GPU latency, and throughput |
| **XAI Evaluation** | Grad-CAM++ and LIME generated on both FP32 and INT8 models to verify explainability robustness under quantization |

### Preprocessing Pipeline
1. **Brain Region Crop:** OpenCV contour detection on grayscale → Gaussian blur → threshold(45) → erode(2) → dilate(2) → bounding box crop.
2. **Resize:** `cv2.resize(240, 240, INTER_CUBIC)`.
3. **Augmentation (train only):** Random affine (±10°), horizontal translation (20%), horizontal flip.
4. **Normalization:** ImageNet statistics — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

### Key Metrics Reported
- Classification: Accuracy, Weighted Precision/Recall/F1, Macro F1, Balanced Accuracy, MCC, Macro AUC (OvR)
- Efficiency: Model size (MB), FLOPs (G), Parameters (M), Inference latency (ms), Throughput (FPS), Peak RAM (MB)
- Per-class metrics for all four tumor categories

---

## ⚠️ Notes

- **Grad-CAM++** runs in milliseconds and is suitable for real-time deployment.
- **LIME** requires hundreds of forward passes per image. For high-concurrency production environments, consider disabling LIME to maintain sub-50ms latency.
- The INT8 quantized model is CPU-optimized and does not require a GPU for inference.
- All preprocessing steps in the web backend must exactly match the training pipeline to avoid distribution shift.

---

## 📜 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{Hasibur Rahman,Imtiaz Ibna kamal,
  title={LiteBrainNet: A Lightweight Knowledge-Distilled Neural Network for Brain Tumor MRI Classification with Explainable AI and INT8 Quantization},
  author={Your Name and Co-Authors},
  journal={Journal Name},
  year={2025},
  publisher={Publisher}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)
- Teacher backbone: [PyTorch EfficientNet](https://pytorch.org/vision/stable/models/efficientnet.html)
- Explainability: [Grad-CAM++](https://github.com/adityac94/Grad-CAM-plus-plus) and [LIME](https://github.com/marcotcr/lime)

---

> **Disclaimer:** This tool is intended for research and educational purposes. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or oncologist for clinical decisions.
