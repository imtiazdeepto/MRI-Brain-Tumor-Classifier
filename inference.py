"""
inference.py — Model definition, preprocessing pipeline, and Grad-CAM++
for the MRI Brain Tumour Classifier.

Architecture: CustomCNN5_Brain
  5 × (Conv2d 3×3 pad=1 → ReLU → MaxPool2d 2×2)
  channels: 3 → 8 → 16 → 32 → 64 → 128
  Classifier: Flatten → Linear(6272,64) → ReLU → Dropout(0.3) → Linear(64,4)

Preprocessing (matches training):
  BGR→RGB → contour-crop → resize 240×240 cubic → ToTensor → ImageNet normalise

Grad-CAM++ target layer: model.features[12]  (Conv2d 64→128)
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
class CustomCNN5_Brain(nn.Module):
    """
    Five-block CNN for 4-class brain tumour classification.

    Input : [B, 3, 240, 240]
    Output: [B, 4]  (raw logits — glioma, meningioma, notumor, pituitary)

    Spatial progression (each MaxPool2d halves each dim):
        240 → 120 → 60 → 30 → 15 → 7
    Flattened: 128 × 7 × 7 = 6 272 features.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 8   output: 120×120
            nn.Conv2d(3,   8,  kernel_size=3, padding=1),  # [0]
            nn.ReLU(inplace=True),                          # [1]
            nn.MaxPool2d(2, 2),                             # [2]
            # Block 2: 8 → 16  output: 60×60
            nn.Conv2d(8,  16,  kernel_size=3, padding=1),  # [3]
            nn.ReLU(inplace=True),                          # [4]
            nn.MaxPool2d(2, 2),                             # [5]
            # Block 3: 16 → 32 output: 30×30
            nn.Conv2d(16, 32,  kernel_size=3, padding=1),  # [6]
            nn.ReLU(inplace=True),                          # [7]
            nn.MaxPool2d(2, 2),                             # [8]
            # Block 4: 32 → 64 output: 15×15
            nn.Conv2d(32, 64,  kernel_size=3, padding=1),  # [9]
            nn.ReLU(inplace=True),                          # [10]
            nn.MaxPool2d(2, 2),                             # [11]
            # Block 5: 64 → 128  ← Grad-CAM++ target layer (features[12])
            # output: 15×15 (before ReLU/pool), 7×7 (after pool)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [12]  ← TARGET
            nn.ReLU(inplace=True),                          # [13]
            nn.MaxPool2d(2, 2),                             # [14]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # 6272
            nn.Linear(6272, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model(weights_path: Path) -> CustomCNN5_Brain:
    """Instantiate the model, load state dict, set to eval mode on CPU."""
    model = CustomCNN5_Brain()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded from %s", weights_path)
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _crop_brain_region(img_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Contour-based brain-region crop.

    Returns
    -------
    cropped : ndarray  —  cropped to largest external contour's bounding box,
                          or the original image if no valid contour is found.
    used_fallback : bool
    """
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No brain contour detected; falling back to original image.")
        return img_rgb, True

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if w <= 0 or h <= 0:
        logger.warning("Degenerate contour bounding box; falling back to original image.")
        return img_rgb, True

    return img_rgb[y : y + h, x : x + w], False


def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Full preprocessing pipeline matching training.

    Parameters
    ----------
    image_bytes : raw bytes of the uploaded image file

    Returns
    -------
    input_tensor : shape [1, 3, 240, 240], float32, ImageNet-normalised
    original_rgb_240 : shape [240, 240, 3], uint8 RGB — for Grad-CAM++ overlay
    """
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(
            "Could not decode the uploaded file. "
            "Please upload a valid image (JPEG, PNG, BMP, TIFF, or WebP)."
        )

    # 1. BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Contour-based brain crop (with fallback)
    cropped, _ = _crop_brain_region(img_rgb)

    # 3. Resize to 240×240 with cubic interpolation
    resized = cv2.resize(cropped, (240, 240), interpolation=cv2.INTER_CUBIC)

    # 4. ToTensor + ImageNet normalise
    pil_img      = Image.fromarray(resized)
    input_tensor = _PREPROCESS(pil_img).unsqueeze(0)  # [1, 3, 240, 240]

    return input_tensor, resized  # resized is kept as the unnormalised overlay base


# ---------------------------------------------------------------------------
# Grad-CAM++
# ---------------------------------------------------------------------------
def _compute_gradcam_pp(
    model: CustomCNN5_Brain,
    input_tensor: torch.Tensor,
    class_idx: int,
    target_layer: nn.Module,
) -> np.ndarray:
    """
    Compute a Grad-CAM++ saliency map for the given class index.

    Uses the practical approximation:
        alpha_k = relu(grad)² / (2·relu(grad)² + Σ_ab A_k^ab·relu(grad)³ + ε)
        w_k     = Σ_ij alpha_k^ij · relu(grad_k^ij)
        CAM     = ReLU( Σ_k w_k · A_k )  then normalised to [0,1]

    Parameters
    ----------
    model        : CustomCNN5_Brain in eval mode
    input_tensor : [1, 3, 240, 240] float tensor
    class_idx    : integer class index for which to generate the CAM
    target_layer : the nn.Module to hook (model.features[12])

    Returns
    -------
    cam : ndarray, shape [h_feat, w_feat], values in [0, 1]
    """
    activation_store: dict[str, torch.Tensor] = {}
    gradient_store:   dict[str, torch.Tensor] = {}

    def _fwd_hook(_mod: nn.Module, _inp, output: torch.Tensor) -> None:
        # Do NOT detach — keep in computation graph for backward
        activation_store["A"] = output

    def _bwd_hook(_mod: nn.Module, _grad_in, grad_out: tuple) -> None:
        gradient_store["dA"] = grad_out[0]

    fh = target_layer.register_forward_hook(_fwd_hook)
    bh = target_layer.register_full_backward_hook(_bwd_hook)

    model.zero_grad()
    output = model(input_tensor)            # full forward; graph is live
    score  = output[0, class_idx]
    score.backward()                        # propagate gradients

    fh.remove()
    bh.remove()

    A     = activation_store["A"]           # [1, C, h, w]
    grads = gradient_store["dA"]            # [1, C, h, w]

    # Grad-CAM++ weight computation
    rg    = F.relu(grads)                   # positive gradients only
    rg2   = rg.pow(2)                       # [1, C, h, w]
    rg3   = rg.pow(3)                       # [1, C, h, w]

    # sum of A weighted by cube of positive gradients (per channel)
    sum_A_rg3 = (F.relu(A) * rg3).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

    alpha   = rg2 / (2.0 * rg2 + sum_A_rg3 + 1e-7)               # [1, C, h, w]
    weights = (alpha * rg).sum(dim=(2, 3))                         # [1, C]

    # Weighted sum of activation maps, then ReLU
    cam = (weights[:, :, None, None] * A).sum(dim=1)               # [1, h, w]
    cam = F.relu(cam).squeeze().detach().cpu().numpy()              # [h, w]

    # Normalise to [0, 1]
    lo, hi = float(cam.min()), float(cam.max())
    if hi > lo:
        cam = (cam - lo) / (hi - lo)
    else:
        cam = np.zeros_like(cam)

    return cam


def _create_overlay(original_rgb: np.ndarray, cam: np.ndarray) -> str:
    """
    Resize CAM to the input image dimensions, apply COLORMAP_JET, blend at
    50 % opacity, and return a base64-encoded PNG string.

    Parameters
    ----------
    original_rgb : [H, W, 3] uint8 array (unnormalised RGB)
    cam          : [h, w]   float array in [0, 1]

    Returns
    -------
    base64 PNG string
    """
    h, w = original_rgb.shape[:2]

    cam_resized  = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 50% blend
    overlay = np.clip(
        0.5 * original_rgb.astype(np.float32) + 0.5 * heatmap_rgb.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_inference(model: CustomCNN5_Brain, image_bytes: bytes) -> dict:
    """
    Complete pipeline: preprocess → predict → Grad-CAM++ → overlay.

    Parameters
    ----------
    model       : loaded CustomCNN5_Brain in eval mode
    image_bytes : raw bytes of the uploaded image file

    Returns
    -------
    dict with keys:
        predicted_class : str  (one of CLASSES)
        confidence      : float  in [0, 1]
        heatmap_image   : str  base64-encoded PNG of the Grad-CAM++ overlay
    """
    # 1. Preprocess
    input_tensor, original_rgb = preprocess_image(image_bytes)

    # 2. Class prediction (no-grad for efficiency)
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    class_idx       = int(idx.item())
    confidence      = float(conf.item())
    predicted_class = CLASSES[class_idx]

    logger.info(
        "Predicted class: %s  (confidence %.2f%%)",
        predicted_class, confidence * 100
    )

    # 3. Grad-CAM++ (separate forward pass; gradients required)
    cam     = _compute_gradcam_pp(model, input_tensor, class_idx, model.features[12])
    heatmap = _create_overlay(original_rgb, cam)

    return {
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 6),
        "heatmap_image":   heatmap,
    }
