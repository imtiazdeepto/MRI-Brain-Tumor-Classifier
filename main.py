"""
main.py — FastAPI application for the Brain Tumour MRI Classifier.

Start the server:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Routes:
    GET  /         → serves index.html
    GET  /health   → model health check
    POST /predict  → multipart image upload, returns JSON classification
    GET  /static/* → CSS / JS / any other static assets
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import load_model, run_inference

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
WEIGHTS    = BASE_DIR / "KD_T6.0_a0.8_latest.pth"
STATIC_DIR = BASE_DIR / "static"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release on shutdown."""
    if not WEIGHTS.exists():
        logger.error(
            "Weights file not found: %s  —  Place KD_T6.0_a0.8_latest.pth "
            "in the same directory as main.py.",
            WEIGHTS,
        )
        _state["model_loaded"] = False
    else:
        logger.info("Loading model weights from %s …", WEIGHTS)
        try:
            _state["model"]        = load_model(WEIGHTS)
            _state["model_loaded"] = True
            logger.info("Model loaded and ready.")
        except Exception as exc:
            logger.error("Model load failed: %s", exc, exc_info=True)
            _state["model_loaded"] = False

    yield

    _state.clear()
    logger.info("Application shutdown — state cleared.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Brain Tumour MRI Classifier",
    description=(
        "Upload a brain MRI image to classify it as glioma, meningioma, "
        "pituitary tumour, or healthy tissue, with a Grad-CAM++ visual explanation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health / readiness check", tags=["meta"])
async def health() -> dict:
    """Returns server status and whether the model is loaded."""
    return {
        "status":       "ok",
        "model_loaded": _state.get("model_loaded", False),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.post("/predict", summary="Classify a brain MRI scan", tags=["inference"])
async def predict(file: UploadFile = File(..., description="Brain MRI image file")) -> JSONResponse:
    """
    Accept a single image file (multipart/form-data), run the
    CustomCNN5_Brain model, and return:

    - **predicted_class**: one of `glioma`, `meningioma`, `notumor`, `pituitary`
    - **confidence**: softmax probability for the predicted class (0 – 1)
    - **heatmap_image**: base64-encoded PNG of the Grad-CAM++ overlay
    """
    # --- Validate MIME type --------------------------------------------------
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Please upload an image file (JPEG, PNG, BMP, TIFF, or WebP)."
            ),
        )

    # --- Read body -----------------------------------------------------------
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # --- Model guard ---------------------------------------------------------
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Please try again in a moment.",
        )

    # --- Inference -----------------------------------------------------------
    try:
        result = run_inference(model, contents)
    except ValueError as exc:
        # Bad image data (e.g., corrupt file, unreadable format)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed for file '%s'", file.filename)
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Please try a different image or contact support.",
        )

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html")


# Mount static assets at /static (CSS, JS, images)
# NOTE: This must come after route definitions so that /health and /predict
#       are matched first by FastAPI's router.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
