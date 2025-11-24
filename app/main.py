# File: app/main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import numpy as np
import traceback

# Import the model utilities
from app.model import predict, gradcam, CLASS_NAMES

app = FastAPI(title="Brain Tumor MRI Classifier (InceptionV3 + Grad-CAM)")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        label, confidence, probs = predict(pil_img)
        return JSONResponse({
            "predicted_label": label,
            "confidence": round(confidence, 3),
            "probabilities": {k: round(v, 6) for k, v in probs.items()}
        })
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

@app.post("/gradcam")
async def gradcam_image(file: UploadFile = File(...), interpolant: float = Query(0.5, ge=0.0, le=1.0)):
    """
    Returns a PNG image (overlay) produced by gradcam().
    `interpolant` controls mixing (0..1).
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Compute overlay (this calls the optimized gradcam in model.py)
        overlay = gradcam(pil_img, interpolant=float(interpolant))

        # Ensure correct dtype and shape
        overlay = np.asarray(overlay).astype("uint8")
        if overlay.ndim == 2:
            overlay = np.stack([overlay] * 3, axis=-1)

        # Convert to PNG bytes
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

# Optional health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "classes": CLASS_NAMES}
