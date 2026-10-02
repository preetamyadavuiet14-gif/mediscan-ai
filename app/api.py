import base64
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.inference import Predictor, load_image, validate_xray

app = FastAPI(title="MediScan AI API")
predictor = Predictor(os.getenv("MODEL_PATH", "models/best.pt"))


def b64_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "classes": predictor.classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = load_image(await file.read(), file.filename or "")
    except Exception:
        raise HTTPException(400, "Could not read the image file.")
    ok, msg = validate_xray(img)
    if not ok:
        raise HTTPException(422, msg)
    r = predictor.predict(img)
    return {"label": r["label"], "confidence": r["confidence"], "probs": r["probs"],
            "low_confidence": r["confidence"] < 0.60, "heatmap_png_base64": b64_png(r["heatmap"]),
            "disclaimer": "Screening aid only, not a diagnosis."}
