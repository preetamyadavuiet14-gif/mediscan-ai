"""Loading images, validating them, and predicting with Grad-CAM explanation."""
import io

import numpy as np
import torch
from PIL import Image

from src.common import display_tf, eval_tf, load_checkpoint, target_layer
from src.gradcam import GradCAM, overlay


def load_image(data: bytes, filename: str = "") -> Image.Image:
    """Open JPEG/PNG or DICOM bytes as a PIL image."""
    if filename.lower().endswith((".dcm", ".dicom")):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 255 - arr
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return Image.open(io.BytesIO(data)).convert("RGB")


def validate_xray(img: Image.Image):
    """Cheap sanity check: reject tiny or clearly colourful pictures. Returns (ok, message)."""
    if min(img.size) < 128:
        return False, "Image is too small (minimum 128 px on the short side)."
    a = np.asarray(img.resize((128, 128))).astype(int)
    colour = np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()
    if colour > 12:
        return False, "This looks like a colour photo, not a chest X-ray."
    return True, "ok"


class Predictor:
    def __init__(self, ckpt_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.classes = ckpt["classes"]
        self.cam = GradCAM(self.model, target_layer(self.model, ckpt["arch"]))
        self.tf, self.show = eval_tf(), display_tf()

    def predict(self, img: Image.Image) -> dict:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        cam, probs, idx = self.cam(x)
        base = np.asarray(self.show(img.convert("RGB")))
        return {
            "label": self.classes[idx],
            "confidence": float(probs[idx]),
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "original": base,
            "heatmap": overlay(base, cam),
        }
