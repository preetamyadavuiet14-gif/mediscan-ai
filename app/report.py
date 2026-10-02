"""One-page PDF report for a prediction."""
import io
from datetime import datetime

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

DISCLAIMER = ("Research/screening aid only. Not a certified medical device and not a diagnosis. "
              "Always consult a qualified radiologist or doctor.")


def build_pdf(result: dict) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18); c.drawString(50, h - 60, "MediScan AI - Chest X-ray Report")
    c.setFont("Helvetica", 10); c.drawString(50, h - 78, datetime.now().strftime("%d %b %Y, %H:%M"))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, h - 110, f"Prediction: {result['label']}  ({result['confidence'] * 100:.1f}% confidence)")
    c.setFont("Helvetica", 11)
    y = h - 135
    for k, v in sorted(result["probs"].items(), key=lambda kv: -kv[1]):
        c.drawString(60, y, f"{k}: {v * 100:.1f}%"); y -= 16
    for i, (title, arr) in enumerate([("Input image", result["original"]), ("Grad-CAM heatmap", result["heatmap"])]):
        x = 50 + i * 260
        c.drawString(x, y - 20, title)
        c.drawImage(ImageReader(Image.fromarray(arr)), x, y - 260, width=240, height=240)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, DISCLAIMER[:95]); c.drawString(50, 38, DISCLAIMER[95:])
    c.save()
    return buf.getvalue()
