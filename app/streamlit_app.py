import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.report import DISCLAIMER, build_pdf
from src.inference import Predictor, load_image, validate_xray

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
LOW_CONF = 0.60

st.set_page_config(page_title="MediScan AI", page_icon="🫁", layout="wide")


@st.cache_resource
def get_predictor():
    return Predictor(MODEL_PATH)


st.title("🫁 MediScan AI - Chest X-ray Classifier")
st.warning(DISCLAIMER)

if not Path(MODEL_PATH).exists():
    st.error(f"Model file not found: {MODEL_PATH}. Train a model and copy it to models/best.pt first.")
    st.stop()

file = st.file_uploader("Upload a chest X-ray (JPG, PNG or DICOM)", type=["jpg", "jpeg", "png", "dcm"])
if file:
    img = load_image(file.getvalue(), file.name)
    ok, msg = validate_xray(img)
    if not ok:
        st.error(msg)
        st.stop()
    with st.spinner("Analysing..."):
        res = get_predictor().predict(img)

    c1, c2 = st.columns(2)
    c1.image(res["original"], caption="Input (preprocessed view)", use_container_width=True)
    c2.image(res["heatmap"], caption="Grad-CAM: regions that influenced the decision", use_container_width=True)

    st.subheader(f"Prediction: {res['label']}  ({res['confidence'] * 100:.1f}%)")
    if res["confidence"] < LOW_CONF:
        st.warning("Low confidence - the result is uncertain. Please consult a qualified doctor.")
    st.bar_chart(res["probs"])
    st.download_button("Download PDF report", build_pdf(res), "mediscan_report.pdf", "application/pdf")
