import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.report import DISCLAIMER, build_pdf
from src.inference import Predictor, load_image, validate_xray

MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "best.pt"))
ASSETS = ROOT / "assets"
LOW_CONF = 0.60
NAMES = {"covid19": "COVID-19", "normal": "Normal", "viral_pneumonia": "Viral Pneumonia",
         "lung_opacity": "Lung Opacity", "tuberculosis": "Tuberculosis"}
pretty = lambda c: NAMES.get(c, c.replace("_", " ").title())

st.set_page_config(page_title="MediScan AI", page_icon="🫁", layout="wide")
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg, #0f4c81, #1aa6b7); color: #fff; padding: 2.6rem 2rem;
       border-radius: 18px; text-align: center; margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.6rem; margin: 0 0 .4rem 0; color: #fff;}
.hero p {font-size: 1.15rem; opacity: .95; margin: 0;}
.card {border: 1px solid rgba(128,128,128,.35); border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%;}
.card h3 {margin: 0 0 .4rem 0;}
.step {border-left: 4px solid #1aa6b7; padding: .2rem 0 .2rem .9rem; margin-bottom: .9rem;}
.foot {text-align: center; opacity: .7; font-size: .85rem; margin-top: 2rem;}
</style>
<div class="hero"><h1>🫁 MediScan AI</h1>
<p>Explainable deep learning for chest X-ray screening - upload an image, get a prediction and a heatmap showing why.</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return Predictor(MODEL_PATH)


home, try_tab, how, results = st.tabs(["🏠 Home", "🔬 Try it", "⚙️ How it works", "📊 Results"])

with home:
    st.subheader("What is MediScan AI?")
    st.write("Chest X-rays are the most common imaging test in the world, but there are too few radiologists to read "
             "them quickly everywhere. MediScan AI is a deep learning model that screens a chest X-ray and "
             "classifies it as Normal, COVID-19, Viral Pneumonia or Lung Opacity - and shows *where* it looked.")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h3>⬆️ Upload</h3>Drop in a JPG, PNG or DICOM chest X-ray. '
                'Nothing is stored - images are processed in memory.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h3>🔥 Explainable</h3>A Grad-CAM heatmap highlights the lung regions '
                'that influenced the decision, so you can judge if it makes sense.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h3>📄 Report</h3>Get class probabilities, a low-confidence warning '
                'and a downloadable PDF report.</div>', unsafe_allow_html=True)
    st.warning(DISCLAIMER)

with try_tab:
    if not Path(MODEL_PATH).exists():
        st.error("Model file not found. Train a model and copy it to models/best.pt first.")
    else:
        file = st.file_uploader("Upload a chest X-ray (JPG, PNG or DICOM)", type=["jpg", "jpeg", "png", "dcm"])
        if file:
            img = load_image(file.getvalue(), file.name)
            ok, msg = validate_xray(img)
            if not ok:
                st.error(msg)
            else:
                with st.spinner("Analysing..."):
                    res = get_predictor().predict(img)
                res["label"] = pretty(res["label"])
                res["probs"] = {pretty(k): v for k, v in res["probs"].items()}
                a, b = st.columns(2)
                a.image(res["original"], caption="Input image", width="stretch")
                b.image(res["heatmap"], caption="Grad-CAM: regions that influenced the decision",
                        width="stretch")
                st.subheader(f"Prediction: {res['label']} ({res['confidence'] * 100:.1f}%)")
                if res["confidence"] < LOW_CONF:
                    st.warning("Low confidence - the result is uncertain. Please consult a qualified doctor.")
                st.bar_chart(res["probs"])
                st.download_button("⬇️ Download PDF report", build_pdf(res), "mediscan_report.pdf", "application/pdf")
        st.caption(DISCLAIMER)

with how:
    steps = [("1. Pre-processing", "The X-ray is contrast-enhanced (CLAHE), resized to 224×224 and normalised."),
             ("2. Deep learning model", "A MobileNetV3 convolutional neural network, pre-trained on ImageNet and "
                                        "fine-tuned on thousands of chest X-rays (transfer learning)."),
             ("3. Prediction", "The model outputs a probability for each class; the highest is the prediction."),
             ("4. Explanation", "Grad-CAM uses the gradients of the last convolutional layer to draw a heatmap "
                                "of the image regions that mattered most."),
             ("5. Safety checks", "Colour photos and tiny images are rejected, and low-confidence results are flagged.")]
    for t, d in steps:
        st.markdown(f'<div class="step"><b>{t}</b><br>{d}</div>', unsafe_allow_html=True)

with results:
    mfile = ASSETS / "metrics.json"
    if mfile.exists():
        m = json.loads(mfile.read_text())
        x, y, z = st.columns(3)
        x.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
        y.metric("Macro F1-score", f"{m['macro_f1']:.3f}")
        z.metric("Macro AUC", f"{m['macro_auc_ovr']:.3f}")
        df = pd.DataFrame(m["per_class"]).T.rename(columns={"sensitivity": "Sensitivity", "specificity": "Specificity"})
        df.index = [pretty(i) for i in df.index]
        st.dataframe(df.style.format("{:.3f}"), width="stretch")
        for f, cap in [("confusion_matrix.png", "Confusion matrix (test set)"), ("roc_curves.png", "ROC curves")]:
            if (ASSETS / f).exists():
                st.image(str(ASSETS / f), caption=cap, width=520)
    else:
        st.info("Run the evaluation and copy the files from reports/<model>/ into the assets/ folder to show results here.")

st.markdown('<div class="foot">Built by Preetam Yadav · Deep Learning Project 2026 · '
            'Research and screening aid only, not a medical device.</div>', unsafe_allow_html=True)
