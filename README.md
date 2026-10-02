# MediScan AI - Explainable Chest X-ray Classifier

Classifies chest X-rays into **covid19 / normal / pneumonia / tuberculosis** using transfer learning
(ResNet50, DenseNet121, EfficientNet-B0), explains each prediction with **Grad-CAM**, and serves it via a
**Streamlit web app** and a **FastAPI** endpoint.

> Research / screening aid only. Not a certified medical device and not a diagnosis.
>
> **Normal PC without GPU? Follow `GUIDE_VSCODE_AND_WEBSITE.md` (VS Code + public website) or `RUN_ON_PC.md`.**

```
mediscan-ai/
├── src/
│   ├── common.py         transforms (CLAHE), model builder, checkpoint loader
│   ├── organize_raw.py   gather downloaded datasets into data/raw/<class>/
│   ├── prepare_data.py   de-duplicate + grouped, stratified 80/10/10 split
│   ├── train.py          warm-up + fine-tune, class weights, early stopping
│   ├── evaluate.py       metrics, confusion matrix, ROC, ensemble, external test
│   ├── gradcam.py        Grad-CAM implementation
│   ├── inference.py      image loading (incl. DICOM), validation, Predictor
│   └── make_dummy_data.py  fake data for a quick pipeline test
├── app/
│   ├── streamlit_app.py  web UI
│   ├── api.py            FastAPI REST API
│   └── report.py         PDF report generator
├── models/               trained checkpoints (.pt) go here
├── Dockerfile
└── requirements.txt
```

## Step 1 - Set up the environment (Month 1)
Python 3.10-3.12. A GPU is strongly recommended: use **Google Colab / Kaggle notebooks** (free GPU) for training.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
(Colab: upload the folder, then `!pip install -r requirements.txt` and prefix commands below with `!`.)

## Step 2 - Test the whole pipeline with dummy data (5 minutes)
This proves your setup works before you spend hours on real data.
```bash
python -m src.make_dummy_data
python -m src.prepare_data
python -m src.train --arch resnet50 --warmup-epochs 1 --epochs 1 --workers 0
python -m src.evaluate --ckpt models/resnet50.pt --workers 0
```
Delete `data/raw`, `data/processed`, `models/*.pt`, `reports`, `runs` afterwards.

## Step 3 - Download the datasets (Month 1)
Public, de-identified datasets (read each licence / terms first):

| Dataset | Source |
|---|---|
| COVID-19 Radiography Database (COVID, Normal, Viral Pneumonia) | Kaggle: `tawsifurrahman/covid19-radiography-database` |
| Chest X-Ray Images - Pneumonia (Kermany) | Kaggle: `paultimothymooney/chest-xray-pneumonia` |
| Tuberculosis (TB) Chest X-ray Database | Kaggle: `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` |
| Montgomery / Shenzhen TB sets, NIH ChestX-ray14, CheXpert | official NIH / Stanford / NLM pages |

With the Kaggle CLI (`pip install kaggle`, put your `kaggle.json` in `~/.kaggle`):
```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p downloads --unzip
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p downloads --unzip
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset -p downloads --unzip
```
Folder names differ between versions, so run `ls downloads` (or look in File Explorer) to find the exact paths.

## Step 4 - Organise into 4 classes
Map source folders to class names. Repeat `--map normal=...` for every source of that class:
```bash
python -m src.organize_raw \
  --map covid19="downloads/COVID-19_Radiography_Dataset/COVID/images" \
  --map normal="downloads/COVID-19_Radiography_Dataset/Normal/images" \
  --map normal="downloads/chest_xray/train/NORMAL" \
  --map pneumonia="downloads/COVID-19_Radiography_Dataset/Viral Pneumonia/images" \
  --map pneumonia="downloads/chest_xray/train/PNEUMONIA" \
  --map tuberculosis="downloads/TB_Chest_Radiography_Database/Tuberculosis"
```
**Important - avoid shortcut learning.** If a class comes from only one source, the model can learn the
*source* (scanner, image size, borders) instead of the disease. Take each class from as many sources as
possible, and hold one whole dataset out for external validation (Step 7). Report this honestly in your thesis.

## Step 5 - Clean and split
```bash
python -m src.prepare_data
```
Removes exact duplicates and creates `data/processed/{train,val,test}/<class>/`. If your files carry a patient
id (NIH ChestX-ray14: `00000013_005.png`), add `--group-regex "^(\d+)_"` so one patient never appears in two
splits.

## Step 6 - Train (Months 2-4)
Train each architecture and compare:
```bash
python -m src.train --arch densenet121
python -m src.train --arch resnet50
python -m src.train --arch efficientnet_b0
tensorboard --logdir runs          # watch curves at http://localhost:6006
```
The trainer warms up the new head for 3 epochs, then fine-tunes all layers with cosine LR, class-weighted loss
and early stopping on validation macro-F1. Useful flags: `--batch-size 16` (out of memory), `--workers 0`
(Windows errors), `--epochs 30`. Tuning ideas: learning rate (1e-4 / 3e-4 / 1e-5), image size, dropout.

## Step 7 - Evaluate (Month 5)
```bash
python -m src.evaluate --ckpt models/densenet121.pt                      # test split
python -m src.evaluate --ckpt models/densenet121.pt models/resnet50.pt --name ensemble
python -m src.evaluate --ckpt models/densenet121.pt --data-dir data/external --name external
```
Outputs in `reports/<name>/`: `metrics.json`, `confusion_matrix.png`, `roc_curves.png`, plus per-class
sensitivity/specificity printed in the terminal. Put these into your report. Prioritise **sensitivity
(recall)** for disease classes: a missed case costs more than a false alarm. For `data/external`, use the same
four class folder names. Compare test vs external results to show how well the model generalises.
Choose your best model and copy it:
```bash
cp models/densenet121.pt models/best.pt        # Windows: copy models\densenet121.pt models\best.pt
```

## Step 8 - Run the app (Month 6)
```bash
streamlit run app/streamlit_app.py                       # web UI at http://localhost:8501
uvicorn app.api:app --host 0.0.0.0 --port 8000           # API docs at http://localhost:8000/docs
```
API example: `curl -F "file=@xray.png" http://localhost:8000/predict`.
Set another model with `MODEL_PATH=models/resnet50.pt` (env variable).
Features: upload (JPG/PNG/DICOM) and validation, class probabilities, Grad-CAM heatmap, low-confidence
warning, downloadable PDF report.

## Step 9 - Docker and deployment
```bash
docker build -t mediscan-ai .
docker run -p 8501:8501 mediscan-ai
```
Free hosting: **Hugging Face Spaces** (Streamlit or Docker Space; upload the folder incl. `models/best.pt`, use
port 7860) or **Render**. Keep a copy of the disclaimer visible in the app.

## Step 10 - Sanity checks before you present
- Open a few Grad-CAM heatmaps per class. Does the model look at the **lungs**, not text markers or borders?
- Check the confusion matrix for which classes get confused (COVID vs viral pneumonia is typically hardest).
- Try obviously wrong inputs (a selfie, a blank image) - the validator and low-confidence warning should react.
- Write down limitations: dataset bias, label noise, few hospitals, no clinical validation.

## Suggested 6-month plan
| Month | Do |
|---|---|
| 1 | Steps 1-5: setup, datasets, literature review |
| 2 | Baseline model, preprocessing/augmentation experiments |
| 3-4 | Step 6: train and tune all three architectures, try ensembles |
| 5 | Step 7 + Grad-CAM analysis, external validation, error analysis |
| 6 | Steps 8-9: app, API, Docker, deploy; final report and presentation |

## Optional extensions
Vision Transformer (`timm`), focal loss, test-time augmentation, temperature scaling for calibrated
confidence, more diseases (NIH ChestX-ray14 is multi-label), CT/MRI.
