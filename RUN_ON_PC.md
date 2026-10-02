# Run MediScan AI on a normal PC (no GPU needed)

"Lite" setup: **MobileNetV3-Small** (tiny CNN, ~2.5M parameters) trained on about 3,200 images from one dataset.
Needs: Windows/Mac/Linux, 8 GB RAM, ~5 GB free disk, internet for the first run (downloads small pretrained weights).

## 1. Install Python
Install Python 3.11 from python.org (Windows: tick **"Add Python to PATH"**). Check: `python --version`

## 2. Open the project
Unzip `mediscan-ai.zip`, open a terminal (PowerShell) inside the `mediscan-ai` folder.

## 3. Create environment and install (CPU version of PyTorch = smaller download)
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
(Mac/Linux: activate with `source .venv/bin/activate`.)

## 4. Quick test with fake data (2-3 minutes)
```
python -m src.make_dummy_data
python -m src.prepare_data
python -m src.train --arch mobilenet_v3_small --workers 0 --batch-size 16 --warmup-epochs 1 --epochs 1
python -m src.evaluate --ckpt models/mobilenet_v3_small.pt --workers 0
```
If this works, delete the test data: remove the folders `data\raw`, `data\processed`, `runs`, `reports` and the file `models\mobilenet_v3_small.pt`.

## 5. Download ONE dataset
Open https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database , log in, click **Download** (~800 MB).
Unzip it into a folder `downloads` inside the project. Check the folder names with `dir downloads` (they may be nested one level deeper).
Using a single dataset for all classes avoids the model learning "which dataset" instead of "which disease".

## 6. Pick 800 random images per class
Run as ONE line (adjust paths if your folders differ):
```
python -m src.organize_raw --limit 800 --map covid19="downloads/COVID-19_Radiography_Dataset/COVID/images" --map normal="downloads/COVID-19_Radiography_Dataset/Normal/images" --map viral_pneumonia="downloads/COVID-19_Radiography_Dataset/Viral Pneumonia/images" --map lung_opacity="downloads/COVID-19_Radiography_Dataset/Lung_Opacity/images"
python -m src.prepare_data
```
(Viral Pneumonia has only ~1,345 images, so it contributes fewer than 800 - that is fine, class weights handle it.)

## 7. Train (about 1-3 minutes per epoch on a modern CPU; leave it running)
```
python -m src.train --arch mobilenet_v3_small --workers 0 --batch-size 16 --warmup-epochs 2 --epochs 8
```
Too slow? Faster mode (trains only the final layer): `--warmup-epochs 6 --epochs 0`.
Want a second model to compare? `--arch resnet18` (slower but still OK on CPU).

## 8. Evaluate
```
python -m src.evaluate --ckpt models/mobilenet_v3_small.pt --workers 0
```
Results (accuracy, sensitivity, specificity, confusion matrix, ROC) are saved in `reports\mobilenet_v3_small\`.

## 9. Run the app
```
copy models\mobilenet_v3_small.pt models\best.pt
streamlit run app/streamlit_app.py
```
Your browser opens at http://localhost:8501 - upload an X-ray from the dataset and you get the prediction,
confidence, Grad-CAM heatmap and a PDF report.

## Optional: add Tuberculosis as a 5th class
Download "Tuberculosis (TB) Chest X-ray Database" from Kaggle (`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`),
add `--map tuberculosis="downloads/TB_Chest_Radiography_Database/Tuberculosis"` to step 6 and train again.
Note: TB images come from a different source than the other classes, so the model may partly learn the source.
Mention this limitation in your report.
