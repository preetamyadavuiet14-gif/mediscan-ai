# MediScan AI - Complete Guide: build it in VS Code and publish a website anyone can use

Everything below is for a normal Windows PC (no GPU). Mac/Linux differences are noted.
Total time: about 1 day of work plus training time (1-3 hours on CPU).

---
## PART A - Install the tools (once)

1. **Python 3.11** - download from python.org/downloads. In the installer tick **"Add python.exe to PATH"**, then Install.
2. **VS Code** - download from code.visualstudio.com and install.
3. **Git** - download from git-scm.com and install (keep all defaults).
4. Make free accounts on **github.com** and **kaggle.com**, and later **share.streamlit.io** (sign in with GitHub).
5. Open VS Code -> Extensions icon (left bar) -> install **Python** (Microsoft) and **Pylance**.

## PART B - Open the project in VS Code

1. Unzip `mediscan-ai.zip` to a simple path such as `C:\Projects\mediscan-ai`.
2. VS Code -> **File -> Open Folder** -> choose `mediscan-ai`. Click "Yes, I trust the authors".
3. Open the terminal: **Terminal -> New Terminal** (it opens at the bottom, inside the project folder).
4. Create the environment and install libraries:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
   (Mac/Linux: `source .venv/bin/activate`.) The first install downloads ~500 MB, so wait for it to finish.
5. Press **Ctrl+Shift+P** -> type **Python: Select Interpreter** -> choose the one inside `.venv`.
   You should see `(.venv)` at the start of the terminal line from now on. If it is missing, run the activate command again.
   If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## PART C - Test everything with fake data (3 minutes)
```
python -m src.make_dummy_data
python -m src.prepare_data
python -m src.train --arch mobilenet_v3_small --workers 0 --batch-size 16 --warmup-epochs 1 --epochs 1
python -m src.evaluate --ckpt models/mobilenet_v3_small.pt --workers 0
```
If these run without red errors, your setup is fine. Now delete the test files: folders `data`, `runs`, `reports` and
the file `models\mobilenet_v3_small.pt` (right-click -> Delete in the VS Code file list).

## PART D - Get the real data

1. Go to https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database , log in, click **Download** (~800 MB).
2. Create a folder `downloads` inside the project and unzip the file into it.
3. In the VS Code file list, expand `downloads` and note the exact folder names (sometimes there is one extra nested folder).
4. Pick 800 random images per class (ONE line; fix the paths if yours differ):
```
python -m src.organize_raw --limit 800 --map covid19="downloads/COVID-19_Radiography_Dataset/COVID/images" --map normal="downloads/COVID-19_Radiography_Dataset/Normal/images" --map viral_pneumonia="downloads/COVID-19_Radiography_Dataset/Viral Pneumonia/images" --map lung_opacity="downloads/COVID-19_Radiography_Dataset/Lung_Opacity/images"
```
5. Remove duplicates and split into train / validation / test:
```
python -m src.prepare_data
```
   It prints a table of image counts. Every class should have images in train, val and test.

## PART E - Train and evaluate

1. Train (1-3 minutes per epoch on a modern CPU; keep the PC awake):
```
python -m src.train --arch mobilenet_v3_small --workers 0 --batch-size 16 --warmup-epochs 2 --epochs 8
```
   Slow PC? Use `--warmup-epochs 6 --epochs 0` (trains only the last layer; much faster, slightly less accurate).
   Optional second model to compare: `--arch resnet18`.
2. Evaluate on the test images:
```
python -m src.evaluate --ckpt models/mobilenet_v3_small.pt --workers 0
```
   Open `reports\mobilenet_v3_small\confusion_matrix.png` and `roc_curves.png` and copy the numbers (accuracy, macro-F1,
   sensitivity and specificity per class) into your final report.
3. Make it the model the website uses:
```
copy models\mobilenet_v3_small.pt models\best.pt
```

## PART F - Run the website on your own PC

```
streamlit run app/streamlit_app.py
```
(or press **F5** in VS Code and choose "Run MediScan website"). Your browser opens http://localhost:8501.
Upload a chest X-ray image (take one from `data\processed\test`) and check: prediction, confidence bars, Grad-CAM
heatmap, PDF report download. Try a normal colour photo too - the app should reject it. Stop the server with Ctrl+C.

## PART G - Put the website on the internet (free, public link)

### G1. Upload the project to GitHub
1. On github.com click **New repository**: name `mediscan-ai`, **Public**, do not add README/license -> Create.
2. In the VS Code terminal (first time only, use your own name and email):
```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
3. Then:
```
git init
git add .
git commit -m "MediScan AI first version"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/mediscan-ai.git
git push -u origin main
```
   A browser window asks you to log in to GitHub - approve it.
4. Refresh the GitHub page and check that `models/best.pt` is there (it is only a few MB). The `data` folder
   is intentionally not uploaded. Never upload real patient data.

### G2. Deploy with Streamlit Community Cloud (easiest)
1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **Create app** -> "Deploy a public app from GitHub".
3. Repository: `YOUR-USERNAME/mediscan-ai`, Branch: `main`, **Main file path: `app/streamlit_app.py`**.
4. Click **Deploy**. The first build takes 5-10 minutes (it installs PyTorch). When it finishes you get a public link like
   `https://your-name-mediscan-ai.streamlit.app` - **share this link with anyone**; they only need a browser.
5. Free apps go to sleep after some days without visitors. A visitor just clicks "Wake up" and it starts again.

### G2 (alternative). Hugging Face Spaces
1. huggingface.co -> New Space -> SDK **Streamlit**, hardware **CPU basic (free)**, visibility Public.
2. Upload all project files (`app`, `src`, `models`, `requirements.txt`, `README.md`).
3. Put these lines at the very top of the Space's `README.md`:
```
---
title: MediScan AI
sdk: streamlit
app_file: app/streamlit_app.py
---
```

### G3. Update the website later
Change code or the model, then run `git add .`, `git commit -m "update"`, `git push`. The site rebuilds by itself.

## PART H - Finish the project work

- Look at 10-20 Grad-CAM heatmaps. A good model highlights the **lungs**. If it highlights corners or text markers, it learned a shortcut.
- Try a larger model (`resnet18`, or `densenet121` if you ever get a GPU) and compare in a table.
- Optional extension: add Tuberculosis (see `RUN_ON_PC.md`).
- Put the live website link, screenshots of the app, the confusion matrix, ROC curves and the comparison table in your report.
- Write limitations: public datasets, label noise, few hospitals, not clinically validated, **not a medical device**.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised | Reinstall Python and tick "Add to PATH", then reopen VS Code |
| `(.venv)` not shown | Run `.venv\Scripts\activate` again, or re-select the interpreter |
| DataLoader / multiprocessing error | Always add `--workers 0` on Windows |
| Out of memory | Use `--batch-size 8` |
| `FileNotFoundError` in organize_raw | A path in `--map` is wrong - check the folder names in `downloads` |
| Class has 0 images in train | Check the counts printed by `prepare_data`, then redo Part D |
| Training too slow | Use `--warmup-epochs 6 --epochs 0` |
| Streamlit says "Model file not found" | Copy your trained model to `models\best.pt` |
| Deploy fails on Streamlit Cloud | Open "Manage app" -> logs; usually `models/best.pt` was not pushed (check `.gitignore`) |
| `git push` rejected | Run `git pull origin main --rebase`, then push again |

**Important:** MediScan AI is a research and screening aid, not a medical device. Keep the disclaimer visible on the website,
and never present its output as a diagnosis.
