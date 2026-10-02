"""Evaluate one checkpoint (or an ensemble of several) on a folder with class sub-folders.

  python -m src.evaluate --ckpt models/densenet121.pt --data-dir data/processed/test
  python -m src.evaluate --ckpt models/densenet121.pt models/resnet50.pt --data-dir data/processed/test --name ensemble
  python -m src.evaluate --ckpt models/densenet121.pt --data-dir data/external --name external
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.common import eval_tf, load_checkpoint


@torch.no_grad()
def get_probs(model, loader, device):
    out = []
    for x, _ in tqdm(loader, leave=False):
        out.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data/processed/test")
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ImageFolder(args.data_dir, eval_tf())
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)
    y = np.array(ds.targets)

    probs = 0
    for path in args.ckpt:
        model, ckpt = load_checkpoint(path, device)
        assert ds.classes == ckpt["classes"], f"class mismatch: {ds.classes} vs {ckpt['classes']}"
        probs = probs + get_probs(model, dl, device)
    probs /= len(args.ckpt)  # ensemble = mean of probabilities
    pred = probs.argmax(1)
    classes = ds.classes

    name = args.name or "_".join(Path(p).stem for p in args.ckpt)
    out = Path("reports") / name
    out.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y, pred)
    report = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        per_class[c] = {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1)}
    y_bin = label_binarize(y, classes=range(len(classes)))
    auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    summary = {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
               "macro_auc_ovr": auc, "per_class": per_class, "report": report}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    print(classification_report(y, pred, target_names=classes, digits=3, zero_division=0))
    print(f"Macro AUC (OvR): {auc:.4f}")
    for c, v in per_class.items():
        print(f"{c:14s} sensitivity {v['sensitivity']:.3f}  specificity {v['specificity']:.3f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(y_bin[:, i], probs[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(); ax.set_title("ROC (one-vs-rest)")
    fig.tight_layout(); fig.savefig(out / "roc_curves.png", dpi=150); plt.close(fig)
    print("saved to", out)


if __name__ == "__main__":
    main()
