"""Transfer-learning trainer: warm-up (head only) -> fine-tune (all layers), class-weighted loss,
cosine LR, early stopping on validation macro-F1, TensorBoard logging."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.common import ARCHS, IMG_SIZE, build_model, eval_tf, get_head, train_tf


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, n, preds, labels = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * len(y)
            n += len(y)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    return (loss_sum / n, f1_score(labels, preds, average="macro"),
            recall_score(labels, preds, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--arch", default="densenet121", choices=ARCHS)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2, help="use 0 on Windows if you get errors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="for offline smoke tests only")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    root = Path(args.data_dir)
    train_ds = ImageFolder(root / "train", train_tf())
    val_ds = ImageFolder(root / "val", eval_tf())
    assert train_ds.classes == val_ds.classes, "train/val class folders differ"
    classes = train_ds.classes
    counts = np.bincount(train_ds.targets, minlength=len(classes))
    assert counts.min() > 0, f"a class has no training images: {dict(zip(classes, counts))}"
    print("classes:", dict(zip(classes, counts.tolist())))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, len(classes), not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)  # handles class imbalance
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(f"runs/{args.arch}")
    Path("models").mkdir(exist_ok=True)
    ckpt_path = Path("models") / f"{args.arch}.pt"

    total = args.warmup_epochs + args.epochs
    best, bad, scheduler = -1.0, 0, None
    for epoch in range(total):
        if epoch == 0 and args.warmup_epochs > 0:  # phase 1: train only the new head
            for p in model.parameters():
                p.requires_grad = False
            for p in get_head(model).parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(get_head(model).parameters(), lr=args.lr_head, weight_decay=1e-4)
        if epoch == args.warmup_epochs:  # phase 2: fine-tune everything
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        tr_loss, tr_f1, _ = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_f1, va_rec = run_epoch(model, val_dl, criterion, device)
        if scheduler:
            scheduler.step()
        print(f"epoch {epoch + 1}/{total} | train loss {tr_loss:.4f} f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} f1 {va_f1:.3f} recall {va_rec:.3f}")
        for k, v in {"loss/train": tr_loss, "loss/val": va_loss, "f1/train": tr_f1,
                     "f1/val": va_f1, "recall/val": va_rec}.items():
            writer.add_scalar(k, v, epoch)

        if va_f1 > best:
            best, bad = va_f1, 0
            torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": args.arch,
                        "img_size": IMG_SIZE, "val_f1": best}, ckpt_path)
            print(f"  saved {ckpt_path} (val macro-F1 {best:.4f})")
        elif epoch >= args.warmup_epochs:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    print(f"done. best val macro-F1 = {best:.4f}")


if __name__ == "__main__":
    main()
