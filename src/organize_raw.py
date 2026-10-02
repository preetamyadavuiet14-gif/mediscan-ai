"""Collect images from downloaded datasets into data/raw/<class>/ (with unique file names).

Example:
  python -m src.organize_raw \
    --map covid19="C:/data/COVID-19_Radiography_Dataset/COVID" \
    --map normal="C:/data/COVID-19_Radiography_Dataset/Normal" \
    --map normal="C:/data/chest_xray/train/NORMAL" \
    --map pneumonia="C:/data/chest_xray/train/PNEUMONIA" \
    --map tuberculosis="C:/data/TB_Chest_Radiography_Database/Tuberculosis"
The same class can be given several times (different sources) - and it SHOULD be, see README.
"""
import argparse
import random
import re
import shutil
from pathlib import Path

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="append", required=True, help="class=folder (repeatable)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit", type=int, default=0,
                    help="copy at most N random images per --map (keeps small PCs happy)")
    args = ap.parse_args()

    for i, item in enumerate(args.map):
        cls, folder = item.split("=", 1)
        src = Path(folder.strip('"'))
        dest = Path(args.out) / cls.lower()
        dest.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"\W+", "", src.parent.name + src.name)[:20] + f"{i}"
        imgs = sorted(f for f in src.rglob("*") if f.suffix.lower() in EXT and f.is_file())
        if args.limit and len(imgs) > args.limit:
            imgs = random.Random(42).sample(imgs, args.limit)
        for f in imgs:
            shutil.copy2(f, dest / f"{tag}_{f.name}")
        n = len(imgs)
        print(f"{cls:14s} <- {src}  ({n} images)")


if __name__ == "__main__":
    main()
