"""Remove duplicates and split data/raw/<class>/* into train/val/test (80/10/10).

Splitting is stratified by class and *grouped*, so images of one patient never land in two splits.
If your file names contain a patient id, pass --group-regex, e.g. for NIH ChestX-ray14
(00000013_005.png) use  --group-regex "^(\\d+)_"   Without it every image is its own group.
"""
import argparse
import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw)
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    files, labels, groups, seen = [], [], [], set()
    for ci, c in enumerate(classes):
        for f in sorted((raw / c).iterdir()):
            if not f.is_file():
                continue
            h = md5(f)
            if h in seen:  # exact duplicate -> skip (duplicates leak between splits)
                continue
            seen.add(h)
            g = f.name
            if args.group_regex:
                m = re.search(args.group_regex, f.name)
                if m:
                    g = c + "_" + (m.group(1) if m.groups() else m.group(0))
            files.append(f); labels.append(ci); groups.append(g)
    print(f"{len(files)} unique images, classes: {classes}")

    y, groups = np.array(labels), np.array(groups)
    folds = [te for _, te in StratifiedGroupKFold(10, shuffle=True, random_state=args.seed)
             .split(np.zeros(len(y)), y, groups)]
    split_of = np.array(["train"] * len(y), dtype=object)
    split_of[folds[0]] = "test"
    split_of[folds[1]] = "val"

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for f, yi, s in zip(files, y, split_of):
        d = out / s / classes[yi]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d / f.name)

    counts = Counter((s, classes[yi]) for yi, s in zip(y, split_of))
    print(f"{'split':6s}" + "".join(f"{c:>14s}" for c in classes))
    for s in ["train", "val", "test"]:
        print(f"{s:6s}" + "".join(f"{counts[(s, c)]:>14d}" for c in classes))


if __name__ == "__main__":
    main()
