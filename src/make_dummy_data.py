"""Create random fake images so you can test the whole pipeline in minutes (no real data needed)."""
import numpy as np
from pathlib import Path
from PIL import Image

rng = np.random.default_rng(0)
for cls in ["covid19", "normal", "pneumonia", "tuberculosis"]:
    d = Path("data/raw") / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(60):
        base = rng.normal(120, 40, (256, 256)).clip(0, 255)
        Image.fromarray(base.astype(np.uint8)).save(d / f"{cls}_{i}.png")
print("dummy data written to data/raw")
