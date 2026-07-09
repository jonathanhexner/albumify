"""One-off audit: which Budapest images get a blur box, and where (sky check)."""
import csv
import sys

import numpy as np
from PIL import Image

from sim_bench.occlusion_bench.saliency import blur_bbox

manifest = r"D:\occlusion_dataset\manifest.csv"
rows = []
with open(manifest, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["source_dataset"] == "budapest":
            rows.append(r)
# manifest has duplicate rows; dedupe by id
seen = {}
for r in rows:
    seen[r["id"]] = r
rows = list(seen.values())

out = []
for r in rows:
    try:
        bb = blur_bbox(r["source_path"])
    except Exception as e:
        print("SKIP", r["id"], e)
        continue
    if bb is None:
        continue
    x0, y0, x1, y1 = bb  # fraction coords 0..1
    cy = (y0 + y1) / 2  # 0 = top of frame
    out.append((r["id"], r["label"], round(cy, 2), round((x1 - x0) * (y1 - y0), 3)))

out.sort(key=lambda t: t[2])
print(f"{len(out)} boxes / {len(rows)} images")
for t in out:
    print(t)
