"""Regenerate SLIC-fusion subject-mask panels for the spec-097 status report.

Panel per image: [original+YOLO-free | fused saliency heat | SLIC-refined mask].
Fusion = 0.5 * spectral-residual + 0.5 * CLIP-semantic (4x4 tile<->global cosine).
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image

import clip
from sim_bench.occlusion_bench.saliency import _read, saliency_map

OUT = r"D:\occlusion_dataset\research_saliency\slic_fusion"
IMAGES = {
    "person_finger": r"D:\sim-bench\examples\finger_occlusion\20250822_122617.jpg",
    "girl_sky": None,  # filled from manifest below
    "architecture": None,
}
# resolve budapest source paths from manifest
import csv
with open(r"D:\occlusion_dataset\manifest.csv", newline="", encoding="utf-8") as f:
    src = {r["id"]: r["source_path"] for r in csv.DictReader(f)}
IMAGES["girl_sky"] = src["budapest__20250822_123510.jpg"]
IMAGES["architecture"] = src["budapest__20250822_112359.jpg"]

os.makedirs(OUT, exist_ok=True)
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def clip_semantic_map(img_bgr: np.ndarray, grid: int = 4) -> np.ndarray:
    """Tile<->global cosine similarity map, upsampled to image size, [0,1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    h, w = img_bgr.shape[:2]
    crops = [pil]
    for r in range(grid):
        for c in range(grid):
            crops.append(pil.crop((c * w // grid, r * h // grid,
                                   (c + 1) * w // grid, (r + 1) * h // grid)))
    batch = torch.stack([preprocess(c) for c in crops])
    with torch.no_grad():
        emb = model.encode_image(batch).float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    sims = (emb[1:] @ emb[0]).reshape(grid, grid).numpy()
    m = cv2.resize(sims, (w, h), interpolation=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(m, (0, 0), min(h, w) / 16)
    rng = m.max() - m.min()
    return (m - m.min()) / (rng if rng > 1e-9 else 1.0)


for name, path in IMAGES.items():
    img = _read(path)
    if img is None:
        print("SKIP", name)
        continue
    h, w = img.shape[:2]
    sr = saliency_map(img, "sr")
    cs = clip_semantic_map(img)
    fused = 0.5 * sr + 0.5 * cs

    slic = cv2.ximgproc.createSuperpixelSLIC(img, cv2.ximgproc.SLICO, region_size=32)
    slic.iterate(10)
    labels = slic.getLabels()
    n = slic.getNumberOfSuperpixels()
    scores = np.array([fused[labels == i].mean() for i in range(n)])
    hot = scores > np.percentile(scores, 65)
    mask = hot[labels].astype(np.uint8)
    nc, comp, stats, _ = cv2.connectedComponentsWithStats(mask)
    if nc > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (comp == keep).astype(np.uint8)

    heat = cv2.applyColorMap((fused * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.addWeighted(img, 0.45, heat, 0.55, 0)
    ov = img.copy()
    ov[mask == 0] = (ov[mask == 0] * 0.35).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
    for t, lab in ((img, "original"), (heat, "fused saliency (SR + CLIP-sem)"),
                   (ov, "SLIC-refined subject mask")):
        cv2.putText(t, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    out = os.path.join(OUT, f"slic__{name}.jpg")
    cv2.imwrite(out, np.hstack([img, heat, ov]))
    print("WROTE", out)
