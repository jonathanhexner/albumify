"""Crop-verify experiment (spec-097 research): can CLIP separate blur-box crops?

The LoG proposer boxes 47 regions on the 122 Budapest images (2 real fingers +
45 false alarms, mostly sky / dark evening foregrounds). Question: does CLIP —
zero-shot, on the CROP alone — rank the fingers above the sky?

Three prompt ensembles are scored (defect-semantic / quality-contrast /
combined); output is a ranked HTML gallery + the fingers' ranks per ensemble.
"""

from __future__ import annotations

import csv
import html
import logging
import os

import numpy as np
import torch
from PIL import Image

import clip
from sim_bench.occlusion_bench.saliency import _read, blur_bbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("crop_verify")

ROOT = r"D:\occlusion_dataset"
OUT = os.path.join(ROOT, "research_saliency", "crop_probe")

OCCLUDER_PROMPTS = [
    "a blurry finger covering the camera lens",
    "a dark out-of-focus object blocking the camera lens",
    "a smudge on the camera lens",
    "an out-of-focus blob covering part of the photo",
]
SCENE_PROMPTS = [
    "the sky", "clouds in the sky", "a lake", "water", "a building",
    "a wall", "trees", "a crowd of people", "pavement", "a dark night scene",
]
BLURRY_PROMPTS = ["a blurry smudged unclear region", "an out of focus blurry area"]
SHARP_PROMPTS = ["a sharp, clear, detailed photo region", "a well focused photograph"]

ENSEMBLES = {
    "defect_semantic": (OCCLUDER_PROMPTS, SCENE_PROMPTS),
    "quality_contrast": (BLURRY_PROMPTS, SHARP_PROMPTS),
    "combined": (OCCLUDER_PROMPTS + BLURRY_PROMPTS, SCENE_PROMPTS + SHARP_PROMPTS),
}


def budapest_rows() -> list:
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f) if r["source_dataset"] == "budapest"}
    return list(rows.values())


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    crops, meta = [], []
    for r in budapest_rows():
        bb = blur_bbox(r["source_path"])
        if bb is None:
            continue
        img = _read(r["source_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        x0, y0, x1, y1 = int(bb[0] * w), int(bb[1] * h), int(bb[2] * w), int(bb[3] * h)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        pil = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
        name = f"crop__{r['id']}"
        pil.save(os.path.join(OUT, name), "JPEG")
        crops.append(preprocess(pil))
        meta.append({"id": r["id"], "label": r["label"], "file": name,
                     "top_frac": round((bb[1] + bb[3]) / 2, 2)})
    logger.info("%d crops (%d true fingers)", len(meta),
                sum(1 for m in meta if m["label"] == "1"))

    with torch.no_grad():
        emb = model.encode_image(torch.stack(crops)).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        for tag, (pos, neg) in ENSEMBLES.items():
            toks = clip.tokenize(pos + neg)
            t = model.encode_text(toks).float()
            t = t / t.norm(dim=-1, keepdim=True)
            probs = (100.0 * emb @ t.T).softmax(dim=-1).numpy()
            score = probs[:, :len(pos)].sum(axis=1)  # P(any occluder prompt)
            for m, s in zip(meta, score):
                m[tag] = float(s)

    # fingers' ranks per ensemble (1 = most occluder-like of all 47 crops)
    summary = {}
    for tag in ENSEMBLES:
        order = sorted(range(len(meta)), key=lambda i: -meta[i][tag])
        ranks = [order.index(i) + 1 for i, m in enumerate(meta) if m["label"] == "1"]
        summary[tag] = ranks
        logger.info("%s: finger ranks %s of %d", tag, ranks, len(meta))

    rows_html = []
    for m in sorted(meta, key=lambda m: -m["combined"]):
        tint = "#fee2e2" if m["label"] == "1" else "#fff"
        rows_html.append(
            f"<tr style='background:{tint}'><td><img src='{m['file']}' style='max-width:220px;max-height:150px'></td>"
            f"<td>{html.escape(m['id'])}<br><b>{'REAL FINGER' if m['label'] == '1' else 'clean image'}</b></td>"
            f"<td>{m['defect_semantic']:.3f}</td><td>{m['quality_contrast']:.3f}</td>"
            f"<td><b>{m['combined']:.3f}</b></td></tr>")
    doc = ("<html><head><meta charset='utf-8'><style>body{font-family:Segoe UI}"
           "table{border-collapse:collapse}td{border:1px solid #ccc;padding:6px}</style></head><body>"
           f"<h2>Crop-verify: CLIP zero-shot on the {len(meta)} LoG candidate boxes (Budapest)</h2>"
           f"<p>finger ranks — defect_semantic: {summary['defect_semantic']}, "
           f"quality_contrast: {summary['quality_contrast']}, combined: {summary['combined']} "
           f"(of {len(meta)}; 1 = most occluder-like). Sorted by combined score.</p>"
           "<table><tr><th>crop</th><th>source</th><th>defect_semantic</th>"
           "<th>quality_contrast</th><th>combined</th></tr>"
           + "".join(rows_html) + "</table></body></html>")
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    logger.info("gallery: %s", os.path.join(OUT, "index.html"))


if __name__ == "__main__":
    main()
