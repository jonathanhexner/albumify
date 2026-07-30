"""Crop-verify part 2 (spec-097 research): LEARNED probe on candidate-box crops.

Train (zero manual labels, zero Budapest contamination):
  positives — occluder crops recovered from synthetic composites by diffing
              synth vs base image (exact region, free ground truth);
  negatives — self-mined false-alarm crops: run the LoG box proposer over
              austria/germany clean images, crop whatever it (wrongly) boxes.
Test: the 47 REAL Budapest candidate crops from part 1 (2 fingers + 45 FPs).
"""

from __future__ import annotations

import csv
import glob
import html
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import clip
from sim_bench.occlusion_bench.saliency import _read, blur_bbox

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("crop_probe")

ROOT = r"D:\occlusion_dataset"
CROPS_DIR = os.path.join(ROOT, "research_saliency", "crop_probe")
MAX_POS = 250
MAX_NEG = 200


def synth_positive_crops() -> list:
    """Diff synth vs base -> exact occluder bbox -> crop the synth image."""
    with open(os.path.join(ROOT, "synthetic", "synth_manifest.csv"),
              newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["base_id"].startswith("budapest")]
    out = []
    for r in rows[:MAX_POS * 2]:
        if len(out) >= MAX_POS:
            break
        base_p = os.path.join(ROOT, "negatives", r["base_id"])
        synth_p = os.path.join(ROOT, "synthetic", r["id"])
        try:
            with Image.open(base_p) as im:
                base = ImageOps.exif_transpose(im).convert("RGB")
            base.thumbnail((1600, 1600))  # mirror synthesize.py preprocessing
            synth = Image.open(synth_p).convert("RGB")
            if synth.size != base.size:
                continue
            a = np.asarray(base, np.int16)
            b = np.asarray(synth, np.int16)
            m = np.abs(a - b).sum(axis=2) > 24
            ys, xs = np.where(m)
            if len(ys) < 500:
                continue
            crop = synth.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            if min(crop.size) < 32:
                continue
            out.append(crop)
        except Exception as e:
            logger.warning("synth skip %s: %s", r["id"], e)
    logger.info("positives: %d synthetic occluder crops", len(out))
    return out


def mined_negative_crops() -> list:
    """FP crops: whatever the LoG proposer boxes on austria/germany clean images."""
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)
                if r["source_dataset"] in ("austria24", "germany1") and r["label"] == "0"}
    out = []
    for i, r in enumerate(rows.values()):
        if len(out) >= MAX_NEG:
            break
        bb = blur_bbox(r["source_path"])
        if bb is None:
            continue
        img = _read(r["source_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        crop = img[int(bb[1] * h):int(bb[3] * h), int(bb[0] * w):int(bb[2] * w)]
        if crop.size == 0 or min(crop.shape[:2]) < 32:
            continue
        out.append(Image.fromarray(crop[:, :, ::-1]))
        if len(out) % 50 == 0:
            logger.info("negatives: %d mined (%d images scanned)", len(out), i + 1)
    logger.info("negatives: %d mined FP crops", len(out))
    return out


def main() -> None:
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    def embed(pils):
        with torch.no_grad():
            e = model.encode_image(torch.stack([preprocess(p) for p in pils])).float()
        return (e / e.norm(dim=-1, keepdim=True)).numpy()

    pos, neg = synth_positive_crops(), mined_negative_crops()
    X = np.vstack([embed(pos), embed(neg)])
    y = np.array([1] * len(pos) + [0] * len(neg))

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    logger.info("train acc %.3f (sanity only)", clf.score(sc.transform(X), y))

    # test on the REAL Budapest crops from part 1
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        lab = {r["id"]: r["label"] for r in csv.DictReader(f)}
    files = sorted(glob.glob(os.path.join(CROPS_DIR, "crop__*.jpg")))
    pils = [Image.open(p).convert("RGB") for p in files]
    p_test = clf.predict_proba(sc.transform(embed(pils)))[:, 1]
    meta = [{"file": os.path.basename(p), "id": os.path.basename(p)[6:],
             "label": lab.get(os.path.basename(p)[6:], "?"), "score": float(s)}
            for p, s in zip(files, p_test)]

    order = sorted(meta, key=lambda m: -m["score"])
    ranks = [i + 1 for i, m in enumerate(order) if m["label"] == "1"]
    logger.info("PROBE finger ranks: %s of %d", ranks, len(meta))

    rows_html = [
        f"<tr style='background:{'#fee2e2' if m['label'] == '1' else '#fff'}'>"
        f"<td><img src='{m['file']}' style='max-width:220px;max-height:150px'></td>"
        f"<td>{html.escape(m['id'])}<br><b>{'REAL FINGER' if m['label'] == '1' else 'clean image'}</b></td>"
        f"<td><b>{m['score']:.3f}</b></td></tr>" for m in order]
    doc = ("<html><head><meta charset='utf-8'><style>body{font-family:Segoe UI}"
           "table{border-collapse:collapse}td{border:1px solid #ccc;padding:6px}</style></head><body>"
           f"<h2>Learned crop probe — {len(pos)} synth positives vs {len(neg)} mined FP negatives "
           f"(train: austria/germany only; test: {len(meta)} real Budapest crops)</h2>"
           f"<p><b>finger ranks: {ranks} of {len(meta)}</b> (1 = most occluder-like)</p>"
           "<table><tr><th>crop</th><th>source</th><th>probe P(occluder)</th></tr>"
           + "".join(rows_html) + "</table></body></html>")
    with open(os.path.join(CROPS_DIR, "probe_index.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    logger.info("gallery: %s", os.path.join(CROPS_DIR, "probe_index.html"))


if __name__ == "__main__":
    main()
