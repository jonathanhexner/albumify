"""Crop-verify part 3: REAL positives x DIVERSE negatives (user-designed).

Positives: LoG-box crops of the real occluded photos (adjudicated labels),
           EXCLUDING Budapest (test domain).
Negatives: LoG false-alarm crops mined from MANY sources, capped per source so
           no dataset dominates (user rule: representation over quantity) —
           austria/germany albums + InriaHolidays, ava, PIQA23, INSTRE, SIDD.
Test:      the 47 real Budapest candidate crops (2 fingers + 45 FPs).
"""

from __future__ import annotations

import csv
import glob
import html
import logging
import os

import numpy as np
import torch
from PIL import Image

import clip
from sim_bench.occlusion_bench.saliency import _read, blur_bbox

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("crop_probe_real")

ROOT = r"D:\occlusion_dataset"
CROPS_DIR = os.path.join(ROOT, "research_saliency", "crop_probe")
NEG_CACHE = os.path.join(CROPS_DIR, "train_neg")
POS_CACHE = os.path.join(CROPS_DIR, "train_pos")

# source name -> directory to scan (albums scan via manifest instead)
EXTRA_SOURCES = {
    "inria": r"D:\DataSets\InriaHolidays",
    "ava": r"D:\DataSets\ava",
    "piqa23": r"D:\DataSets\PIQA23",
    "instre": r"D:\DataSets\INSTRE_release",
    "sidd": r"D:\DataSets\SIDD_Small_sRGB_Only",
}
SCAN_CAP = 120   # images scanned per source
CROP_CAP = 40    # FP crops kept per source
EXTS = (".jpg", ".jpeg", ".png", ".heic")


def _crop_for(path: str):
    bb = blur_bbox(path)
    if bb is None:
        return None
    img = _read(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    c = img[int(bb[1] * h):int(bb[3] * h), int(bb[0] * w):int(bb[2] * w)]
    if c.size == 0 or min(c.shape[:2]) < 32:
        return None
    return Image.fromarray(c[:, :, ::-1])


def effective_positive_rows() -> list:
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    with open(os.path.join(ROOT, "corrections.csv"), newline="", encoding="utf-8") as f:
        for c in csv.DictReader(f):
            if c["id"] in rows:
                rows[c["id"]]["label"] = "1" if c["decision"].startswith("occluded") else "0"
    return [r for r in rows.values()
            if r["label"] == "1" and r["source_dataset"] != "budapest"]


def real_positive_crops() -> list:
    os.makedirs(POS_CACHE, exist_ok=True)
    out, missed = [], 0
    for r in effective_positive_rows():
        sub = "positives" if r["id"].startswith(("occl", "germany", "austria")) else "positives"
        p = os.path.join(ROOT, "positives", r["id"])
        if not os.path.isfile(p):  # adjudicated flip still lives under negatives/
            p = os.path.join(ROOT, "negatives", r["id"])
        c = _crop_for(p)
        if c is None:
            missed += 1
            continue
        c.save(os.path.join(POS_CACHE, f"pos__{r['id']}".replace(".heic", ".jpg")), "JPEG")
        out.append(c)
    logger.info("positives: %d real occluder crops (%d had no LoG box)", len(out), missed)
    return out


def album_negative_paths() -> list:
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)
                if r["source_dataset"] in ("austria24", "germany1") and r["label"] == "0"}
    return [("albums", r["source_path"]) for r in rows.values()]


def diverse_negative_crops() -> list:
    os.makedirs(NEG_CACHE, exist_ok=True)
    sources = {"albums": album_negative_paths()}
    for tag, d in EXTRA_SOURCES.items():
        files = sorted(p for p in glob.glob(os.path.join(d, "**", "*.*"), recursive=True)
                       if p.lower().endswith(EXTS))
        step = max(1, len(files) // SCAN_CAP)  # deterministic spread, not first-N
        sources[tag] = [(tag, p) for p in files[::step][:SCAN_CAP]]
        logger.info("source %s: %d files, scanning %d", tag, len(files), len(sources[tag]))
    out = []
    for tag, paths in sources.items():
        kept = 0
        for _, p in paths:
            if kept >= CROP_CAP:
                break
            try:
                c = _crop_for(p)
            except Exception as e:
                logger.warning("neg skip %s: %s", p, e)
                continue
            if c is None:
                continue
            c.save(os.path.join(NEG_CACHE, f"neg_{tag}_{kept:03d}.jpg"), "JPEG")
            out.append(c)
            kept += 1
        logger.info("source %s: kept %d FP crops", tag, kept)
    logger.info("negatives: %d total from %d sources", len(out), len(sources))
    return out


def main() -> None:
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    def embed(pils):
        with torch.no_grad():
            e = model.encode_image(torch.stack([preprocess(p) for p in pils])).float()
        return (e / e.norm(dim=-1, keepdim=True)).numpy()

    pos, neg = real_positive_crops(), diverse_negative_crops()
    X = np.vstack([embed(pos), embed(neg)])
    y = np.array([1] * len(pos) + [0] * len(neg))

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    logger.info("train acc %.3f", clf.score(sc.transform(X), y))

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
    logger.info("REAL+DIVERSE PROBE finger ranks: %s of %d", ranks, len(meta))

    rows_html = [
        f"<tr style='background:{'#fee2e2' if m['label'] == '1' else '#fff'}'>"
        f"<td><img src='{m['file']}' style='max-width:220px;max-height:150px'></td>"
        f"<td>{html.escape(m['id'])}<br><b>{'REAL FINGER' if m['label'] == '1' else 'clean image'}</b></td>"
        f"<td><b>{m['score']:.3f}</b></td></tr>" for m in order]
    doc = ("<html><head><meta charset='utf-8'><style>body{font-family:Segoe UI}"
           "table{border-collapse:collapse}td{border:1px solid #ccc;padding:6px}</style></head><body>"
           f"<h2>REAL-crop probe, diverse negatives — {len(pos)} real occluder crops vs "
           f"{len(neg)} mined FP crops from 6 sources; test = {len(meta)} real Budapest crops</h2>"
           f"<p><b>finger ranks: {ranks} of {len(meta)}</b> (1 = most occluder-like)</p>"
           "<table><tr><th>crop</th><th>source</th><th>probe P(occluder)</th></tr>"
           + "".join(rows_html) + "</table></body></html>")
    with open(os.path.join(CROPS_DIR, "real_probe_index.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    logger.info("gallery: %s", os.path.join(CROPS_DIR, "real_probe_index.html"))


if __name__ == "__main__":
    main()
