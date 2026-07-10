"""Experiment Part A (reports/2026-07-10_blur_robustness): does NATURAL blur in the
user's own albums trigger the occlusion detector?

Method: rank all clean negatives by Laplacian variance (lowest = blurriest),
score the blurriest N with the production OcclusionScorer (P + tile heatmap),
and also report the honest out-of-fold probability (model_scores.csv B_clip_oof)
since the production artifact saw these images in training.

Outputs: reports/2026-07-10_blur_robustness/imgs_partA/*.jpg (heat overlays),
partA_results.json (stats + per-image rows for the report assembler).
"""

from __future__ import annotations

import csv
import json
import logging
import os

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("partA")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_blur_robustness")
N_BLURRIEST = 40
GATE = 0.8


def read_bgr(path: str):
    img = cv2.imread(path)
    if img is None:  # heic
        try:
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = cv2.cvtColor(np.array(ImageOps.exif_transpose(
                Image.open(path)).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("unreadable %s (%s)", path, e)
            return None
    return img


def lap_var(img) -> float:
    h, w = img.shape[:2]
    s = 1024.0 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def heat_overlay(img, tiles, out_path, max_side=480):
    h, w = img.shape[:2]
    t = np.array(tiles).reshape(3, 3)
    heat = cv2.applyColorMap(cv2.resize((t * 255).astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    hy, hx = np.unravel_index(t.argmax(), t.shape)
    cv2.rectangle(vis, (hx * w // 3, hy * h // 3),
                  ((hx + 1) * w // 3, (hy + 1) * h // 3), (255, 255, 255), 4)
    s = max_side / max(h, w)
    if s < 1:
        vis = cv2.resize(vis, (int(w * s), int(h * s)))
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 82])


def main():
    from sim_bench.occlusion_bench.scorer import OcclusionScorer
    os.makedirs(os.path.join(OUT, "imgs_partA"), exist_ok=True)

    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(ROOT, "model_scores.csv"), newline="", encoding="utf-8") as f:
        oof = {r["id"]: float(r["B_clip_oof"]) for r in csv.DictReader(f)}

    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    idx = {rid: i for i, rid in enumerate(ids)}
    scorer = OcclusionScorer()
    logger.info("artifact %s", scorer.version)

    # 1) score ALL negatives from cached embeddings + blurriness rank
    negs = [r for r in rows if r["label"] == "0"]
    scored = []
    for k, r in enumerate(negs):
        rid = r["id"]
        path = os.path.join(ROOT, "negatives", rid)
        img = read_bgr(path)
        if img is None or rid not in idx:
            continue
        i = idx[rid]
        p, tiles = scorer.score_embeddings(d["emb_global"][i], d["emb_tiles"][i])
        scored.append({"id": rid, "path": path, "lapvar": lap_var(img),
                       "p_artifact": p, "p_oof": oof.get(rid), "tiles": tiles})
        if (k + 1) % 100 == 0:
            logger.info("%d/%d negatives scored", k + 1, len(negs))

    # 2) population stats over ALL negatives
    pa = np.array([s["p_artifact"] for s in scored])
    po = np.array([s["p_oof"] for s in scored if s["p_oof"] is not None])
    stats = {
        "n_negatives": len(scored), "gate": GATE,
        "artifact_over_gate": int((pa >= GATE).sum()),
        "artifact_over_half": int((pa >= 0.5).sum()),
        "artifact_max": float(pa.max()),
        "oof_over_gate": int((po >= GATE).sum()),
        "oof_over_half": int((po >= 0.5).sum()),
        "oof_max": float(po.max()),
    }

    # 3) blurriest N -> heat-overlay gallery
    blurriest = sorted(scored, key=lambda s: s["lapvar"])[:N_BLURRIEST]
    gallery = []
    for s in blurriest:
        img = read_bgr(s["path"])
        name = ("A__" + s["id"]).replace(".heic", ".jpg").replace(".HEIC", ".jpg")
        heat_overlay(img, s["tiles"], os.path.join(OUT, "imgs_partA", name))
        gallery.append({"id": s["id"], "img": f"imgs_partA/{name}",
                        "lapvar": round(s["lapvar"], 1),
                        "p_artifact": round(s["p_artifact"], 4),
                        "p_oof": round(s["p_oof"], 4) if s["p_oof"] is not None else None})
    bl_pa = np.array([g["p_artifact"] for g in gallery])
    bl_po = np.array([g["p_oof"] for g in gallery if g["p_oof"] is not None])
    stats["blurriest_n"] = len(gallery)
    stats["blurriest_artifact_over_gate"] = int((bl_pa >= GATE).sum())
    stats["blurriest_artifact_max"] = float(bl_pa.max())
    stats["blurriest_oof_over_gate"] = int((bl_po >= GATE).sum())
    stats["blurriest_oof_max"] = float(bl_po.max())

    with open(os.path.join(OUT, "partA_results.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "gallery": gallery}, f, indent=2)
    logger.info("PART A: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
