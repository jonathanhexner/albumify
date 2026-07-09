"""Train + serialize the production occlusion probe (spec-097 T1.1) and run the
two user-mandated validations before it may ship:

1. BLUR-vs-OCCLUSION separation: motion-blurred and defocused variants of clean
   negatives must stay below the penalty gate — proof we learned "occluder",
   not "blur" (bokeh/night-shot safety).
2. EXPLAINABILITY: on every positive, per-tile scores must peak on the occluded
   region (gallery + hot-tile-vs-LoG-box agreement stat), not on background.

Artifact: models/occlusion/clip_b32_gmax_v1.npz (mean, scale, coef, intercept,
version, metadata). Training data: clip_embeddings.npz labels corrected by
corrections.csv (adjudicated), class x 1/group-size weights — the exact recipe
the 0.86 benchmark number was measured with.
"""

from __future__ import annotations

import csv
import json
import logging
import os

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_occlusion")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(REPO, "models", "occlusion", "clip_b32_gmax_v1.npz")
VERSION = "clip_b32_gmax_v1"
GATE = 0.8
N_BLUR_SAMPLE = 60


def corrected_labels(ids, y_orig):
    with open(os.path.join(ROOT, "corrections.csv"), newline="", encoding="utf-8") as f:
        corr = {r["id"]: r["decision"] for r in csv.DictReader(f)}
    y = y_orig.copy().astype(int)
    for i, rid in enumerate(ids):
        d = corr.get(str(rid))
        if d and d != "foreground_object":
            y[i] = 1 if d.startswith("occluded") else 0
    return y


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    y = corrected_labels(ids, d["y"].astype(int))
    g = d["groups"]
    X = np.concatenate([d["emb_global"], d["emb_tiles"].max(axis=1)], axis=1)

    gsize = {gg: int((g == gg).sum()) for gg in set(g.tolist())}
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    cls_w = {1: len(y) / (2.0 * n_pos), 0: len(y) / (2.0 * n_neg)}
    w = np.array([cls_w[int(yi)] / gsize[gg] for yi, gg in zip(y, g)])

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.01)
    clf.fit(sc.transform(X), y, sample_weight=w)

    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    np.savez_compressed(
        ARTIFACT, mean=sc.mean_, scale=sc.scale_, coef=clf.coef_,
        intercept=np.array(clf.intercept_[0]), version=VERSION,
        meta=json.dumps({"trained": "2026-07-09", "n_images": len(y),
                         "n_pos": int(y.sum()), "clip": "ViT-B/32",
                         "benchmark": "spec-096 FINAL 0.86 scene PR-AUC"}))
    logger.info("artifact saved: %s (%d imgs, %d pos)", ARTIFACT, len(y), int(y.sum()))
    return d, ids, y


def validate_blur(scorer, d, ids, y):
    """User gate: blurred CLEAN images must not cross the penalty threshold."""
    from PIL import Image
    neg_ids = [i for i in range(len(ids)) if y[i] == 0]
    step = max(1, len(neg_ids) // N_BLUR_SAMPLE)
    sample = neg_ids[::step][:N_BLUR_SAMPLE]
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        sub = {r["id"]: ("positives" if r["label"] == "1" else "negatives")
               for r in csv.DictReader(f)}
    flagged = {"motion": 0, "defocus": 0, "clean": 0}
    scored = 0
    for i in sample:
        p = os.path.join(ROOT, sub[str(ids[i])], str(ids[i]))
        pair = scorer.embed(p)
        if pair is None:
            continue
        img = cv2.imread(p)
        if img is None:  # heic — read via scorer's PIL path instead
            from PIL import ImageOps
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                pass
            img = cv2.cvtColor(np.array(ImageOps.exif_transpose(
                Image.open(p)).convert("RGB")), cv2.COLOR_RGB2BGR)
        scored += 1
        flagged["clean"] += scorer.score_embeddings(*pair)[0] >= GATE
        # motion blur: strong linear kernel
        k = np.zeros((25, 25), np.float32)
        k[12, :] = 1.0 / 25
        variants = {"motion": cv2.filter2D(img, -1, k),
                    "defocus": cv2.GaussianBlur(img, (0, 0), 6)}
        for tag, v in variants.items():
            tmp = os.path.join(ROOT, "_tmp_blur.jpg")
            cv2.imwrite(tmp, v)
            pr = scorer.embed(tmp)
            if pr is not None and scorer.score_embeddings(*pr)[0] >= GATE:
                flagged[tag] += 1
    if os.path.isfile(os.path.join(ROOT, "_tmp_blur.jpg")):
        os.remove(os.path.join(ROOT, "_tmp_blur.jpg"))
    pos_scores = []
    dd = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    for i in range(len(ids)):
        if y[i] == 1:
            pos_scores.append(scorer.score_embeddings(dd["emb_global"][i], dd["emb_tiles"][i])[0])
    pos_flag = sum(s >= GATE for s in pos_scores)
    logger.info("BLUR CHECK (gate %.1f): clean %d/%d flagged | motion-blurred %d/%d | "
                "defocused %d/%d | REAL OCCLUDED %d/%d flagged",
                GATE, flagged["clean"], scored, flagged["motion"], scored,
                flagged["defocus"], scored, pos_flag, len(pos_scores))
    # cast: cv2/numpy comparisons yield numpy ints, which json.dumps rejects
    return {"n": int(scored), **{k: int(v) for k, v in flagged.items()},
            "pos_flagged": int(pos_flag), "n_pos": len(pos_scores)}


def explain_positives(scorer, d, ids, y):
    """User gate: tile heatmaps on all positives -> gallery + hot-corner stat."""
    import html as H
    from sim_bench.occlusion_bench.saliency import blur_bbox
    out_dir = os.path.join(ROOT, "research_saliency", "tile_explain")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        sub = {r["id"]: ("positives" if r["label"] == "1" else "negatives")
               for r in csv.DictReader(f)}
    rows, agree, with_box = [], 0, 0
    for i in range(len(ids)):
        if y[i] != 1:
            continue
        rid = str(ids[i])
        path = os.path.join(ROOT, sub[rid], rid)
        p, tiles = scorer.score_embeddings(d["emb_global"][i], d["emb_tiles"][i])
        img = cv2.imread(path)
        if img is None:
            try:
                from PIL import Image, ImageOps
                from pillow_heif import register_heif_opener
                register_heif_opener()
                img = cv2.cvtColor(np.array(ImageOps.exif_transpose(
                    Image.open(path)).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                continue
        h, w = img.shape[:2]
        t = np.array(tiles).reshape(3, 3)
        heat = cv2.resize((t * 255).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
        hy, hx = np.unravel_index(t.argmax(), t.shape)
        cv2.rectangle(vis, (hx * w // 3, hy * h // 3),
                      ((hx + 1) * w // 3, (hy + 1) * h // 3), (255, 255, 255), 4)
        bb = blur_bbox(path)
        if bb is not None:
            with_box += 1
            bx0, by0, bx1, by1 = bb
            # hot tile center inside the LoG box (fraction coords)?
            cx, cy = (hx + 0.5) / 3, (hy + 0.5) / 3
            hit = bx0 <= cx <= bx1 and by0 <= cy <= by1
            agree += hit
            cv2.rectangle(vis, (int(bx0 * w), int(by0 * h)),
                          (int(bx1 * w), int(by1 * h)), (0, 0, 255), 3)
        name = f"tiles__{rid}".replace(".heic", ".jpg")
        cv2.imwrite(os.path.join(out_dir, name), vis)
        rows.append((rid, p, name))
    rows.sort(key=lambda r: -r[1])
    body = "".join(
        f"<tr><td><img src='{n}' style='max-width:420px'></td>"
        f"<td>{H.escape(r)}<br>P(occluded)={p:.2f}</td></tr>" for r, p, n in rows)
    doc = ("<html><head><meta charset='utf-8'><style>body{font-family:Segoe UI}"
           "table{border-collapse:collapse}td{border:1px solid #ccc;padding:6px}</style>"
           f"</head><body><h2>Tile-attention on all {len(rows)} positives</h2>"
           f"<p>white box = hottest tile; red = LoG blur box. Hot tile inside LoG box: "
           f"<b>{agree}/{with_box}</b> (of positives where a box exists).</p>"
           f"<table>{body}</table></body></html>")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    logger.info("EXPLAIN CHECK: hot tile inside LoG box %d/%d; gallery %s",
                agree, with_box, os.path.join(out_dir, "index.html"))
    return {"agree": agree, "with_box": with_box, "n_pos": len(rows)}


if __name__ == "__main__":
    d, ids, y = train()
    from sim_bench.occlusion_bench.scorer import OcclusionScorer
    scorer = OcclusionScorer()
    logger.info("loaded artifact version=%s", scorer.version)
    blur_stats = validate_blur(scorer, d, ids, y)
    exp_stats = explain_positives(scorer, d, ids, y)
    print(json.dumps({"blur": blur_stats, "explain": exp_stats}, indent=2))
