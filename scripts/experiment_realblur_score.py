"""Experiment Part B (reports/2026-07-10_blur_robustness): does REAL camera-shake
blur (RealBlur-J, Rim et al. ECCV 2020) trigger the occlusion detector?

Method: stream-extract a per-scene-capped sample of blur (and matching gt) images
from RealBlur.tar.gz, embed with the production scorer's CLIP (global + 9 tiles),
score P(occluded). Embeddings are SAVED for Part C retraining reuse.

Outputs under reports/2026-07-10_blur_robustness/:
  partB_results.json, imgs_partB/*.jpg (worst offenders, heat overlays)
Data: D:/occlusion_dataset/realblur/j_sample/{blur,gt}/, realblur_embeddings.npz
"""

from __future__ import annotations

import json
import logging
import os
import re
import tarfile

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("partB")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
RB = os.path.join(ROOT, "realblur")
TAR = os.path.join(RB, "RealBlur.tar.gz")
SAMPLE_DIR = os.path.join(RB, "j_sample")
EMB_OUT = os.path.join(RB, "realblur_embeddings.npz")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_blur_robustness")
PER_SCENE = 3
GATE = 0.8
N_GALLERY = 24


def extract_sample():
    """One streaming pass; keep <=PER_SCENE blur imgs per scene + matching gt."""
    os.makedirs(os.path.join(SAMPLE_DIR, "blur"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_DIR, "gt"), exist_ok=True)
    pat = re.compile(r"RealBlur-J[^/]*/(scene\d+)/(blur|gt)/(blur|gt)_(\d+)\.(png|jpg)$",
                     re.IGNORECASE)
    kept: dict[str, set] = {}
    n = 0
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            g = pat.search(m.name.replace("\\", "/"))
            if not g:
                continue
            scene, kind, _, num, ext = g.groups()
            key = f"{scene}_{num}"
            sel = kept.setdefault(scene, set())
            if key not in sel and len(sel) >= PER_SCENE:
                continue
            sel.add(key)
            dst = os.path.join(SAMPLE_DIR, kind.lower(), f"{scene}__{num}.{ext}")
            if not os.path.isfile(dst):
                with tf.extractfile(m) as src, open(dst, "wb") as out:
                    out.write(src.read())
                n += 1
                if n % 200 == 0:
                    logger.info("extracted %d files (%d scenes)", n, len(kept))
    logger.info("extraction done: %d files, %d scenes", n, len(kept))


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
    if not os.path.isdir(os.path.join(SAMPLE_DIR, "blur")) or not any(
            os.scandir(os.path.join(SAMPLE_DIR, "blur"))):
        extract_sample()
    os.makedirs(os.path.join(OUT, "imgs_partB"), exist_ok=True)

    scorer = OcclusionScorer()
    logger.info("artifact %s", scorer.version)
    files = sorted(os.listdir(os.path.join(SAMPLE_DIR, "blur")))
    names, e_glob, e_tile, rows = [], [], [], []
    for k, fn in enumerate(files):
        path = os.path.join(SAMPLE_DIR, "blur", fn)
        pair = scorer.embed(path)
        if pair is None:
            continue
        p, tiles = scorer.score_embeddings(*pair)
        names.append(fn)
        e_glob.append(pair[0])
        e_tile.append(pair[1])
        rows.append({"file": fn, "scene": fn.split("__")[0],
                     "p": round(p, 4), "tiles": [round(t, 3) for t in tiles]})
        if (k + 1) % 100 == 0:
            logger.info("%d/%d scored (over gate so far: %d)",
                        k + 1, len(files), sum(r["p"] >= GATE for r in rows))
    np.savez_compressed(EMB_OUT, names=np.array(names),
                        emb_global=np.array(e_glob), emb_tiles=np.array(e_tile))

    ps = np.array([r["p"] for r in rows])
    scenes = sorted({r["scene"] for r in rows})
    scene_max = {s: max(r["p"] for r in rows if r["scene"] == s) for s in scenes}
    stats = {
        "n_images": len(rows), "n_scenes": len(scenes), "gate": GATE,
        "over_gate": int((ps >= GATE).sum()),
        "over_half": int((ps >= 0.5).sum()),
        "p_max": float(ps.max()), "p_mean": float(ps.mean()),
        "p_median": float(np.median(ps)),
        "scenes_over_gate": int(sum(v >= GATE for v in scene_max.values())),
        "scenes_over_half": int(sum(v >= 0.5 for v in scene_max.values())),
    }

    # gallery: worst offenders (with gt side-by-side when available)
    gallery = []
    for r in sorted(rows, key=lambda r: -r["p"])[:N_GALLERY]:
        img = cv2.imread(os.path.join(SAMPLE_DIR, "blur", r["file"]))
        if img is None:
            continue
        name = "B__" + os.path.splitext(r["file"])[0] + ".jpg"
        heat_overlay(img, r["tiles"], os.path.join(OUT, "imgs_partB", name))
        gt = os.path.join(SAMPLE_DIR, "gt", r["file"].replace("blur", "gt"))
        gt_name = None
        gt_img = cv2.imread(gt) if os.path.isfile(gt) else None
        if gt_img is not None:
            gt_name = "B__gt__" + os.path.splitext(r["file"])[0] + ".jpg"
            h, w = gt_img.shape[:2]
            s = 480 / max(h, w)
            if s < 1:
                gt_img = cv2.resize(gt_img, (int(w * s), int(h * s)))
            cv2.imwrite(os.path.join(OUT, "imgs_partB", gt_name), gt_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
        gallery.append({"file": r["file"], "p": r["p"],
                        "img": f"imgs_partB/{name}",
                        "gt_img": f"imgs_partB/{gt_name}" if gt_name else None})

    with open(os.path.join(OUT, "partB_results.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "gallery": gallery,
                   "scene_max": {k: round(v, 4) for k, v in scene_max.items()}},
                  f, indent=2)
    logger.info("PART B: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
