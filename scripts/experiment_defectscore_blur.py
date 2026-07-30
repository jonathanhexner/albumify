"""
Defect-scoring benchmark #1: BLUR (spec-free experiment, report-mandate applies).

Dataset: RealBlur-J sample, 702 paired (blur, gt) images per scene.
Compares 3 scorers on ability to rank the sharp image above its blurred twin:
  - ours      : Laplacian variance (pipeline sharpness, rule_based.py) + its /1000-clip normalization
  - classical : Tenengrad (Sobel gradient energy)
  - sota      : MUSIQ (pyiqa, no-reference deep IQA)

Outputs reports/2026-07-10_defect_blur/{data/scores.csv, data/metrics.json, samples/*, report.html, summary.md}
"""

import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.noise_robust import noise_robust_laplacian_var

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("defect_blur")

DATA_ROOT = Path(r"D:\occlusion_dataset\realblur\j_sample")
# DEFECTSCORE_OUT lets validation re-runs (spec-098) write elsewhere instead of
# overwriting the 2026-07-10 baseline report.
REPORT_DIR = Path(os.environ.get(
    "DEFECTSCORE_OUT",
    Path(__file__).resolve().parents[1] / "reports" / "2026-07-10_defect_blur"))
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # pairs; 0 = all


def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g = np.sqrt(sx ** 2 + sy ** 2)
    m = g[g > g.mean()]
    return float(np.mean(m ** 2)) if m.size else 0.0  # mean (not sum) -> resolution invariant


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC via rank statistic: P(score_sharp > score_blur)."""
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def main():
    (REPORT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "samples").mkdir(exist_ok=True)

    names = sorted(p.name for p in (DATA_ROOT / "gt").glob("*.png"))
    if LIMIT:
        names = names[:LIMIT]
    log.info("pairs: %d", len(names))

    import pyiqa
    musiq = pyiqa.create_metric("musiq", device="cpu")

    rows = []
    t0 = time.time()
    for i, name in enumerate(names):
        row = {"name": name, "scene": name.split("__")[0]}
        for kind in ("gt", "blur"):
            p = DATA_ROOT / kind / name
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            row[f"lap_{kind}"] = lap_var(gray)
            row[f"rlap_{kind}"] = noise_robust_laplacian_var(gray)  # spec-098
            row[f"ten_{kind}"] = tenengrad(gray)
            row[f"musiq_{kind}"] = float(musiq(str(p)).item())
        rows.append(row)
        if (i + 1) % 50 == 0:
            log.info("%d/%d (%.1fs)", i + 1, len(names), time.time() - t0)

    with open(REPORT_DIR / "data" / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- metrics ----
    metrics = {"n_pairs": len(rows), "dataset": str(DATA_ROOT)}
    ours_norm_gt = np.array([min(r["lap_gt"] / 1000.0, 1.0) for r in rows])
    ours_norm_bl = np.array([min(r["lap_blur"] / 1000.0, 1.0) for r in rows])
    for key, label in [("lap", "ours_laplacian"), ("rlap", "ours_noise_robust_laplacian"),
                       ("ten", "classical_tenengrad"), ("musiq", "sota_musiq")]:
        g = np.array([r[f"{key}_gt"] for r in rows])
        b = np.array([r[f"{key}_blur"] for r in rows])
        metrics[label] = {
            "pair_acc": float(np.mean(g > b)),
            "auc": auc(g, b),
            "median_gt": float(np.median(g)),
            "median_blur": float(np.median(b)),
        }
    # normalization saturation: pairs where our /1000-clipped score ties
    metrics["ours_normalized"] = {
        "pair_acc": float(np.mean(ours_norm_gt > ours_norm_bl)),
        "tie_rate": float(np.mean(ours_norm_gt == ours_norm_bl)),
        "saturated_at_1": float(np.mean((ours_norm_gt == 1.0) & (ours_norm_bl == 1.0))),
    }

    # ---- samples: 2 wins + 2 failures of ours (by lap ranking), downscaled ----
    lap_margin = np.array([r["lap_gt"] - r["lap_blur"] for r in rows])
    picks = [int(np.argmax(lap_margin)), int(np.argsort(lap_margin)[len(rows) // 2]),
             int(np.argmin(lap_margin)), int(np.argsort(lap_margin)[1])]
    sample_meta = []
    for idx in dict.fromkeys(picks):
        r = rows[idx]
        for kind in ("gt", "blur"):
            img = cv2.imread(str(DATA_ROOT / kind / r["name"]))
            h, w0 = img.shape[:2]
            s = 420.0 / max(h, w0)
            out = REPORT_DIR / "samples" / f"{r['scene']}_{Path(r['name']).stem}_{kind}.jpg"
            cv2.imwrite(str(out), cv2.resize(img, (int(w0 * s), int(h * s))))
        sample_meta.append(r)
    metrics["samples"] = sample_meta

    with open(REPORT_DIR / "data" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("done in %.1fs -> %s", time.time() - t0, REPORT_DIR)


if __name__ == "__main__":
    main()
