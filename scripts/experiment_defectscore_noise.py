"""
Defect-scoring benchmark #2: NOISE.

Dataset: SIDD_Small_sRGB_Only, 160 real smartphone scenes, pre-cropped patches,
paired GT_SRGB_* / NOISY_SRGB_* per patch index. We sample N patches per scene.

Pipeline has NO noise metric (declared in rule_based.py docstring, never implemented).
Tested scorers (higher = better quality convention, sigma negated):
  - ours_overall  : RuleBasedQuality.assess_image (weighted sharpness/exposure/color/contrast)
  - ours_sharpness: Laplacian variance -- hypothesis: noise INFLATES it (wrong direction)
  - classical     : skimage.restoration.estimate_sigma (wavelet noise sigma, negated)
  - sota          : MUSIQ (pyiqa)

Outputs reports/2026-07-10_defect_noise/{data/*, samples/*}
"""

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.noise_robust import noise_robust_laplacian_var
from sim_bench.quality_assessment.rule_based import RuleBasedQuality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("defect_noise")

DATA_ROOT = Path(r"D:\DataSets\SIDD_Small_sRGB_Only")
# DEFECTSCORE_OUT: see experiment_defectscore_blur.py (spec-098 validation re-runs)
REPORT_DIR = Path(os.environ.get(
    "DEFECTSCORE_OUT",
    Path(__file__).resolve().parents[1] / "reports" / "2026-07-10_defect_noise"))
PATCHES_PER_SCENE = 3
LIMIT_SCENES = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))


def main():
    (REPORT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "samples").mkdir(exist_ok=True)
    rng = random.Random(42)

    scenes = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    if LIMIT_SCENES:
        scenes = scenes[:LIMIT_SCENES]

    import pyiqa
    from skimage.restoration import estimate_sigma
    musiq = pyiqa.create_metric("musiq", device="cpu")
    rb = RuleBasedQuality()

    rows = []
    t0 = time.time()
    for si, scene in enumerate(scenes):
        gts = sorted(scene.glob("GT_SRGB_*.png"))
        picks = rng.sample(gts, min(PATCHES_PER_SCENE, len(gts)))
        for gt_path in picks:
            noisy_path = scene / gt_path.name.replace("GT_", "NOISY_")
            if not noisy_path.exists():
                continue
            # smartphone code e.g. S6/GP/IP + ISO from dir name 0001_001_S6_00100_...
            parts = scene.name.split("_")
            row = {"scene": scene.name, "patch": gt_path.stem.replace("GT_SRGB_", ""),
                   "camera": parts[2], "iso": int(parts[3])}
            for kind, p in (("gt", gt_path), ("noisy", noisy_path)):
                img = cv2.imread(str(p))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                row[f"lap_{kind}"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                row[f"rlap_{kind}"] = noise_robust_laplacian_var(gray)  # spec-098
                row[f"overall_{kind}"] = rb.assess_image(str(p))
                row[f"sigma_{kind}"] = float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))
                row[f"musiq_{kind}"] = float(musiq(str(p)).item())
            rows.append(row)
        if (si + 1) % 20 == 0:
            log.info("scene %d/%d (%.1fs)", si + 1, len(scenes), time.time() - t0)

    with open(REPORT_DIR / "data" / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = {"n_pairs": len(rows), "n_scenes": len(scenes), "dataset": str(DATA_ROOT)}
    for key, label, negate in [("overall", "ours_overall", False),
                               ("lap", "ours_sharpness_laplacian", False),
                               ("rlap", "ours_noise_robust_laplacian", False),
                               ("sigma", "classical_wavelet_sigma", True),
                               ("musiq", "sota_musiq", False)]:
        g = np.array([r[f"{key}_gt"] for r in rows])
        n = np.array([r[f"{key}_noisy"] for r in rows])
        if negate:  # sigma: higher = noisier -> negate for higher-is-better
            g, n = -g, -n
        metrics[label] = {
            "pair_acc": float(np.mean(g > n)),
            "auc": auc(g, n),
            "median_gt": float(np.median(g)),
            "median_noisy": float(np.median(n)),
        }

    # high-ISO subset (>= 1600): where noise is most visible
    hi = [r for r in rows if r["iso"] >= 1600]
    if hi:
        metrics["high_iso_subset"] = {"n": len(hi)}
        for key, label, negate in [("overall", "ours_overall", False),
                                   ("lap", "ours_sharpness_laplacian", False),
                                   ("rlap", "ours_noise_robust_laplacian", False),
                                   ("sigma", "classical_wavelet_sigma", True),
                                   ("musiq", "sota_musiq", False)]:
            g = np.array([r[f"{key}_gt"] for r in hi])
            n = np.array([r[f"{key}_noisy"] for r in hi])
            if negate:
                g, n = -g, -n
            metrics["high_iso_subset"][label] = {"pair_acc": float(np.mean(g > n))}

    # samples: strongest + weakest sigma gap
    gaps = np.array([r["sigma_noisy"] - r["sigma_gt"] for r in rows])
    for idx in {int(np.argmax(gaps)), int(np.argmin(gaps)), int(np.argsort(gaps)[len(rows) // 2])}:
        r = rows[idx]
        scene = DATA_ROOT / r["scene"]
        for kind, prefix in (("gt", "GT_SRGB_"), ("noisy", "NOISY_SRGB_")):
            img = cv2.imread(str(scene / f"{prefix}{r['patch']}.png"))
            out = REPORT_DIR / "samples" / f"{r['scene'][:8]}_{r['patch']}_{kind}.jpg"
            cv2.imwrite(str(out), img)
    metrics["samples"] = [rows[i] for i in
                          {int(np.argmax(gaps)), int(np.argmin(gaps)), int(np.argsort(gaps)[len(rows) // 2])}]

    with open(REPORT_DIR / "data" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("done in %.1fs -> %s", time.time() - t0, REPORT_DIR)


if __name__ == "__main__":
    main()
