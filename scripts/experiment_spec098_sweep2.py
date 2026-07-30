"""
spec-098 sweep v2: sigma-corrected sharpness.

corrected = max(lap_med - k * sigma^2, 0)
Rationale: additive white noise with std sigma contributes ~20*sigma^2 to raw
Laplacian variance (kernel sum-of-squares = 20); after a median blur a residual
fraction remains. Fit k empirically on SIDD (inflation < 2x gate) and verify on
RealBlur (pair acc >= 99% gate).
"""

import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.noise_robust import estimate_noise_sigma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sweep2")

SIDD = Path(r"D:\DataSets\SIDD_Small_sRGB_Only")
REALBLUR = Path(r"D:\occlusion_dataset\realblur\j_sample")
KS = [0.0, 5.0, 10.0, 20.0, 40.0]
MEDIANS = {"med3": 3, "med5": 5}


def feats(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    sigma = estimate_noise_sigma(img_bgr)
    lap = {name: float(cv2.Laplacian(cv2.medianBlur(gray, k), cv2.CV_64F).var())
           for name, k in MEDIANS.items()}
    return lap, sigma


def corrected(lap, sigma, k):
    return max(lap - k * sigma * sigma, 0.0)


def main():
    rng = random.Random(42)

    log.info("SIDD...")
    sidd = []  # (iso, feats_gt, feats_noisy)
    scenes = sorted(d for d in SIDD.iterdir() if d.is_dir())
    for si, scene in enumerate(scenes):
        gts = sorted(scene.glob("GT_SRGB_*.png"))
        for gt_path in rng.sample(gts, min(3, len(gts))):
            noisy_path = scene / gt_path.name.replace("GT_", "NOISY_")
            if noisy_path.exists():
                sidd.append((int(scene.name.split("_")[3]),
                             feats(cv2.imread(str(gt_path))), feats(cv2.imread(str(noisy_path)))))
        if (si + 1) % 60 == 0:
            log.info("sidd scene %d/%d", si + 1, len(scenes))

    log.info("RealBlur...")
    rb = []  # (feats_gt, feats_blur)
    names = sorted(p.name for p in (REALBLUR / "gt").glob("*.png"))
    for i, name in enumerate(names):
        rb.append((feats(cv2.imread(str(REALBLUR / "gt" / name))),
                   feats(cv2.imread(str(REALBLUR / "blur" / name)))))
        if (i + 1) % 200 == 0:
            log.info("realblur %d/%d", i + 1, len(names))

    print(f"SIDD pairs={len(sidd)}  RealBlur pairs={len(rb)}")
    print(f"{'median':7}{'k':>6}  {'SIDD inflation':>14}  {'SIDD sharp acc':>14}  {'RealBlur acc':>13}")
    for med in MEDIANS:
        for k in KS:
            c_gt = np.array([corrected(g[0][med], g[1], k) for _, g, n in sidd])
            c_ns = np.array([corrected(n[0][med], n[1], k) for _, g, n in sidd])
            infl = np.median(c_ns) / max(np.median(c_gt), 1e-9)
            sidd_acc = float(np.mean(c_gt > c_ns))
            rb_acc = float(np.mean(
                [corrected(g[0][med], g[1], k) > corrected(b[0][med], b[1], k) for g, b in rb]))
            print(f"{med:7}{k:6.0f}  {infl:13.2f}x  {sidd_acc:14.3f}  {rb_acc:13.4f}")
    print("gates: inflation < 2x, RealBlur acc >= 0.99")


if __name__ == "__main__":
    main()
