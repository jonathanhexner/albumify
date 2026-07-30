"""
spec-098 sweep v3: overall-score pair accuracy on SIDD with the FINAL sharpness
formula (med3 + max(lap - 5*sigma^2, 0)) across candidate weight sets.
Gate A1: overall pair acc >= 0.95 (and high-ISO subset reported).
"""

import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.noise_robust import estimate_noise_sigma, noise_sigma_to_score
from sim_bench.quality_assessment.rule_based import RuleBasedQuality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sweep3")

DATA_ROOT = Path(r"D:\DataSets\SIDD_Small_sRGB_Only")
K = 5.0

WEIGHT_SETS = {
    "w_v2":      {"sharpness": .35, "exposure": .25, "colorfulness": .15, "contrast": .10, "noise": .15},
    "w_noise20": {"sharpness": .30, "exposure": .25, "colorfulness": .15, "contrast": .10, "noise": .20},
    "w_noise25": {"sharpness": .30, "exposure": .25, "colorfulness": .10, "contrast": .10, "noise": .25},
    "w_noise30": {"sharpness": .30, "exposure": .25, "colorfulness": .05, "contrast": .10, "noise": .30},
}


def components(img, rb):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = estimate_noise_sigma(img)
    lap = float(cv2.Laplacian(cv2.medianBlur(gray, 3), cv2.CV_64F).var())
    return {
        "sharp_norm": min(max(lap - K * sigma * sigma, 0.0) / 1000.0, 1.0),
        "exposure": rb._compute_exposure_quality(gray),
        "color": rb._normalize_colorfulness(rb._compute_colorfulness(img)),
        "contrast": rb._compute_contrast(gray),
        "noise": noise_sigma_to_score(sigma),
    }


def overall(c, w):
    return (w["sharpness"] * c["sharp_norm"] + w["exposure"] * c["exposure"]
            + w["colorfulness"] * c["color"] + w["contrast"] * c["contrast"] + w["noise"] * c["noise"])


def main():
    rng = random.Random(42)
    rb = RuleBasedQuality()
    pairs = []
    scenes = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    for si, scene in enumerate(scenes):
        gts = sorted(scene.glob("GT_SRGB_*.png"))
        for gt_path in rng.sample(gts, min(3, len(gts))):
            noisy_path = scene / gt_path.name.replace("GT_", "NOISY_")
            if noisy_path.exists():
                pairs.append((int(scene.name.split("_")[3]),
                              components(cv2.imread(str(gt_path)), rb),
                              components(cv2.imread(str(noisy_path)), rb)))
        if (si + 1) % 60 == 0:
            log.info("scene %d/%d", si + 1, len(scenes))

    hi = [p for p in pairs if p[0] >= 1600]
    print(f"pairs={len(pairs)} hi_iso={len(hi)}  (sharpness = med3, k={K})")
    for wname, w in WEIGHT_SETS.items():
        acc = np.mean([overall(g, w) > overall(n, w) for _, g, n in pairs])
        acc_hi = np.mean([overall(g, w) > overall(n, w) for _, g, n in hi])
        print(f"{wname:10} overall_acc={acc:.3f}  hi_iso={acc_hi:.3f}")
    print("gate A1: overall_acc >= 0.95")


if __name__ == "__main__":
    main()
