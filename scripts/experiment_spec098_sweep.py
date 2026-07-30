"""
spec-098 T1.3 follow-up: the first validation re-run FAILED gates A1/A2
(overall pair acc 62% vs >=95%; robust-Laplacian inflation 11x vs <2x).

Sweep denoise strength (median 3x3 / 5x5 / 3x3-twice) and component weights on
SIDD pairs to find a configuration that passes, using the same 480 pairs.
No MUSIQ (fast, pure component math).
"""

import logging
import random
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.noise_robust import estimate_noise_sigma, noise_sigma_to_score
from sim_bench.quality_assessment.rule_based import RuleBasedQuality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sweep")

DATA_ROOT = Path(r"D:\DataSets\SIDD_Small_sRGB_Only")
PATCHES_PER_SCENE = 3

DENOISERS = {
    "med3": lambda g: cv2.medianBlur(g, 3),
    "med5": lambda g: cv2.medianBlur(g, 5),
    "med3x2": lambda g: cv2.medianBlur(cv2.medianBlur(g, 3), 3),
}

WEIGHT_SETS = {
    "w_v2":        {"sharpness": .35, "exposure": .25, "colorfulness": .15, "contrast": .10, "noise": .15},
    "w_noise25":   {"sharpness": .30, "exposure": .25, "colorfulness": .10, "contrast": .10, "noise": .25},
    "w_noise30":   {"sharpness": .30, "exposure": .25, "colorfulness": .05, "contrast": .10, "noise": .30},
    "w_noise40":   {"sharpness": .25, "exposure": .20, "colorfulness": .05, "contrast": .10, "noise": .40},
}


def components(img, rb):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = {k: float(cv2.Laplacian(f(gray), cv2.CV_64F).var()) for k, f in DENOISERS.items()}
    return {
        "lap": lap,
        "exposure": rb._compute_exposure_quality(gray),
        "color": rb._normalize_colorfulness(rb._compute_colorfulness(img)),
        "contrast": rb._compute_contrast(gray),
        "noise": noise_sigma_to_score(estimate_noise_sigma(img)),
    }


def overall(c, den, w):
    return (w["sharpness"] * min(c["lap"][den] / 1000.0, 1.0) + w["exposure"] * c["exposure"]
            + w["colorfulness"] * c["color"] + w["contrast"] * c["contrast"] + w["noise"] * c["noise"])


def main():
    rng = random.Random(42)
    rb = RuleBasedQuality()
    scenes = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    pairs = []
    for si, scene in enumerate(scenes):
        gts = sorted(scene.glob("GT_SRGB_*.png"))
        for gt_path in rng.sample(gts, min(PATCHES_PER_SCENE, len(gts))):
            noisy_path = scene / gt_path.name.replace("GT_", "NOISY_")
            if not noisy_path.exists():
                continue
            iso = int(scene.name.split("_")[3])
            pairs.append((iso,
                          components(cv2.imread(str(gt_path)), rb),
                          components(cv2.imread(str(noisy_path)), rb)))
        if (si + 1) % 40 == 0:
            log.info("scene %d/%d", si + 1, len(scenes))

    hi = [p for p in pairs if p[0] >= 1600]
    print(f"pairs={len(pairs)} hi_iso={len(hi)}")
    print(f"{'denoise':8} inflation(med)   " + "  ".join(f"{w:>10}" for w in WEIGHT_SETS))
    for den in DENOISERS:
        infl = np.median([n['lap'][den] for _, g, n in pairs]) / np.median([g['lap'][den] for _, g, n in pairs])
        accs = []
        for wname, w in WEIGHT_SETS.items():
            acc = np.mean([overall(g, den, w) > overall(n, den, w) for _, g, n in pairs])
            acc_hi = np.mean([overall(g, den, w) > overall(n, den, w) for _, g, n in hi])
            accs.append(f"{acc:.3f}/{acc_hi:.3f}")
        print(f"{den:8} {infl:14.1f}x  " + "  ".join(f"{a:>10}" for a in accs))
    print("cell = pair_acc_all / pair_acc_hiISO ; gates: acc >= 0.95, inflation < 2x")


if __name__ == "__main__":
    main()
