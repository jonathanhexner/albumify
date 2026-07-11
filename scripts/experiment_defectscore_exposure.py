"""
Defect-scoring benchmark #3: OVER/UNDER-EXPOSURE.

Dataset: PIQA23 Exposure attribute -- 5116 smartphone images with human JOD
(just-objectionable-difference) quality scores from pairwise comparisons
(Scores_Exposure.csv). Higher JOD = better perceived quality.

Scorers (higher = better):
  - ours      : RuleBasedQuality._compute_exposure_quality (histogram clipping + entropy)
  - classical : mid-gray brightness score = 1 - 2*|mean/255 - 0.5|
  - sota      : MUSIQ + CLIPIQA (pyiqa)

Evaluation: Spearman rank correlation (SROCC) vs JOD -- global and per-scene mean.
Sampling: stratified per scene (JOD quantiles) to cap deep-metric wall time.

Outputs reports/2026-07-10_defect_exposure/{data/*, samples/*}
"""

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.rule_based import RuleBasedQuality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("defect_exposure")

DATA_ROOT = Path(r"D:\DataSets\PIQA23")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "2026-07-10_defect_exposure"
PER_SCENE = 30  # stratified cap per scene (~7.5s/img for MUSIQ+CLIPIQA at native res)
LIMIT_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def srocc(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def main():
    (REPORT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "samples").mkdir(exist_ok=True)

    by_scene = defaultdict(list)
    with open(DATA_ROOT / "Scores_Exposure.csv", newline="") as f:
        for rec in csv.DictReader(f):
            by_scene[rec["SCENE"]].append(
                {"path": DATA_ROOT / rec["IMAGE PATH"], "jod": float(rec["JOD"]), "scene": rec["SCENE"]})

    # stratified sample: sort by JOD, take evenly spaced PER_SCENE per scene
    items = []
    for scene, lst in sorted(by_scene.items()):
        lst.sort(key=lambda r: r["jod"])
        idx = np.unique(np.linspace(0, len(lst) - 1, min(PER_SCENE, len(lst))).astype(int))
        items.extend(lst[i] for i in idx)
    if LIMIT_TOTAL:
        items = items[:LIMIT_TOTAL]
    log.info("scenes=%d sampled=%d of %d", len(by_scene), len(items), sum(map(len, by_scene.values())))

    import pyiqa
    musiq = pyiqa.create_metric("musiq", device="cpu")
    clipiqa = pyiqa.create_metric("clipiqa", device="cpu")
    rb = RuleBasedQuality()

    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        gray = cv2.imread(str(it["path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            log.warning("unreadable: %s", it["path"])
            continue
        rows.append({
            "name": it["path"].name, "scene": it["scene"], "jod": it["jod"],
            "ours_exposure": rb._compute_exposure_quality(gray),
            "classical_midgray": float(1.0 - 2.0 * abs(gray.mean() / 255.0 - 0.5)),
            "sota_musiq": float(musiq(str(it["path"])).item()),
            "sota_clipiqa": float(clipiqa(str(it["path"])).item()),
        })
        if (i + 1) % 100 == 0:
            log.info("%d/%d (%.1fs)", i + 1, len(items), time.time() - t0)

    with open(REPORT_DIR / "data" / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    methods = ["ours_exposure", "classical_midgray", "sota_musiq", "sota_clipiqa"]
    jod = [r["jod"] for r in rows]
    metrics = {"n_images": len(rows), "n_scenes": len(by_scene), "dataset": str(DATA_ROOT / "Exposure")}
    for m in methods:
        per_scene = []
        for scene in by_scene:
            sub = [r for r in rows if r["scene"] == scene]
            if len(sub) >= 10:
                per_scene.append(srocc([r["jod"] for r in sub], [r[m] for r in sub]))
        metrics[m] = {
            "srocc_global": srocc(jod, [r[m] for r in rows]),
            "srocc_per_scene_mean": float(np.mean(per_scene)),
            "srocc_per_scene_min": float(np.min(per_scene)),
            "srocc_per_scene_max": float(np.max(per_scene)),
        }

    # samples: worst / mid / best JOD from 2 scenes (only scenes actually scored)
    picked = []
    scored_scenes = list(dict.fromkeys(r["scene"] for r in rows))
    for scene in scored_scenes[:2]:
        sub = sorted([r for r in rows if r["scene"] == scene], key=lambda r: r["jod"])
        for tag, r in [("low", sub[0]), ("mid", sub[len(sub) // 2]), ("high", sub[-1])]:
            img = cv2.imread(str(DATA_ROOT / "Exposure" / r["name"]))
            h, w0 = img.shape[:2]
            s = 420.0 / max(h, w0)
            cv2.imwrite(str(REPORT_DIR / "samples" / f"{scene}_{tag}_jod{r['jod']:.1f}.jpg"),
                        cv2.resize(img, (int(w0 * s), int(h * s))))
            picked.append(r)
    metrics["samples"] = picked

    with open(REPORT_DIR / "data" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("done in %.1fs -> %s", time.time() - t0, REPORT_DIR)


if __name__ == "__main__":
    main()
