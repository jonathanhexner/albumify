"""
spec-100 T2: GeoCalib learned-tilt benchmark on real album photos.

Same relative-recovery protocol as the classical spec-099 benchmark
(scripts/experiment_tilt_benchmark.py) so results are directly comparable:
EXIF-normalize each Budapest photo, inject known roll (+-2,4,6,8,10 deg) by
rotate + central crop, measure delta_hat = angle(rotated) - angle(original) - d.
Confidence gating is on GeoCalib's native roll uncertainty (degrees).

Gates (spec-100): G1 MAE<1.5 deg + >=80% of >=4 deg within +-2 deg; G2 coverage>=70%;
G3 <=5% originals flagged >3 deg AND the 3 named classical-FP cases NOT flagged;
G4 ROC-AUC>=0.85 separating injected-tilt from upright; G5 <3 s/img.

Kept as a separate script from the Hough benchmark so spec-099's committed
report (reports/2026-07-12_tilt_benchmark/) stays reproducible.
Output: reports/2026-07-12_geocalib_tilt/.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.tilt_geocalib import estimate_tilt_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("geocalib_tilt")

SRC = Path(r"D:\Budapest2025_Google")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "2026-07-12_geocalib_tilt"
ANGLES = [-10.0, -8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0, 10.0]
# "Confident" if GeoCalib's roll uncertainty (deg) <= gate. Swept.
UNC_GATES = [0.5, 1.0, 1.5, 2.0, 3.0]
WORK_SIDE = 1400  # rotate at this size; wrapper downscales to 1024 internally
# Originals the classical Hough estimator confidently mis-flagged (>3 deg @ gate 0.3):
# upright photos with slanted scenery. GeoCalib must NOT confidently flag these.
NAMED_FP = ["20250822_194451", "20250822_195832", "20250824_170838"]


def rotated_crop(rgb: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate content clockwise by angle_deg; central 72% crop excludes corners."""
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
    rot = cv2.warpAffine(rgb, m, (w, h))
    mh, mw = int(0.14 * h), int(0.14 * w)
    return rot[mh:h - mh, mw:w - mw]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum (Mann-Whitney) identity; no sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {i: (csum[i] - counts[i] + 1 + csum[i]) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N photos (0=all)")
    args = ap.parse_args()

    (REPORT_DIR / "samples").mkdir(parents=True, exist_ok=True)
    photos = sorted(SRC.rglob("*.jpg")) + sorted(SRC.rglob("*.jpeg"))
    if args.limit:
        photos = photos[:args.limit]
    log.info("photos=%d angles=%d", len(photos), len(ANGLES))

    rows, times = [], []
    for i, p in enumerate(photos):
        with Image.open(p) as pil:
            rgb = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
        s = WORK_SIDE / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))

        t0 = time.time()
        base = estimate_tilt_raw(rgb)
        times.append(time.time() - t0)
        row = {"photo": p.stem, "base_angle": base.angle_deg,
               "base_unc": base.roll_uncertainty_deg, "variants": {}}
        for d in ANGLES:
            r = estimate_tilt_raw(rotated_crop(rgb, d))
            row["variants"][str(d)] = {"angle": r.angle_deg, "unc": r.roll_uncertainty_deg}
        rows.append(row)
        if (i + 1) % 10 == 0:
            log.info("%d/%d  (%.1fs/img)", i + 1, len(photos), np.mean(times))

    metrics = {"n_photos": len(rows), "source": str(SRC), "angles": ANGLES,
               "named_fp": NAMED_FP, "perf_s_per_img": float(np.mean(times)), "gates": {}}
    for gate in UNC_GATES:
        errs, big_ok, big_n = [], 0, 0
        cov_num = 0
        for row in rows:
            if row["base_unc"] > gate:
                continue
            for d in ANGLES:
                v = row["variants"][str(d)]
                if v["unc"] > gate:
                    continue
                cov_num += 1
                err = (v["angle"] - row["base_angle"]) - d
                errs.append(abs(err))
                if abs(d) >= 4.0:
                    big_n += 1
                    big_ok += abs(err) <= 2.0
        false_pos = [r["photo"] for r in rows
                     if r["base_unc"] <= gate and abs(r["base_angle"]) > 3.0]
        named_flagged = {n: any(r["photo"] == n and r["base_unc"] <= gate
                                and abs(r["base_angle"]) > 3.0 for r in rows)
                         for n in NAMED_FP}
        # G4: separate injected tilt (|d|>=4 variants) from upright originals by |angle|
        scores, labels = [], []
        for row in rows:
            scores.append(abs(row["base_angle"])); labels.append(0)
            for d in ANGLES:
                if abs(d) >= 4.0:
                    scores.append(abs(row["variants"][str(d)]["angle"])); labels.append(1)
        auc = roc_auc(np.array(scores), np.array(labels))
        metrics["gates"][str(gate)] = {
            "coverage": cov_num / (len(rows) * len(ANGLES)),
            "base_images_confident": sum(r["base_unc"] <= gate for r in rows) / len(rows),
            "mae_deg": float(np.mean(errs)) if errs else None,
            "recall_ge4deg_within2": (big_ok / big_n) if big_n else None,
            "originals_flagged_gt3deg": len(false_pos) / len(rows),
            "named_fp_flagged": named_flagged,
            "roc_auc_tilt_vs_upright": auc,
            "flagged_names": false_pos[:15],
        }

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump({"rows": rows, **metrics}, f, indent=2)
    for gate, g in metrics["gates"].items():
        log.info("unc<=%s: cov=%.2f baseConf=%.2f MAE=%s recall4=%s origFlag=%.3f "
                 "named=%s AUC=%.3f", gate, g["coverage"], g["base_images_confident"],
                 f"{g['mae_deg']:.2f}" if g["mae_deg"] else "-",
                 f"{g['recall_ge4deg_within2']:.2f}" if g["recall_ge4deg_within2"] else "-",
                 g["originals_flagged_gt3deg"], g["named_fp_flagged"], g["roc_auc_tilt_vs_upright"])
    log.info("perf: %.2f s/img", metrics["perf_s_per_img"])

    # samples: named-FP originals + one recovered rotation, estimate burned in
    sample_stems = NAMED_FP + [rows[0]["photo"]]
    for stem in sample_stems:
        row = next((r for r in rows if r["photo"] == stem), None)
        if row is None:
            continue
        p = next(SRC.rglob(stem + ".jp*"))
        with Image.open(p) as pil:
            rgb = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
        s = 800 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))
        est = row["base_angle"]; unc = row["base_unc"]
        out = cv2.putText(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                          f"roll {est:+.1f} +-{unc:.1f} deg", (12, 34),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(REPORT_DIR / "samples" / f"{stem}_orig.jpg"), out)

    log.info("done -> %s", REPORT_DIR)


if __name__ == "__main__":
    main()
