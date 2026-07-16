"""
spec-099 T0.2/T0.3: tilt-estimator benchmark on real album photos.

Protocol: EXIF-normalize each Budapest photo, downscale, then inject known roll
angles (+-2,4,6,8,10 deg) by rotating and taking the central valid crop.
Recovery is measured RELATIVELY: delta_hat = angle(rotated) - angle(original),
so a photo's own (unknown) tilt cancels out.

Gates: A1 MAE < 1.5 deg + >=80% of >=4 deg tilts within +-2 deg (confident pairs);
A2 <=5% of originals confidently flagged >3 deg; A5 < 100 ms/estimate.
Also sweeps the confidence gate. Outputs reports/2026-07-12_tilt_benchmark/.
"""

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.tilt import estimate_tilt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tilt_bench")

SRC = Path(r"D:\Budapest2025_Google")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "2026-07-12_tilt_benchmark"
ANGLES = [-10.0, -8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0, 10.0]
CONF_GATES = [0.2, 0.3, 0.5]
WORK_SIDE = 1400  # rotate at this size, estimator downscales to 1024 internally


def rotated_crop(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate content clockwise by angle_deg; central 72% crop excludes corners."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
    rot = cv2.warpAffine(gray, m, (w, h))
    mh, mw = int(0.14 * h), int(0.14 * w)
    return rot[mh:h - mh, mw:w - mw]


def main():
    (REPORT_DIR / "samples").mkdir(parents=True, exist_ok=True)
    photos = sorted(SRC.rglob("*.jpg")) + sorted(SRC.rglob("*.jpeg"))
    log.info("photos=%d angles=%d", len(photos), len(ANGLES))

    rows, times = [], []
    for i, p in enumerate(photos):
        with Image.open(p) as pil:
            pil = ImageOps.exif_transpose(pil).convert("L")
        gray = np.array(pil)
        s = WORK_SIDE / max(gray.shape)
        if s < 1:
            gray = cv2.resize(gray, (int(gray.shape[1] * s), int(gray.shape[0] * s)))

        t0 = time.time()
        base = estimate_tilt(gray)
        times.append(time.time() - t0)
        row = {"photo": p.stem, "base_angle": base.angle_deg, "base_conf": base.confidence,
               "base_lines": base.n_lines, "variants": {}}
        for d in ANGLES:
            r = estimate_tilt(rotated_crop(gray, d))
            row["variants"][str(d)] = {"angle": r.angle_deg, "conf": r.confidence}
        rows.append(row)
        if (i + 1) % 25 == 0:
            log.info("%d/%d", i + 1, len(photos))

    metrics = {"n_photos": len(rows), "source": str(SRC), "angles": ANGLES,
               "perf_ms_per_estimate": float(np.mean(times) * 1000.0), "gates": {}}
    for gate in CONF_GATES:
        errs, big_ok, big_n = [], 0, 0
        for row in rows:
            if row["base_conf"] < gate:
                continue
            for d in ANGLES:
                v = row["variants"][str(d)]
                if v["conf"] < gate:
                    continue
                err = (v["angle"] - row["base_angle"]) - d
                errs.append(abs(err))
                if abs(d) >= 4.0:
                    big_n += 1
                    big_ok += abs(err) <= 2.0
        false_pos = [r for r in rows if r["base_conf"] >= gate and abs(r["base_angle"]) > 3.0]
        metrics["gates"][str(gate)] = {
            "n_confident_pairs": len(errs),
            "coverage": len(errs) / (len(rows) * len(ANGLES)),
            # real-world abstention: variants are 72% crops (lines shortened),
            # base images are what production actually sees.
            "base_images_confident": sum(r["base_conf"] >= gate for r in rows) / len(rows),
            "mae_deg": float(np.mean(errs)) if errs else None,
            "recall_ge4deg_within2": (big_ok / big_n) if big_n else None,
            "originals_flagged_gt3deg": len(false_pos) / len(rows),
            "flagged_names": [r["photo"] for r in false_pos][:15],
        }

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    for gate, g in metrics["gates"].items():
        log.info("conf>=%s: pairs=%d cov=%.2f MAE=%.2f recall4=%.2f origFlag=%.3f",
                 gate, g["n_confident_pairs"], g["coverage"], g["mae_deg"] or -1,
                 g["recall_ge4deg_within2"] or -1, g["originals_flagged_gt3deg"])
    log.info("perf: %.0f ms/estimate", metrics["perf_ms_per_estimate"])

    # samples: 2 photos, original + one rotated variant with estimates burned in
    for row in rows[:2]:
        p = next(SRC.rglob(row["photo"] + ".jp*"))
        with Image.open(p) as pil:
            g = np.array(ImageOps.exif_transpose(pil).convert("L"))
        s = 800 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)))
        for tag, img, est in [("orig", g, row["base_angle"]),
                              ("rot6", rotated_crop(g, 6.0), row["variants"]["6.0"]["angle"])]:
            out = cv2.putText(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                              f"est {est:+.1f} deg", (12, 34),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite(str(REPORT_DIR / "samples" / f"{row['photo']}_{tag}.jpg"), out)

    log.info("done -> %s", REPORT_DIR)


if __name__ == "__main__":
    main()
