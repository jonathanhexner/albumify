"""
Defect-scoring benchmark #4: ROTATION (wrong orientation).

Dataset: generated -- 122 real album photos (D:\\Budapest2025_Google), EXIF-normalized,
then pixel-rotated to 90/180/270 deg. EXIF is stripped by re-encoding, so EXIF-based
correction (all our pipeline does) cannot help: this isolates CONTENT-based detection.

Protocol: for each photo, score all 4 variants with each method's "upright score";
predict orientation = argmax. 4-way chance = 25%.
  - ours      : RuleBasedQuality.assess_image (pipeline overall) -- expected ~rotation-invariant
  - classical : brightness-centroid "sky prior" (brightest side should be up)
  - sota      : CLIP ViT-B/32 zero-shot (upright vs rotated text prompts)

Outputs reports/2026-07-10_defect_rotation/{data/*, samples/*}
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.rule_based import RuleBasedQuality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("defect_rotation")

SRC = Path(r"D:\Budapest2025_Google")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "2026-07-10_defect_rotation"
WORK = REPORT_DIR / "data" / "variants"
ROTS = [0, 90, 180, 270]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

CV_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def classical_upright(img_bgr: np.ndarray) -> float:
    """Sky prior: top third should be brighter than bottom third."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    h = gray.shape[0]
    return float(gray[: h // 3].mean() - gray[-(h // 3):].mean())


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "samples").mkdir(exist_ok=True)

    import torch
    import clip
    from PIL import Image, ImageOps
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    prompts = ["a photo with correct orientation, right side up",
               "a photo rotated sideways, tilted 90 degrees",
               "an upside down photo"]
    with torch.no_grad():
        text_emb = model.encode_text(clip.tokenize(prompts).to(device))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    rb = RuleBasedQuality()

    photos = sorted(SRC.rglob("*.jpg")) + sorted(SRC.rglob("*.jpeg"))
    if LIMIT:
        photos = photos[:LIMIT]
    log.info("photos: %d x %d rotations", len(photos), len(ROTS))

    rows, t0 = [], time.time()
    for i, photo in enumerate(photos):
        # EXIF-normalize once, downscale to 1024 for speed, then generate pixel rotations
        with Image.open(photo) as pil:
            pil = ImageOps.exif_transpose(pil).convert("RGB")
        base = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        s = 1024.0 / max(base.shape[:2])
        if s < 1:
            base = cv2.resize(base, (int(base.shape[1] * s), int(base.shape[0] * s)))
        for rot in ROTS:
            img = base if rot == 0 else cv2.rotate(base, CV_ROT[rot])
            vpath = WORK / f"{photo.stem}_r{rot}.jpg"
            cv2.imwrite(str(vpath), img)  # re-encode strips EXIF
            with torch.no_grad():
                t = preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0)
                e = model.encode_image(t)
                e /= e.norm(dim=-1, keepdim=True)
                sims = (e @ text_emb.T).squeeze(0).tolist()
            rows.append({
                "photo": photo.stem, "rot": rot,
                "ours_overall": rb.assess_image(str(vpath)),
                "classical_sky": classical_upright(img),
                "sota_clip": float(sims[0] - max(sims[1], sims[2])),
            })
        if (i + 1) % 20 == 0:
            log.info("%d/%d (%.1fs)", i + 1, len(photos), time.time() - t0)

    with open(REPORT_DIR / "data" / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from scipy.stats import mannwhitneyu
    metrics = {"n_photos": len(photos), "source": str(SRC),
               "note": "EXIF stripped by re-encoding; content-based detection only. 4-way chance = 25%."}
    by_photo = {}
    for r in rows:
        by_photo.setdefault(r["photo"], {})[r["rot"]] = r
    for m in ("ours_overall", "classical_sky", "sota_clip"):
        correct = sum(1 for v in by_photo.values() if max(v.values(), key=lambda r: r[m])["rot"] == 0)
        up = np.array([r[m] for r in rows if r["rot"] == 0])
        rot = np.array([r[m] for r in rows if r["rot"] != 0])
        metrics[m] = {
            "argmax_acc_4way": correct / len(by_photo),
            "auc_upright_vs_rotated": float(
                mannwhitneyu(up, rot, alternative="greater").statistic / (len(up) * len(rot))),
            "mean_upright": float(up.mean()), "mean_rotated": float(rot.mean()),
        }

    # samples: one photo's 4 variants (downscaled)
    demo = photos[len(photos) // 2].stem
    for rot in ROTS:
        img = cv2.imread(str(WORK / f"{demo}_r{rot}.jpg"))
        s = 320.0 / max(img.shape[:2])
        cv2.imwrite(str(REPORT_DIR / "samples" / f"{demo}_r{rot}.jpg"),
                    cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s))))
    metrics["sample_photo"] = demo

    with open(REPORT_DIR / "data" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("done in %.1fs -> %s", time.time() - t0, REPORT_DIR)


if __name__ == "__main__":
    main()
