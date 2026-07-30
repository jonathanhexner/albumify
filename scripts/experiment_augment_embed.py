"""Generate augmented variants of every training image and embed them (2026-07-10).

K=3 variants per original: hflip / rotation +-12deg (reflect fill, no black
corners) / random crop 85-97% area / brightness-contrast-gamma jitter /
Gaussian noise. BOTH classes are augmented with the same distribution so the
head cannot key on augmentation artifacts. Positive crops are kept >=85% area
to limit cutting corner occluders out of frame (accepted risk, noted in report).

Sources: 856 manifest originals (corrected labels) + 140 RealBlur train
negatives (1/scene, same selection rule as experiment_realblur_retrain).
Output: D:/occlusion_dataset/aug_embeddings.npz (names, src, y, groups,
emb_global, emb_tiles) + sample gallery images for the report.
"""

from __future__ import annotations

import csv
import logging
import os
import random

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("augment")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
RB_SAMPLE = os.path.join(ROOT, "realblur", "j_sample", "blur")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_augmented_retrain")
K = 3
SEED = 42
TRAIN_SCENE_FRAC = 0.6  # must match experiment_realblur_retrain


def read_bgr(path):
    img = cv2.imread(path)
    if img is None:
        try:
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = cv2.cvtColor(np.array(ImageOps.exif_transpose(
                Image.open(path)).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("unreadable %s (%s)", path, e)
            return None
    h, w = img.shape[:2]
    s = 1600.0 / max(h, w)  # downscale: CLIP sees 224px anyway; speeds aug+encode
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def augment(img, rng):
    ops = []
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        ops.append("hflip")
    if rng.random() < 0.7:
        ang = rng.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        ops.append(f"rot{ang:.0f}")
    if rng.random() < 0.7:
        area = rng.uniform(0.85, 0.97)
        s = area ** 0.5
        cw, ch = int(w * s), int(h * s)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img[y0:y0 + ch, x0:x0 + cw]
        h, w = img.shape[:2]
        ops.append(f"crop{area:.2f}")
    if rng.random() < 0.8:
        alpha = rng.uniform(0.7, 1.3)                      # contrast
        beta = rng.uniform(-35, 35)                        # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        gamma = rng.uniform(0.7, 1.4)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                       dtype=np.uint8)
        img = cv2.LUT(img, lut)
        ops.append("light")
    if rng.random() < 0.5:
        sigma = rng.uniform(4, 10)
        img = np.clip(img.astype(np.float32)
                      + np.random.default_rng(rng.randint(0, 1 << 30))
                      .normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
        ops.append("noise")
    if not ops:  # guarantee at least one op
        img = cv2.flip(img, 1)
        ops.append("hflip")
    return img, "+".join(ops)


def corrected_labels(ids, y_orig):
    with open(os.path.join(ROOT, "corrections.csv"), newline="", encoding="utf-8") as f:
        corr = {r["id"]: r["decision"] for r in csv.DictReader(f)}
    y = y_orig.copy().astype(int)
    for i, rid in enumerate(ids):
        d = corr.get(str(rid))
        if d and d != "foreground_object":
            y[i] = 1 if d.startswith("occluded") else 0
    return y


def main():
    from sim_bench.occlusion_bench.scorer import OcclusionScorer
    rng = random.Random(SEED)
    os.makedirs(os.path.join(OUT, "imgs_aug"), exist_ok=True)
    tmp = os.path.join(ROOT, "_tmp_aug.jpg")

    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    y = corrected_labels(ids, d["y"].astype(int))
    grp = [str(x) for x in d["groups"]]
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        sub = {r["id"]: ("positives" if r["label"] == "1" else "negatives")
               for r in csv.DictReader(f)}

    work = []  # (src_id, path, label, group)
    for i, rid in enumerate(ids):
        work.append((rid, os.path.join(ROOT, sub[rid], rid), int(y[i]), grp[i]))

    rb = np.load(os.path.join(ROOT, "realblur", "realblur_embeddings.npz"),
                 allow_pickle=True)
    rb_names = [str(x) for x in rb["names"]]
    rb_scenes = sorted({n.split("__")[0] for n in rb_names})
    train_scenes = set(rb_scenes[:int(len(rb_scenes) * TRAIN_SCENE_FRAC)])
    seen = set()
    for n in rb_names:
        s = n.split("__")[0]
        if s in train_scenes and s not in seen:
            seen.add(s)
            work.append((n, os.path.join(RB_SAMPLE, n), 0, "rb_" + s))
    logger.info("augmenting %d originals x %d variants", len(work), K)

    scorer = OcclusionScorer()
    names, src, yy, gg, eg, et = [], [], [], [], [], []
    n_sample = 0
    for k, (rid, path, lab, g) in enumerate(work):
        img = read_bgr(path)
        if img is None:
            continue
        for v in range(K):
            aug, ops = augment(img.copy(), rng)
            cv2.imwrite(tmp, aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            pair = scorer.embed(tmp)
            if pair is None:
                continue
            names.append(f"{rid}#a{v}:{ops}")
            src.append(rid)
            yy.append(lab)
            gg.append(g)
            eg.append(pair[0])
            et.append(pair[1])
            # sample gallery: every ~120th aug, both classes
            if (len(names) % 120) == 1 and n_sample < 24:
                h, w = aug.shape[:2]
                s = 360 / max(h, w)
                small = cv2.resize(aug, (int(w * s), int(h * s))) if s < 1 else aug
                cv2.imwrite(os.path.join(
                    OUT, "imgs_aug",
                    f"aug_{'pos' if lab else 'neg'}_{len(names)}.jpg"),
                    small, [cv2.IMWRITE_JPEG_QUALITY, 82])
                n_sample += 1
        if (k + 1) % 100 == 0:
            logger.info("%d/%d originals done (%d aug embeddings)",
                        k + 1, len(work), len(names))
    if os.path.isfile(tmp):
        os.remove(tmp)
    np.savez_compressed(os.path.join(ROOT, "aug_embeddings.npz"),
                        names=np.array(names), src=np.array(src),
                        y=np.array(yy), groups=np.array(gg),
                        emb_global=np.array(eg), emb_tiles=np.array(et))
    logger.info("DONE: %d augmented embeddings (%d pos, %d neg)",
                len(names), int(np.sum(yy)), int(len(yy) - np.sum(yy)))


if __name__ == "__main__":
    main()
