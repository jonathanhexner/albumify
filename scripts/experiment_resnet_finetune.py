"""Fine-tune a ResNet18 CNN (pretrained, all layers) with on-the-fly augmentation
for occlusion detection and compare against the CLIP+LR probe (2026-07-10).

Protocol matches experiment_augmented_retrain.py exactly so numbers are
comparable: 856 originals (corrected labels), +140 RealBlur train negatives in
every train fold, 5-fold StratifiedGroupKFold seed 0, OOF metrics on originals,
scene-level = max over group. CPU training: ResNet18, 224px, weighted BCE
(class x 1/group-size), Adam (backbone 1e-4 / head 1e-3), EPOCHS per fold.

Speed: all images pre-decoded once into a 256px JPEG cache
(D:/occlusion_dataset/_cache256). Augs: hflip, +-12deg rotation, random-resized
crop (85-100%), color jitter, Gaussian noise — the user-requested menu.

Outputs: reports/2026-07-10_resnet_finetune/resnet_results.json + resnet_oof.npz
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("resnet")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
CACHE = os.path.join(ROOT, "_cache256")
RB_SAMPLE = os.path.join(ROOT, "realblur", "j_sample", "blur")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_resnet_finetune")
EPOCHS = 6
BATCH = 32
SEED = 0
GATE = 0.8
TRAIN_SCENE_FRAC = 0.6


def corrected_labels(ids, y_orig):
    with open(os.path.join(ROOT, "corrections.csv"), newline="", encoding="utf-8") as f:
        corr = {r["id"]: r["decision"] for r in csv.DictReader(f)}
    y = y_orig.copy().astype(int)
    for i, rid in enumerate(ids):
        d = corr.get(str(rid))
        if d and d != "foreground_object":
            y[i] = 1 if d.startswith("occluded") else 0
    return y


def build_cache(items):
    """items: list of (key, src_path). Decode once, save 256px jpg."""
    import cv2
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    os.makedirs(CACHE, exist_ok=True)
    n = 0
    for key, src in items:
        dst = os.path.join(CACHE, key + ".jpg")
        if os.path.isfile(dst):
            continue
        try:
            with Image.open(src) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:
            logger.warning("cache skip %s (%s)", key, e)
            continue
        img = np.array(img)[:, :, ::-1]
        h, w = img.shape[:2]
        s = 256.0 / min(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)))
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
        if n % 200 == 0:
            logger.info("cached %d", n)
    logger.info("cache ready (%d new)", n)


class OcclusionDs:
    def __init__(self, keys, labels, weights, train):
        import torchvision.transforms as T
        self.keys, self.labels, self.weights, self.train = keys, labels, weights, train
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            self.tf = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(12),
                T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.11)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
                T.ToTensor(),
                T.Lambda(lambda t: (t + 0.03 * (t.new_empty(t.shape).normal_())
                                    ).clamp(0, 1)),
                norm])
        else:
            self.tf = T.Compose([T.CenterCrop(224), T.ToTensor(), norm])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(os.path.join(CACHE, self.keys[i] + ".jpg")).convert("RGB")
        if not self.train:  # deterministic eval: resize handled by cache (short side 256)
            pass
        return self.tf(img), self.labels[i], self.weights[i]


def make_weights(y, groups):
    gsize = {}
    for g in groups:
        gsize[g] = gsize.get(g, 0) + 1
    n_pos, n_neg = int(sum(y)), int(len(y) - sum(y))
    cls_w = {1: len(y) / (2.0 * n_pos), 0: len(y) / (2.0 * n_neg)}
    return np.array([cls_w[int(yi)] / gsize[g] for yi, g in zip(y, groups)],
                    dtype=np.float32)


def train_fold(keys_tr, y_tr, w_tr, keys_te, fold):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18, ResNet18_Weights
    torch.manual_seed(SEED + fold)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    opt = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")],
         "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3}])
    ds = OcclusionDs(keys_tr, y_tr.astype(np.float32), w_tr, train=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    model.train()
    for ep in range(EPOCHS):
        t0, tot, nb = time.time(), 0.0, 0
        for xb, yb, wb in dl:
            opt.zero_grad()
            logit = model(xb).squeeze(1)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logit, yb, reduction="none") * wb).mean()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        logger.info("fold %d epoch %d/%d loss %.4f (%.0fs)",
                    fold, ep + 1, EPOCHS, tot / max(nb, 1), time.time() - t0)
    model.eval()
    ds_te = OcclusionDs(keys_te, np.zeros(len(keys_te), np.float32),
                        np.ones(len(keys_te), np.float32), train=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for xb, _, _ in dl_te:
            probs.extend(torch.sigmoid(model(xb).squeeze(1)).tolist())
    return np.array(probs)


def main():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    os.makedirs(OUT, exist_ok=True)

    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    y0 = corrected_labels(ids, d["y"].astype(int))
    g0 = np.array([str(x) for x in d["groups"]])
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        sub = {r["id"]: ("positives" if r["label"] == "1" else "negatives")
               for r in csv.DictReader(f)}

    rb_files = sorted(os.listdir(RB_SAMPLE))
    rb_scenes = sorted({n.split("__")[0] for n in rb_files})
    train_scenes = set(rb_scenes[:int(len(rb_scenes) * TRAIN_SCENE_FRAC)])
    seen, rb_sel = set(), []
    for n in rb_files:
        s = n.split("__")[0]
        if s in train_scenes and s not in seen:
            seen.add(s)
            rb_sel.append(n)

    build_cache([(rid, os.path.join(ROOT, sub[rid], rid)) for rid in ids]
                + [(n, os.path.join(RB_SAMPLE, n)) for n in rb_sel])
    have = {k for k in ids if os.path.isfile(os.path.join(CACHE, k + ".jpg"))}
    keep = np.array([rid in have for rid in ids])
    ids, y0, g0 = ids[keep], y0[keep], g0[keep]
    rb_sel = [n for n in rb_sel if os.path.isfile(os.path.join(CACHE, n + ".jpg"))]
    logger.info("training on %d originals (%d pos) + %d realblur",
                len(ids), int(y0.sum()), len(rb_sel))

    oof = np.full(len(y0), np.nan)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rb_g = np.array(["rb_" + n.split("__")[0] for n in rb_sel])
    for fold, (tr, te) in enumerate(skf.split(ids, y0, g0)):
        keys_tr = list(ids[tr]) + rb_sel
        y_tr = np.concatenate([y0[tr], np.zeros(len(rb_sel), int)])
        g_tr = np.concatenate([g0[tr], rb_g])
        w_tr = make_weights(y_tr, g_tr)
        oof[te] = train_fold(keys_tr, y_tr, w_tr, list(ids[te]), fold)
        logger.info("fold %d done (%d test imgs)", fold, len(te))

    sp, sy = {}, {}
    for p, yy, g in zip(oof, y0, g0):
        sp[g] = max(sp.get(g, 0.0), float(p))
        sy[g] = max(sy.get(g, 0), int(yy))
    ks = sorted(sp)
    svp, svy = [sp[k] for k in ks], [sy[k] for k in ks]
    res = {
        "model": "resnet18 fine-tuned (all layers), aug on-the-fly", "epochs": EPOCHS,
        "n": len(y0), "n_pos": int(y0.sum()), "gate": GATE, "seed": SEED,
        "img_roc": round(float(roc_auc_score(y0, oof)), 4),
        "img_pr": round(float(average_precision_score(y0, oof)), 4),
        "scene_roc": round(float(roc_auc_score(svy, svp)), 4),
        "scene_pr": round(float(average_precision_score(svy, svp)), 4),
        "fp_over_gate": int(((oof >= GATE) & (y0 == 0)).sum()),
        "fn_under_gate": int(((oof < GATE) & (y0 == 1)).sum()),
        "fp_over_half": int(((oof >= 0.5) & (y0 == 0)).sum()),
    }
    np.savez_compressed(os.path.join(OUT, "resnet_oof.npz"),
                        ids=ids, y=y0, groups=g0, oof=oof)
    with open(os.path.join(OUT, "resnet_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info("RESNET: %s", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
