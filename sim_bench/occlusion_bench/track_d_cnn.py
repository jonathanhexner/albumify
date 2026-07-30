"""Track D — ResNet18 candidates (spec-096).

Two variants, honesty rules baked in:
  D1  frozen ResNet18 features -> logistic probe (REAL data only, grouped CV)
  D2  fine-tuned ResNet18 (layer4+fc) trained on TRAIN split:
        real positives + SYNTHETIC positives (composited onto train negatives,
        inheriting the base group) + real negatives.
      Evaluated ONLY on the untouched REAL test split. Synthetic never scores.
"""

from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)
ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")

try:  # albums contain .heic — register before any PIL open
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # pragma: no cover
    pass


def _device_setup():
    import torch
    torch.set_num_threads(4)
    return torch


def frozen_features(rows) -> dict:
    """D1: 512-d avgpool features for all images -> grouped-CV probe."""
    torch = _device_setup()
    import torchvision.models as M
    import torchvision.transforms as T
    from PIL import Image
    from sim_bench.occlusion_bench.eval import cv_evaluate

    net = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    X, y, g = [], [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            sub = "positives" if r["label"] == "1" else "negatives"
            try:
                img = Image.open(os.path.join(ROOT, sub, r["id"])).convert("RGB")
            except Exception as e:
                logger.warning("skip %s: %s", r["id"], e)
                continue
            X.append(net(tf(img).unsqueeze(0)).squeeze(0).numpy())
            y.append(int(r["label"])); g.append(r["group_id"])
            if (i + 1) % 100 == 0:
                logger.info("D1 features %d/%d", i + 1, len(rows))
    res = cv_evaluate(np.array(X), np.array(y), np.array(g), C=0.01)
    return {k: v for k, v in res.items() if k != "folds"}


def gen_synth(rows, per_neg: int = 1) -> list:
    """Composite occluders onto TRAIN negatives -> synthetic positives list."""
    from sim_bench.occlusion_bench.synthesize import synth_occlusion
    out_dir = os.path.join(ROOT, "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, "synth_manifest.csv")
    synth = []
    if os.path.isfile(man_path):
        with open(man_path, newline="", encoding="utf-8") as f:
            synth = list(csv.DictReader(f))
        if synth:
            logger.info("synth: reusing %d existing", len(synth))
            return synth
    train_negs = [r for r in rows if r["split"] == "train" and r["label"] == "0"]
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "base_id", "group_id", "level", "params"])
        for i, r in enumerate(train_negs):
            for seed in range(per_neg):
                sid = f"synth{seed}__{r['id']}".replace(".heic", ".jpg")
                dst = os.path.join(out_dir, sid)
                try:
                    lvl, params = synth_occlusion(
                        os.path.join(ROOT, "negatives", r["id"]), dst, seed=seed)
                except Exception as e:
                    logger.warning("synth failed for %s: %s", r["id"], e)
                    continue
                w.writerow([sid, r["id"], r["group_id"], lvl, json.dumps(params)])
                synth.append({"id": sid, "base_id": r["id"], "group_id": r["group_id"],
                              "level": str(lvl)})
            if (i + 1) % 100 == 0:
                logger.info("synth %d/%d", i + 1, len(train_negs))
    return synth


def finetune_and_eval(rows, synth) -> dict:
    """D2: fine-tune layer4+fc on train (real+synth), score ALL real, eval on test."""
    torch = _device_setup()
    import torchvision.models as M
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from sim_bench.occlusion_bench.eval import score_evaluate

    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                          T.RandomHorizontalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_items = []  # (path, label)
    for r in rows:
        if r["split"] != "train":
            continue
        sub = "positives" if r["label"] == "1" else "negatives"
        train_items.append((os.path.join(ROOT, sub, r["id"]), int(r["label"])))
    for s in synth:
        train_items.append((os.path.join(ROOT, "synthetic", s["id"]), 1))
    labels = np.array([l for _, l in train_items])
    logger.info("D2 train set: %d (pos=%d incl %d synth)", len(train_items),
                labels.sum(), len(synth))

    class DS(Dataset):
        def __len__(self):
            return len(train_items)
        def __getitem__(self, i):
            p, l = train_items[i]
            img = Image.open(p).convert("RGB")
            return train_tf(img), float(l)

    wts = np.where(labels == 1, 0.5 / max(labels.sum(), 1),
                   0.5 / max((labels == 0).sum(), 1))
    dl = DataLoader(DS(), batch_size=24,
                    sampler=WeightedRandomSampler(wts.tolist(), len(train_items)),
                    num_workers=0)

    net = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.layer4.parameters():
        p.requires_grad = True
    net.fc = torch.nn.Linear(512, 1)
    opt = torch.optim.AdamW([{"params": net.layer4.parameters(), "lr": 1e-4},
                             {"params": net.fc.parameters(), "lr": 1e-3}])
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    for epoch in range(3):
        tot = 0.0
        for bi, (xb, yb) in enumerate(dl):
            opt.zero_grad()
            out = net(xb).squeeze(1)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += float(loss)
            if (bi + 1) % 10 == 0:
                logger.info("D2 epoch %d batch %d/%d loss=%.3f",
                            epoch + 1, bi + 1, len(dl), tot / (bi + 1))
        logger.info("D2 epoch %d done, mean loss %.3f", epoch + 1, tot / len(dl))

    # score every REAL image
    net.eval()
    scores = {}
    with torch.no_grad():
        for i, r in enumerate(rows):
            sub = "positives" if r["label"] == "1" else "negatives"
            try:
                img = Image.open(os.path.join(ROOT, sub, r["id"])).convert("RGB")
                scores[r["id"]] = float(torch.sigmoid(net(eval_tf(img).unsqueeze(0))).item())
            except Exception as e:
                logger.warning("score skip %s: %s", r["id"], e)
            if (i + 1) % 100 == 0:
                logger.info("D2 scoring %d/%d", i + 1, len(rows))
    with open(os.path.join(ROOT, "cnn_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for k, v in scores.items():
            w.writerow([k, f"{v:.4f}"])

    def _eval(split):
        sel = [r for r in rows if r["split"] == split and r["id"] in scores]
        return score_evaluate(np.array([int(r["label"]) for r in sel]),
                              np.array([r["group_id"] for r in sel]),
                              np.array([scores[r["id"]] for r in sel]))
    return {"test_real": _eval("test"), "train_real_IN_SAMPLE": _eval("train")}


def main():
    from sim_bench.occlusion_bench.dataset import load_manifest
    rows = load_manifest(ROOT)
    results = {}
    results["D1_frozen_probe_cv"] = frozen_features(rows)
    logger.info("D1: %s", results["D1_frozen_probe_cv"])
    synth = gen_synth(rows)
    results["D2_synth_cnn"] = finetune_and_eval(rows, synth)
    logger.info("D2: %s", results["D2_synth_cnn"])
    with open(os.path.join(ROOT, "results_track_d.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
