"""spec-096 final refit — re-evaluate every candidate on ADJUDICATED labels.

Inputs:  D:\\occlusion_dataset\\{clip_embeddings,features_log}.npz,
         corrections.csv (82 adjudications), classical_scores.csv,
         haiku_labels.csv, cnn_scores.csv.
Outputs: results_final.json, model_scores.csv (refreshed OOF columns),
         resnet_features.npz (D1 features, persisted this time).

Label policy: corrections override the manifest; ``occluded*`` -> 1,
``clean`` -> 0, ``foreground_object`` -> excluded from eval entirely.
"""

from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("refit")
ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from sim_bench.occlusion_bench.eval import cv_evaluate, cv_oof_scores, score_evaluate


def corrected_labels(ids: np.ndarray, y_orig: np.ndarray):
    """Apply corrections.csv; returns (y_corrected, keep_mask, n_flips)."""
    with open(os.path.join(ROOT, "corrections.csv"), newline="", encoding="utf-8") as f:
        corr = {r["id"]: r["decision"] for r in csv.DictReader(f)}  # last write wins
    y = y_orig.copy().astype(int)
    keep = np.ones(len(ids), dtype=bool)
    flips = 0
    for i, rid in enumerate(ids):
        d = corr.get(str(rid))
        if d is None:
            continue
        if d == "foreground_object":
            keep[i] = False
        else:
            new = 1 if d.startswith("occluded") else 0
            flips += int(new != y[i])
            y[i] = new
    return y, keep, flips


def _csv_scores(name: str, col: str) -> dict:
    p = os.path.join(ROOT, name)
    if not os.path.isfile(p):
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        out = {}
        for r in csv.DictReader(f):
            try:
                out[r["id"]] = float(r[col])
            except (TypeError, ValueError, KeyError):
                pass
        return out


def main() -> None:
    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    y, keep, flips = corrected_labels(ids, d["y"].astype(int))
    g = d["groups"]
    logger.info("labels: %d images, %d flips, %d excluded(fg), %d positives",
                len(ids), flips, int((~keep).sum()), int(y[keep].sum()))
    ids_k, y_k, g_k = ids[keep], y[keep], g[keep]
    results: dict = {"n_images": int(keep.sum()), "n_flips": flips,
                     "n_pos": int(y_k.sum()),
                     "random_baseline": float(y_k.mean())}

    # ---- Track B: CLIP probe variants -------------------------------------
    eg, et = d["emb_global"][keep], d["emb_tiles"][keep]
    variants = {"global": eg, "tile_max": et.max(axis=1), "tile_mean": et.mean(axis=1),
                "global+max": np.concatenate([eg, et.max(axis=1)], axis=1)}
    for name, X in variants.items():
        r = cv_evaluate(X, y_k, g_k, C=0.01)
        results[f"B_{name}"] = {k: v for k, v in r.items() if k != "folds"}
        logger.info("B/%s scene=%.3f [%.3f,%.3f] image=%.3f", name,
                    r["scene_pr_auc_mean"], r["scene_pr_auc_min"],
                    r["scene_pr_auc_max"], r["image_pr_auc_mean"])
    b_oof = cv_oof_scores(variants["global+max"], y_k, g_k, C=0.01)

    # ---- Track F: LoG stats ------------------------------------------------
    f = np.load(os.path.join(ROOT, "features_log.npz"), allow_pickle=True)
    # 'names' = the 78 FEATURE names; rows follow manifest order like the clip npz.
    # Alignment proof: identical length + identical original label vector + groups.
    assert f["X"].shape[0] == len(ids) and (f["y"].astype(int) == d["y"].astype(int)).all() \
        and (f["groups"] == d["groups"]).all(), "features_log.npz not aligned"
    Xf = f["X"][keep]
    r = cv_evaluate(Xf, y_k, g_k, C=0.1)
    results["F_log"] = {k: v for k, v in r.items() if k != "folds"}
    logger.info("F scene=%.3f image=%.3f", r["scene_pr_auc_mean"], r["image_pr_auc_mean"])
    f_oof = cv_oof_scores(Xf, y_k, g_k, C=0.1)

    # ---- Track D1: frozen ResNet18 probe (features persisted this run) ----
    feat_p = os.path.join(ROOT, "resnet_features.npz")
    if os.path.isfile(feat_p):
        rn = np.load(feat_p, allow_pickle=True)
        rn_ids, Xr = np.array([str(x) for x in rn["ids"]]), rn["X"]
    else:
        import torch
        import torchvision.models as M
        import torchvision.transforms as T
        from PIL import Image
        torch.set_num_threads(4)
        net = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval()
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        rows_x, rn_ids = [], []
        with torch.no_grad():
            for i, (rid, yo) in enumerate(zip(ids, d["y"].astype(int))):
                sub = "positives" if yo == 1 else "negatives"  # folder = ORIGINAL label
                try:
                    img = Image.open(os.path.join(ROOT, sub, rid)).convert("RGB")
                except Exception as e:
                    logger.warning("D1 skip %s: %s", rid, e)
                    rows_x.append(np.zeros(512, np.float32))
                    rn_ids.append(rid)
                    continue
                rows_x.append(net(tf(img).unsqueeze(0)).squeeze(0).numpy())
                rn_ids.append(rid)
                if (i + 1) % 100 == 0:
                    logger.info("D1 features %d/%d", i + 1, len(ids))
        Xr, rn_ids = np.array(rows_x), np.array(rn_ids)
        np.savez_compressed(feat_p, ids=rn_ids, X=Xr)
        logger.info("saved %s", feat_p)
    assert (rn_ids == ids).all(), "resnet_features.npz not aligned"
    r = cv_evaluate(Xr[keep], y_k, g_k, C=0.01)
    results["D1_frozen_probe"] = {k: v for k, v in r.items() if k != "folds"}
    logger.info("D1 scene=%.3f image=%.3f", r["scene_pr_auc_mean"], r["image_pr_auc_mean"])

    # ---- Zero-shot / fixed-score candidates on corrected labels -----------
    def eval_fixed(tag: str, scores_by_id: dict):
        m = np.array([scores_by_id.get(rid, np.nan) for rid in ids_k])
        ok = ~np.isnan(m)
        if ok.sum() < 10:
            logger.info("%s: too few scores, skipped", tag)
            return
        results[tag] = score_evaluate(y_k[ok], g_k[ok], m[ok])
        results[tag]["n_scored"] = int(ok.sum())
        logger.info("%s scene=%.3f image=%.3f (n=%d)", tag,
                    results[tag]["pr_auc_scene"], results[tag]["pr_auc_image"], ok.sum())

    eval_fixed("E_classical", _csv_scores("classical_scores.csv", "score"))
    eval_fixed("A_haiku", {rid: v for rid, v in
                           _csv_scores("haiku_labels.csv", "haiku_occluded").items()})
    eval_fixed("D2_cnn_all_real", _csv_scores("cnn_scores.csv", "score"))  # round-1 model, NOT retrained

    with open(os.path.join(ROOT, "results_final.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- refresh model_scores.csv (OOF under corrected labels) ------------
    with open(os.path.join(ROOT, "model_scores.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "B_clip_oof", "F_log_oof"])
        for rid, b, fl in zip(ids_k, b_oof, f_oof):
            w.writerow([rid, f"{b:.4f}", f"{fl:.4f}"])
    logger.info("wrote results_final.json + model_scores.csv")


if __name__ == "__main__":
    main()
