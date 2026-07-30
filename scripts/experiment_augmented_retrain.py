"""Train + evaluate the occlusion probe on the grown dataset (2026-07-10):
856 originals (80 positives incl. batch-3) / +140 RealBlur negatives / +augs.

Variants:
  base     — 856 originals
  rb       — + 140 RealBlur train negatives
  rb_aug   — + augmentations of everything (train folds only)

Honest evaluation: 5-fold StratifiedGroupKFold OOF over the originals
(RB + augs pinned to train; augs of test-fold groups excluded), 3 seeds.
Outputs partD_results.json, roc_curves.png, FP/FN galleries (gate 0.8) under
reports/2026-07-10_augmented_retrain/. Winner candidate artifact saved to
D:/occlusion_dataset/clip_b32_gmax_v3aug.npz (NOT production).
"""

from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("partD")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_augmented_retrain")
CANDIDATE = os.path.join(ROOT, "clip_b32_gmax_v3aug.npz")
GATE = 0.8
SEEDS = [0, 1, 2]
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


def fit(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    gsize = {}
    for g in groups:
        gsize[g] = gsize.get(g, 0) + 1
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    cls_w = {1: len(y) / (2.0 * n_pos), 0: len(y) / (2.0 * n_neg)}
    w = np.array([cls_w[int(yi)] / gsize[g] for yi, g in zip(y, groups)])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(X), y, sample_weight=w)
    return sc, clf


def feats(eg, et):
    return np.concatenate([eg, et.max(axis=1)], axis=1)


def main():
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold
    os.makedirs(OUT, exist_ok=True)

    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    y0 = corrected_labels(ids, d["y"].astype(int))
    g0 = np.array([str(x) for x in d["groups"]])
    X0 = feats(d["emb_global"], d["emb_tiles"])
    logger.info("originals: %d imgs, %d pos, %d scenes",
                len(y0), int(y0.sum()), len(set(g0.tolist())))

    rb = np.load(os.path.join(ROOT, "realblur", "realblur_embeddings.npz"),
                 allow_pickle=True)
    rb_names = [str(x) for x in rb["names"]]
    rb_scenes = sorted({n.split("__")[0] for n in rb_names})
    train_scenes = set(rb_scenes[:int(len(rb_scenes) * TRAIN_SCENE_FRAC)])
    seen, tr_sel, ho_sel = set(), [], []
    for i, n in enumerate(rb_names):
        s = n.split("__")[0]
        if s in train_scenes:
            if s not in seen:
                seen.add(s)
                tr_sel.append(i)
        else:
            ho_sel.append(i)
    Xrb = feats(rb["emb_global"], rb["emb_tiles"])
    Xrb_tr, g_rb = Xrb[tr_sel], np.array(["rb_" + rb_names[i].split("__")[0] for i in tr_sel])
    Xrb_ho = Xrb[ho_sel]

    a = np.load(os.path.join(ROOT, "aug_embeddings.npz"), allow_pickle=True)
    Xa = feats(a["emb_global"], a["emb_tiles"])
    ya, ga = a["y"].astype(int), np.array([str(x) for x in a["groups"]])
    logger.info("augs: %d (%d pos)", len(ya), int(ya.sum()))

    variants = {
        "base":   dict(rb=False, aug=False),
        "rb":     dict(rb=True, aug=False),
        "rb_aug": dict(rb=True, aug=True),
    }
    res = {"gate": GATE, "n_orig": len(y0), "n_pos": int(y0.sum()),
           "n_scenes": len(set(g0.tolist())), "n_rb_train": len(tr_sel),
           "n_rb_holdout": len(ho_sel), "n_aug": len(ya)}
    oof_store = {}

    for name, cfg in variants.items():
        aucs = {"img_roc": [], "img_pr": [], "scene_roc": [], "scene_pr": []}
        for seed in SEEDS:
            oof = np.full(len(y0), np.nan)
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in skf.split(X0, y0, g0):
                tr_groups = set(g0[tr].tolist())
                Xtr, ytr, gtr = [X0[tr]], [y0[tr]], [g0[tr]]
                if cfg["rb"]:
                    Xtr.append(Xrb_tr)
                    ytr.append(np.zeros(len(tr_sel), int))
                    gtr.append(g_rb)
                if cfg["aug"]:
                    m = np.array([(g in tr_groups) or g.startswith("rb_") for g in ga])
                    if not cfg["rb"]:
                        m &= ~np.char.startswith(ga, "rb_")
                    Xtr.append(Xa[m])
                    ytr.append(ya[m])
                    gtr.append(ga[m])
                sc, clf = fit(np.vstack(Xtr), np.concatenate(ytr), np.concatenate(gtr))
                oof[te] = clf.predict_proba(sc.transform(X0[te]))[:, 1]
            sp, sy = {}, {}
            for p, yy, g in zip(oof, y0, g0):
                sp[g] = max(sp.get(g, 0.0), float(p))
                sy[g] = max(sy.get(g, 0), int(yy))
            ks = sorted(sp)
            svp, svy = [sp[k] for k in ks], [sy[k] for k in ks]
            aucs["img_roc"].append(roc_auc_score(y0, oof))
            aucs["img_pr"].append(average_precision_score(y0, oof))
            aucs["scene_roc"].append(roc_auc_score(svy, svp))
            aucs["scene_pr"].append(average_precision_score(svy, svp))
            if seed == SEEDS[0]:
                oof_store[name] = oof.copy()
        res[name] = {k: f"{np.mean(v):.4f} +- {np.std(v):.4f}" for k, v in aucs.items()}
        logger.info("%s: %s", name, res[name])

    # ---- ROC curves (seed-0 OOF), image + scene level ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in variants:
        oof = oof_store[name]
        fpr, tpr, _ = roc_curve(y0, oof)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y0, oof):.3f})")
        sp, sy = {}, {}
        for p, yy, g in zip(oof, y0, g0):
            sp[g] = max(sp.get(g, 0.0), float(p))
            sy[g] = max(sy.get(g, 0), int(yy))
        ks = sorted(sp)
        svy, svp = [sy[k] for k in ks], [sp[k] for k in ks]
        fpr, tpr, _ = roc_curve(svy, svp)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(svy, svp):.3f})")
    for ax, title in zip(axes, [f"Image-level ROC (n={len(y0)}, {int(y0.sum())} pos)",
                                f"Scene-level ROC ({res['n_scenes']} scenes)"]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_curves.png"), dpi=110)

    # ---- FP / FN at the gate from the BEST variant's OOF (seed 0) ----
    best = max(variants, key=lambda n: float(res[n]["scene_pr"].split(" ")[0]))
    res["best_variant"] = best
    oof = oof_store[best]
    fp = [(ids[i], float(oof[i])) for i in range(len(y0))
          if y0[i] == 0 and oof[i] >= 0.5]
    fn = [(ids[i], float(oof[i])) for i in range(len(y0))
          if y0[i] == 1 and oof[i] < GATE]
    fp.sort(key=lambda t: -t[1])
    fn.sort(key=lambda t: t[1])
    res["fp_over_gate"] = sum(1 for _, p in fp if p >= GATE)
    res["fp_over_half"] = len(fp)
    res["fn_under_gate"] = len(fn)
    res["fp_list"] = [{"id": i, "p": round(p, 4)} for i, p in fp[:20]]
    res["fn_list"] = [{"id": i, "p": round(p, 4)} for i, p in fn]

    # ---- full-fit candidate on best variant + RealBlur holdout check ----
    Xtr, ytr, gtr = [X0], [y0], [g0]
    if variants[best]["rb"]:
        Xtr.append(Xrb_tr)
        ytr.append(np.zeros(len(tr_sel), int))
        gtr.append(g_rb)
    if variants[best]["aug"]:
        Xtr.append(Xa)
        ytr.append(ya)
        gtr.append(ga)
    sc, clf = fit(np.vstack(Xtr), np.concatenate(ytr), np.concatenate(gtr))
    np.savez_compressed(CANDIDATE, mean=sc.mean_, scale=sc.scale_, coef=clf.coef_,
                        intercept=np.array(clf.intercept_[0]),
                        version="clip_b32_gmax_v3aug",
                        meta=json.dumps({"trained": "2026-07-10", "variant": best,
                                         "n_orig": len(y0), "n_pos": int(y0.sum()),
                                         "rb": variants[best]["rb"],
                                         "aug": variants[best]["aug"]}))
    z = (Xrb_ho - sc.mean_) / sc.scale_
    p_ho = 1 / (1 + np.exp(-(z @ clf.coef_.ravel() + clf.intercept_[0])))
    res["rb_holdout_candidate"] = {"over_gate": int((p_ho >= GATE).sum()),
                                   "over_half": int((p_ho >= 0.5).sum()),
                                   "max": float(p_ho.max()), "mean": float(p_ho.mean())}
    zpos = (X0[y0 == 1] - sc.mean_) / sc.scale_
    p_pos = 1 / (1 + np.exp(-(zpos @ clf.coef_.ravel() + clf.intercept_[0])))
    res["pos_recall_gate_candidate"] = f"{int((p_pos >= GATE).sum())}/{len(p_pos)}"

    with open(os.path.join(OUT, "partD_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info("PART D: %s", json.dumps(
        {k: v for k, v in res.items() if k not in ("fp_list", "fn_list")}, indent=2))


if __name__ == "__main__":
    main()
