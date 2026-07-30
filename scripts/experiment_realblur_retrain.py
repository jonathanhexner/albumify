"""Experiment Part C (reports/2026-07-10_blur_robustness): does adding RealBlur-J
blurred images as hard negatives IMPROVE the occlusion detector?

Recipe: user's representation-over-quantity rule — 1 image per RealBlur TRAIN
scene (~60% of scenes), holding out the remaining scenes to measure the false
alarm rate honestly. Candidate artifact saved OUTSIDE models/ (production v1
untouched until user approves).

Compared v1 vs v2-candidate:
  1. RealBlur HOLDOUT scenes: over-gate false alarms (the improvement target)
  2. Scene PR-AUC on the original 832 via grouped+stratified 5-fold CV
     (realblur negatives pinned to train folds; metric over original scenes only)
  3. Positive recall at gate (must not collapse)
  4. Synthetic blur separation (0/60 baseline)
  5. Explainability: hot-tile-vs-LoG-box agreement (13/26 baseline)
  6. Own-album near-misses (Austria burst) P before/after

Outputs: reports/2026-07-10_blur_robustness/partC_results.json (+ imgs_partC/),
candidate artifact D:/occlusion_dataset/realblur/clip_b32_gmax_v2rb.npz
"""

from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("partC")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
RB = os.path.join(ROOT, "realblur")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_blur_robustness")
CANDIDATE = os.path.join(RB, "clip_b32_gmax_v2rb.npz")
GATE = 0.8
TRAIN_SCENE_FRAC = 0.6

NEAR_MISSES = ["austria24__20230818_112645.jpg", "austria24__20230818_112650.jpg",
               "austria24__20230818_112656.jpg", "occl__20260705_214913.jpg",
               "budapest__20250822_123518.jpg", "budapest__20250822_123521.jpg"]


def corrected_labels(ids, y_orig):
    with open(os.path.join(ROOT, "corrections.csv"), newline="", encoding="utf-8") as f:
        corr = {r["id"]: r["decision"] for r in csv.DictReader(f)}
    y = y_orig.copy().astype(int)
    for i, rid in enumerate(ids):
        d = corr.get(str(rid))
        if d and d != "foreground_object":
            y[i] = 1 if d.startswith("occluded") else 0
    return y


def head_batch(X, mean, scale, coef, intercept):
    z = (X - mean) / scale
    return 1.0 / (1.0 + np.exp(-(z @ coef.ravel() + intercept)))


def fit(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    gsize = {gg: int((groups == gg).sum()) for gg in set(groups.tolist())}
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    cls_w = {1: len(y) / (2.0 * n_pos), 0: len(y) / (2.0 * n_neg)}
    w = np.array([cls_w[int(yi)] / gsize[gg] for yi, gg in zip(y, groups)])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(X), y, sample_weight=w)
    return sc, clf


def scene_pr_auc(probs, y, groups, keep_mask):
    """Scene-level PR-AUC restricted to scenes whose images pass keep_mask."""
    from sklearn.metrics import average_precision_score
    sp, sy = {}, {}
    for p, yy, g, k in zip(probs, y, groups, keep_mask):
        if not k:
            continue
        sp[g] = max(sp.get(g, 0.0), float(p))
        sy[g] = max(sy.get(g, 0), int(yy))
    keys = sorted(sp)
    return float(average_precision_score([sy[k] for k in keys], [sp[k] for k in keys]))


def cv_scene_auc(X, y, groups, orig_mask, pin_train_mask, seed=0):
    """5-fold grouped+stratified CV; rows with pin_train_mask are ALWAYS train.
    Returns scene PR-AUC over original scenes + OOF probs for original rows."""
    from sklearn.model_selection import StratifiedGroupKFold
    oof = np.full(len(y), np.nan)
    idx_cv = np.where(~pin_train_mask)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(idx_cv, y[idx_cv], groups[idx_cv]):
        tr_idx = np.concatenate([idx_cv[tr], np.where(pin_train_mask)[0]])
        te_idx = idx_cv[te]
        sc, clf = fit(X[tr_idx], y[tr_idx], groups[tr_idx])
        oof[te_idx] = clf.predict_proba(sc.transform(X[te_idx]))[:, 1]
    m = orig_mask & ~np.isnan(oof)
    return scene_pr_auc(oof[m], y[m], groups[m], np.ones(m.sum(), bool)), oof


def explain_agree(mean, scale, coef, intercept, d, ids, y, sub):
    """Hot-tile-vs-LoG-box agreement on positives (baseline 13/26)."""
    from sim_bench.occlusion_bench.saliency import blur_bbox
    agree, with_box, hot = 0, 0, {}
    for i in range(len(ids)):
        if y[i] != 1:
            continue
        rid = str(ids[i])
        eg, et = d["emb_global"][i], d["emb_tiles"][i]
        tiles = [float(head_batch(np.concatenate([eg, t])[None, :],
                                  mean, scale, coef, intercept)[0]) for t in et]
        t = np.array(tiles).reshape(3, 3)
        hy, hx = np.unravel_index(t.argmax(), t.shape)
        hot[rid] = (int(hx), int(hy))
        bb = blur_bbox(os.path.join(ROOT, sub[rid], rid))
        if bb is None:
            continue
        with_box += 1
        cx, cy = (hx + 0.5) / 3, (hy + 0.5) / 3
        agree += bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]
    return agree, with_box, hot


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    y0 = corrected_labels(ids, d["y"].astype(int))
    g0 = np.array([str(x) for x in d["groups"]])
    X0 = np.concatenate([d["emb_global"], d["emb_tiles"].max(axis=1)], axis=1)
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        sub = {r["id"]: ("positives" if r["label"] == "1" else "negatives")
               for r in csv.DictReader(f)}

    rb = np.load(os.path.join(RB, "realblur_embeddings.npz"), allow_pickle=True)
    rb_names = [str(x) for x in rb["names"]]
    rb_scenes = sorted({n.split("__")[0] for n in rb_names})
    n_train_scenes = int(len(rb_scenes) * TRAIN_SCENE_FRAC)
    train_scenes = set(rb_scenes[:n_train_scenes])       # deterministic split
    holdout_scenes = set(rb_scenes[n_train_scenes:])
    # representation rule: 1 image per train scene
    seen, train_sel = set(), []
    for i, n in enumerate(rb_names):
        s = n.split("__")[0]
        if s in train_scenes and s not in seen:
            seen.add(s)
            train_sel.append(i)
    hold_idx = [i for i, n in enumerate(rb_names) if n.split("__")[0] in holdout_scenes]
    logger.info("realblur: %d train imgs (1/scene from %d scenes), %d holdout imgs (%d scenes)",
                len(train_sel), len(train_scenes), len(hold_idx), len(holdout_scenes))

    Xrb = np.concatenate([rb["emb_global"], rb["emb_tiles"].max(axis=1)], axis=1)
    X = np.vstack([X0, Xrb[train_sel]])
    y = np.concatenate([y0, np.zeros(len(train_sel), int)])
    g = np.concatenate([g0, np.array(["rb_" + rb_names[i].split("__")[0] for i in train_sel])])
    orig_mask = np.arange(len(y)) < len(y0)
    rb_mask = ~orig_mask

    # ---- v1 (production artifact) ----
    a1 = np.load(os.path.join(REPO, "models", "occlusion", "clip_b32_gmax_v1.npz"),
                 allow_pickle=True)
    v1 = (a1["mean"], a1["scale"], a1["coef"], float(a1["intercept"]))

    # ---- v2 candidate: full fit on 832 + realblur train negatives ----
    sc2, clf2 = fit(X, y, g)
    np.savez_compressed(CANDIDATE, mean=sc2.mean_, scale=sc2.scale_, coef=clf2.coef_,
                        intercept=np.array(clf2.intercept_[0]), version="clip_b32_gmax_v2rb",
                        meta=json.dumps({"trained": "2026-07-10",
                                         "added_negatives": f"realblur-J 1/scene x{len(train_sel)}",
                                         "base": "clip_b32_gmax_v1 recipe"}))
    v2 = (sc2.mean_, sc2.scale_, clf2.coef_, float(clf2.intercept_[0]))

    res = {"gate": GATE, "n_rb_train": len(train_sel), "n_rb_holdout": len(hold_idx)}

    # 1) RealBlur holdout false alarms
    Xh = Xrb[hold_idx]
    for tag, (m_, s_, c_, b_) in (("v1", v1), ("v2", v2)):
        p = head_batch(Xh, m_, s_, c_, b_)
        res[f"holdout_{tag}"] = {"over_gate": int((p >= GATE).sum()),
                                 "over_half": int((p >= 0.5).sum()),
                                 "max": float(p.max()), "mean": float(p.mean())}

    # 2) CV scene PR-AUC on original scenes (v1 recipe vs v2 recipe), same seed
    auc_v1, _ = cv_scene_auc(X0, y0, g0, np.ones(len(y0), bool),
                             np.zeros(len(y0), bool))
    auc_v2, _ = cv_scene_auc(X, y, g, orig_mask, rb_mask)
    res["cv_scene_pr_auc_orig"] = {"v1_recipe": round(auc_v1, 4),
                                   "v2_recipe_rb_pinned": round(auc_v2, 4)}

    # 3) positive recall at gate (in-sample, both heads)
    pos = y0 == 1
    for tag, (m_, s_, c_, b_) in (("v1", v1), ("v2", v2)):
        p = head_batch(X0[pos], m_, s_, c_, b_)
        res[f"pos_recall_gate_{tag}"] = f"{int((p >= GATE).sum())}/{int(pos.sum())}"

    # 4) synthetic blur separation for v2 (v1 baseline: 0/60,0/60,0/60)
    import sys
    from sim_bench.occlusion_bench.scorer import OcclusionScorer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_occlusion_artifact as ta
    scorer2 = OcclusionScorer(artifact_path=CANDIDATE)
    res["synthetic_blur_v2"] = ta.validate_blur(scorer2, d, ids, y0)

    # 5) explainability agreement
    for tag, (m_, s_, c_, b_) in (("v1", v1), ("v2", v2)):
        agree, nbox, hot = explain_agree(m_, s_, c_, b_, d, ids, y0, sub)
        res[f"explain_{tag}"] = f"{agree}/{nbox}"
        res[f"_hot_{tag}"] = hot

    # 6) near-misses before/after
    nm = {}
    idx = {rid: i for i, rid in enumerate(ids)}
    for rid in NEAR_MISSES:
        i = idx[rid]
        row = {}
        for tag, (m_, s_, c_, b_) in (("v1", v1), ("v2", v2)):
            row[tag] = round(float(head_batch(X0[i][None, :], m_, s_, c_, b_)[0]), 4)
        nm[rid] = row
    res["near_misses"] = nm

    hot1, hot2 = res.pop("_hot_v1"), res.pop("_hot_v2")
    res["hot_tile_changed"] = sum(1 for k in hot1 if hot1[k] != hot2.get(k))
    with open(os.path.join(OUT, "partC_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info("PART C: %s", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
