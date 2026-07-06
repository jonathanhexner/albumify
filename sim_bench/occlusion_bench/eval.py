"""Shared evaluation harness (spec-096 T9.1).

Encodes the four group rules: grouped+stratified CV, scene-level primary metric,
inverse-group-size sample weights, and never plain accuracy.

Two entry points:
- ``cv_evaluate(X, y, groups)`` — for trainable candidates (LoG-stats, probes):
  StratifiedGroupKFold CV; per-fold PR-AUC at image AND scene level.
- ``score_evaluate(y, groups, scores)`` — for zero-shot candidates (classical
  detector, VLMs): direct PR-AUC on given scores.

Scene-level: ground truth per group = its label (groups never mix labels by
construction); scene score = MAX member score (an occlusion anywhere in the
burst flags the scene).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _scene_level(y: np.ndarray, groups: np.ndarray, scores: np.ndarray):
    gids = {}
    for i, g in enumerate(groups):
        gids.setdefault(g, []).append(i)
    ys, ss = [], []
    for g, idx in gids.items():
        ys.append(int(max(y[i] for i in idx)))
        ss.append(float(max(scores[i] for i in idx)))
    return np.array(ys), np.array(ss)


def _pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def score_evaluate(y, groups, scores) -> Dict[str, float]:
    """Zero-shot candidate: PR-AUC image-level + scene-level."""
    y, groups, scores = np.asarray(y), np.asarray(groups), np.asarray(scores, dtype=float)
    sy, ss = _scene_level(y, groups, scores)
    return {"pr_auc_image": _pr_auc(y, scores), "pr_auc_scene": _pr_auc(sy, ss),
            "n_scenes_pos": int(sy.sum()), "n_scenes": len(sy)}


def cv_evaluate(X, y, groups, n_splits: int = 5, C: float = 0.1) -> Dict[str, object]:
    """Trainable candidate: standardized features -> L2 logistic regression.

    sample_weight = balanced class weight x 1/group_size (a burst = one scene's
    worth of gradient). Returns per-fold scene/image PR-AUC + summary.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    X, y, groups = np.asarray(X, dtype=float), np.asarray(y), np.asarray(groups)
    gsize = {g: int((groups == g).sum()) for g in set(groups.tolist())}
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    cls_w = {1: len(y) / (2.0 * n_pos), 0: len(y) / (2.0 * n_neg)}
    w = np.array([cls_w[int(yi)] / gsize[g] for yi, g in zip(y, groups)])

    folds: List[Dict[str, float]] = []
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(sc.transform(X[tr]), y[tr], sample_weight=w[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        sy, ss = _scene_level(y[te], groups[te], p)
        folds.append({"pr_auc_image": _pr_auc(y[te], p), "pr_auc_scene": _pr_auc(sy, ss),
                      "n_pos_scenes": int(sy.sum())})

    scene = [f["pr_auc_scene"] for f in folds if not np.isnan(f["pr_auc_scene"])]
    image = [f["pr_auc_image"] for f in folds if not np.isnan(f["pr_auc_image"])]
    return {"folds": folds,
            "scene_pr_auc_min": min(scene), "scene_pr_auc_max": max(scene),
            "scene_pr_auc_mean": float(np.mean(scene)),
            "image_pr_auc_mean": float(np.mean(image))}


def cv_oof_scores(X, y, groups, n_splits: int = 5, C: float = 0.1):
    """Out-of-fold probabilities: every image scored by the fold-model that did
    NOT train on its group. The honest way to show per-image model scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    X, y, groups = np.asarray(X, dtype=float), np.asarray(y), np.asarray(groups)
    gsize = {g: int((groups == g).sum()) for g in set(groups.tolist())}
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    cls_w = {1: len(y) / (2.0 * n_pos), 0: len(y) / (2.0 * n_neg)}
    w = np.array([cls_w[int(yi)] / gsize[g] for yi, g in zip(y, groups)])
    oof = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(sc.transform(X[tr]), y[tr], sample_weight=w[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof
