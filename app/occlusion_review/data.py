"""Occlusion review app — pure data layer (spec-096, no Streamlit).

Loads the dataset manifest + every model's opinions into one aligned table,
computes the disagreement worklist, and reads/writes ``corrections.csv``
(the adjudication output; resumable, append-only).

Decision vocabulary (user-approved): ``occluded_l1|occluded_l2|occluded_l3``,
``clean``, ``foreground_object`` (branch/head/strap near the camera but not on
the lens — kept as its own class so its fate is a later, explicit decision).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

DECISIONS = ["occluded_l1", "occluded_l2", "occluded_l3", "clean", "foreground_object"]


def dataset_root() -> str:
    return os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")


def image_path(root: str, row: dict) -> str:
    sub = "positives" if str(row["label"]) == "1" else "negatives"
    return os.path.join(root, sub, row["id"])


def load_all(root: Optional[str] = None) -> List[dict]:
    """One aligned record per image: manifest + haiku + classical (+ LoG prob if cached)."""
    from sim_bench.occlusion_bench.dataset import load_manifest
    root = root or dataset_root()
    rows = load_manifest(root)

    def _csv(name: str) -> Dict[str, dict]:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            return {}
        with open(p, newline="", encoding="utf-8") as f:
            return {r["id"]: r for r in csv.DictReader(f)}

    haiku = _csv("haiku_labels.csv")
    classical = _csv("classical_scores.csv")
    log_probs = _load_log_probs(root, rows)
    # per-image model scores: OOF probes (model_scores.csv), CNN, tiny-VLM —
    # merged dynamically so new candidates appear in the app automatically
    oof = _csv("model_scores.csv")
    cnn = _csv("cnn_scores.csv")
    vlm = _csv("tinyvlm_scores.csv")

    def _f(d, rid, key):
        try:
            return float(d.get(rid, {}).get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    out = []
    for r in rows:
        h = haiku.get(r["id"], {})
        rec = dict(r)
        rec["haiku_occluded"] = h.get("haiku_occluded", "")
        rec["haiku_level"] = h.get("haiku_level", "")
        rec["haiku_reason"] = h.get("haiku_reason", "")
        rec["classical_score"] = float(classical.get(r["id"], {}).get("score", 0.0) or 0.0)
        rec["log_prob"] = log_probs.get(r["id"], float("nan"))
        rec["B_clip_oof"] = _f(oof, r["id"], "B_clip_oof")
        rec["F_log_oof"] = _f(oof, r["id"], "F_log_oof")
        rec["D2_cnn"] = _f(cnn, r["id"], "score")
        rec["C_tinyvlm_level"] = _f(vlm, r["id"], "level")
        out.append(rec)
    return out


def _load_log_probs(root: str, rows: List[dict]) -> Dict[str, float]:
    """In-sample LoG-LR probabilities for display/explainability.

    NOTE: fit on ALL data — fine for explanation, NOT an eval number (the
    leaderboard stays CV-based). npz rows align with the manifest (extraction
    iterates the manifest and skipped 0); verified by length check.
    """
    import numpy as np
    p = os.path.join(root, "features_log.npz")
    if not os.path.isfile(p):
        return {}
    d = np.load(p, allow_pickle=True)
    X, y = d["X"], d["y"]
    if X.shape[0] != len(rows):
        return {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    probs = clf.predict_proba(sc.transform(X))[:, 1]
    return {r["id"]: float(pb) for r, pb in zip(rows, probs)}


def disagreements(records: List[dict]) -> List[dict]:
    """Haiku-vs-user worklist: flagged negatives first (hidden positives?), then missed positives."""
    fp = [r for r in records if r["label"] == "0" and r["haiku_occluded"] == "1"]
    fn = [r for r in records if r["label"] == "1" and r["haiku_occluded"] == "0"]
    fp.sort(key=lambda r: -int(r["haiku_level"] or 0))
    return fp + fn


def corrections_path(root: Optional[str] = None) -> str:
    return os.path.join(root or dataset_root(), "corrections.csv")


def load_corrections(root: Optional[str] = None) -> Dict[str, dict]:
    p = corrections_path(root)
    if not os.path.isfile(p):
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}  # last write wins per id


def save_correction(image_id: str, original_label: str, decision: str,
                    notes: str = "", root: Optional[str] = None) -> None:
    assert decision in DECISIONS, f"bad decision: {decision}"
    p = corrections_path(root)
    new = not os.path.isfile(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "original_label", "decision", "notes", "reviewed_at"])
        w.writerow([image_id, original_label, decision, notes,
                    datetime.now().isoformat(timespec="seconds")])


def effective_label(rec: dict, corrections: Dict[str, dict]) -> str:
    """Label after corrections: '1', '0', or 'fg' (foreground_object)."""
    c = corrections.get(rec["id"])
    if not c:
        return str(rec["label"])
    d = c["decision"]
    if d.startswith("occluded"):
        return "1"
    if d == "clean":
        return "0"
    return "fg"
