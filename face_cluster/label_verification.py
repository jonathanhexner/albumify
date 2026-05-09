"""Backend logic for the Merge Label Verification tab (spec 017).

Responsible for:
- load_canonical_runs() — reads configs/label_runs.csv at call time (no restart needed)
- load_crop() with fallback to crop_source run
- load_label_verification_data() — background worker: loads run, computes
  candidate pair features, pre-populates DB with heuristic labels.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from face_cluster.features import FeatureComputer, MergeFeatureContext
from face_cluster.loader import load_pipeline_result
from face_cluster.training_db import (
    get_labels_for_run,
    insert_heuristic_samples,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = _REPO_ROOT / "results"
_LABEL_RUNS_CSV = _REPO_ROOT / "configs" / "label_runs.csv"


# ---------------------------------------------------------------------------
# Canonical runs — loaded from configs/label_runs.csv at call time
# ---------------------------------------------------------------------------

def load_canonical_runs(csv_path: Optional[Path] = None) -> Dict[str, Dict]:
    """Load enabled canonical runs from configs/label_runs.csv.

    Returns a dict keyed by dataset name:
        {
            "Google_Germany": {
                "run": "Germany_12",
                "crop_source": None,
                "notes": "...",
                ...
            },
            ...
        }

    Reads the file on every call so changes take effect without restarting the app.
    Rows with enabled=0 are excluded from the returned dict but are preserved in the
    CSV as an audit trail.
    """
    path = Path(csv_path) if csv_path else _LABEL_RUNS_CSV
    if not path.exists():
        logger.warning("label_runs.csv not found at %s — returning empty dict", path)
        return {}
    result: Dict[str, Dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            enabled = row.get("enabled", "1").strip()
            if enabled not in ("1", "true", "yes"):
                continue
            dataset = row["dataset"].strip()
            crop_source_raw = row.get("crop_source", "").strip()
            result[dataset] = {
                "run": row["run_id"].strip(),
                "crop_source": crop_source_raw if crop_source_raw else None,
                "notes": row.get("notes", "").strip(),
            }
    return result


# ---------------------------------------------------------------------------
# Crop loading with fallback
# ---------------------------------------------------------------------------

def load_crop(
    face_id: int,
    run_dir: Path,
    fallback_crop_dir: Optional[Path] = None,
) -> Optional[Image.Image]:
    """Load aligned crop for face_id.

    Search order:
    1. {run_dir}/crops/face_{fid:04d}_aligned.jpg
    2. {fallback_crop_dir}/face_{fid:04d}_aligned.jpg  (if provided)
    """
    dirs: List[Path] = [Path(run_dir) / "crops"]
    if fallback_crop_dir:
        dirs.append(Path(fallback_crop_dir))
    filename = f"face_{face_id:04d}_aligned.jpg"
    for crop_dir in dirs:
        p = crop_dir / filename
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# Result dataclass returned by the background worker
# ---------------------------------------------------------------------------

@dataclass
class LabelVerificationData:
    """All data needed to render the label verification tab."""
    run_id: str
    run_dir: Path
    crop_fallback_dir: Optional[Path]
    # Sorted by min_exemplar_dist. Columns include cluster_a/b, sizes, distance
    # metrics, gate pass/fail, heuristic_label, db_label, db_source, verified.
    pairs_df: pd.DataFrame
    cluster_to_exemplar_ids: Dict[int, List[int]]  # cid -> [face_id, ...]
    cluster_to_face_ids: Dict[int, List[int]]       # cid -> [face_id, ...]


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def load_label_verification_data(
    dataset_key: str,
    candidate_threshold: float = 0.45,
    results_dir: Optional[Path] = None,
) -> LabelVerificationData:
    """Load all data for label verification. Runs in a background thread.

    Steps:
    1. Load pipeline result (faces, clusters, embeddings).
    2. Compute pairwise distance matrix.
    3. Compute candidate pair features (FeatureComputer).
    4. Parse merge_log for heuristic labels and gate outcomes.
    5. Pre-populate training_db with heuristic labels (INSERT OR IGNORE).
    6. Merge existing human labels from DB into pairs_df.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    canonical_runs = load_canonical_runs()
    if dataset_key not in canonical_runs:
        raise ValueError(f"Dataset '{dataset_key}' not found in label_runs.csv (enabled runs: {list(canonical_runs)})")
    cfg = canonical_runs[dataset_key]
    run_id = cfg["run"]
    run_dir = results_dir / run_id
    crop_source = cfg.get("crop_source")
    # crop_fallback_dir points to the crops/ subdirectory of the crop source run
    crop_fallback_dir: Optional[Path] = None
    if crop_source:
        candidate = results_dir / crop_source / "crops"
        if candidate.exists():
            crop_fallback_dir = candidate
        else:
            logger.warning("crop_source dir not found: %s", candidate)

    logger.info("Loading pipeline result: %s", run_dir)
    result = load_pipeline_result(run_dir)
    faces = result.faces
    cr = result.cluster_result

    # Build face-level index maps
    cluster_to_face_ids: Dict[int, List[int]] = {
        cid: [faces[idx].face_id for idx in idxs]
        for cid, idxs in cr.clusters.items()
    }
    cluster_to_exemplar_ids: Dict[int, List[int]] = {
        cid: [faces[idx].face_id for idx in idxs]
        for cid, idxs in cr.exemplars.items()
    }

    # Distance matrix from L2-normalised embeddings
    logger.info("Computing distance matrix for %d faces...", len(faces))
    n = len(faces)
    dim = next(
        (f.embedding_normalized.shape[0] for f in faces if f.embedding_normalized is not None),
        512,
    )
    embs = np.zeros((n, dim), dtype=np.float32)
    for i, f in enumerate(faces):
        if f.embedding_normalized is not None:
            embs[i] = f.embedding_normalized
    dm = (1.0 - embs @ embs.T).clip(0.0, 2.0)

    # Compute candidate pair features
    logger.info("Computing candidate pairs (threshold=%.2f)...", candidate_threshold)
    ctx = MergeFeatureContext(cluster_result=cr, faces=faces, distance_matrix=dm)
    fc = FeatureComputer()
    pair_features = fc.compute_all_pairs(ctx, candidate_threshold=candidate_threshold)
    logger.info("Found %d candidate pairs", len(pair_features))

    # Parse merge_log for heuristic labels and gate outcomes
    heuristic_info: Dict[Tuple[int, int], dict] = {}
    if result.merge_log:
        for entry in result.merge_log:
            ca, cb = int(entry["cluster_a"]), int(entry["cluster_b"])
            key = (min(ca, cb), max(ca, cb))
            heuristic_info[key] = {
                "heuristic_label": 1 if entry.get("action") == "merged" else 0,
                "passes_exemplar": entry.get("passes_exemplar"),
                "passes_support": entry.get("passes_support"),
                "passes_margin": entry.get("passes_margin"),
                "passes_diameter": entry.get("passes_diameter"),
                "rejection_reason": entry.get("rejection_reason"),
            }

    # Build pairs DataFrame
    rows = []
    for (cid_a, cid_b), feat in pair_features.items():
        key = (cid_a, cid_b)
        heur = heuristic_info.get(key, {})
        rows.append({
            "cluster_a": cid_a,
            "cluster_b": cid_b,
            "size_a": feat.size_a or len(cluster_to_face_ids.get(cid_a, [])),
            "size_b": feat.size_b or len(cluster_to_face_ids.get(cid_b, [])),
            "min_exemplar_dist": round(feat.min_exemplar_dist, 4) if feat.min_exemplar_dist is not None else None,
            "p10_cross_dist": round(feat.p10_cross_dist, 4) if feat.p10_cross_dist is not None else None,
            "p50_cross_dist": round(feat.p50_cross_dist, 4) if feat.p50_cross_dist is not None else None,
            "support_fraction": round(feat.support_fraction, 3) if feat.support_fraction is not None else None,
            "post_merge_diameter": round(feat.post_merge_diameter, 4) if feat.post_merge_diameter is not None else None,
            "diameter_expansion": round(feat.diameter_expansion, 3) if feat.diameter_expansion is not None else None,
            "heuristic_label": heur.get("heuristic_label"),
            "passes_exemplar": heur.get("passes_exemplar"),
            "passes_support": heur.get("passes_support"),
            "passes_margin": heur.get("passes_margin"),
            "passes_diameter": heur.get("passes_diameter"),
            "rejection_reason": heur.get("rejection_reason"),
        })

    pairs_df = pd.DataFrame(rows).sort_values("min_exemplar_dist").reset_index(drop=True)

    # Pre-populate DB with heuristic labels (INSERT OR IGNORE — never overwrites human)
    _prepopulate_heuristic_labels(run_id, run_dir, pairs_df, pair_features)

    # Load existing labels from DB and merge in
    existing = get_labels_for_run(run_id)
    if not existing.empty:
        label_map = {
            (int(r["cluster_a"]), int(r["cluster_b"])): r
            for _, r in existing.iterrows()
        }
        def _enrich(row):
            key = (int(row["cluster_a"]), int(row["cluster_b"]))
            db_row = label_map.get(key, {})
            row["db_label"] = db_row.get("label") if isinstance(db_row, dict) else db_row["label"]
            row["db_source"] = db_row.get("source") if isinstance(db_row, dict) else db_row["source"]
            raw_verified = db_row.get("verified", 0) if isinstance(db_row, dict) else db_row["verified"]
            row["verified"] = bool(raw_verified)
            return row
        pairs_df = pairs_df.apply(_enrich, axis=1)
    else:
        pairs_df["db_label"] = None
        pairs_df["db_source"] = None
        pairs_df["verified"] = False

    return LabelVerificationData(
        run_id=run_id,
        run_dir=run_dir,
        crop_fallback_dir=crop_fallback_dir,
        pairs_df=pairs_df,
        cluster_to_exemplar_ids=cluster_to_exemplar_ids,
        cluster_to_face_ids=cluster_to_face_ids,
    )


def _prepopulate_heuristic_labels(
    run_id: str,
    run_dir: Path,
    pairs_df: pd.DataFrame,
    pair_features: Dict,
) -> None:
    """INSERT OR IGNORE heuristic-labelled rows into training_db."""
    existing = get_labels_for_run(run_id)
    existing_keys: set = set()
    if not existing.empty:
        existing_keys = {
            (int(r["cluster_a"]), int(r["cluster_b"]))
            for _, r in existing.iterrows()
        }

    samples = []
    now = datetime.utcnow().isoformat()
    for _, row in pairs_df.iterrows():
        key = (int(row["cluster_a"]), int(row["cluster_b"]))
        if key in existing_keys:
            continue
        feat = pair_features.get(key)
        if feat is None:
            continue
        heur_label = row.get("heuristic_label")
        label_val = None if pd.isna(heur_label) or heur_label is None else int(heur_label)
        samples.append({
            "run_id": run_id,
            "album_path": None,
            "cluster_a": int(row["cluster_a"]),
            "cluster_b": int(row["cluster_b"]),
            "label": label_val,
            "feature_version": 3,
            "features_json": json.dumps(feat.to_dict()),
            "output_dir": str(run_dir),
            "exemplar_ids": None,
            "saved_at": now,
            "source": "heuristic" if label_val is not None else None,
            "verified": 0,
        })

    if samples:
        n = insert_heuristic_samples(samples)
        logger.info("Pre-populated %d heuristic pairs for run %s", n, run_id)
