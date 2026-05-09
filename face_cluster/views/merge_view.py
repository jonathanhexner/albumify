"""MergeAnalysisView, MergeComparisonView, MergeDecisionRow, MergeGroup, and ML merge helpers."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_cluster.merge import CandidateGroup, group_merge_candidates
from face_cluster.pipeline import PipelineResult
from face_cluster.types import FaceRecord
from face_cluster.views._base import _embeddings_matrix

logger = logging.getLogger(__name__)

_GATES = ("exemplar", "support", "margin", "diameter")


@dataclass
class MergeDecisionRow:
    """One merge decision entry (either an actual merge or a rejected candidate)."""
    cluster_a: int
    cluster_b: int
    exemplar_dist: float
    threshold_used: float
    support: int
    action: str                      # "merged" | "rejected"
    rejection_reason: Optional[str]
    exemplar_face_ids_a: List[int]   # up to 3 exemplar face_ids from base cluster A
    exemplar_face_ids_b: List[int]   # up to 3 exemplar face_ids from base cluster B
    # Enriched fields (populated from full merge_log; safe defaults for old-format logs)
    cluster_a_size: int = 0
    cluster_b_size: int = 0
    T_a: Optional[float] = None
    T_b: Optional[float] = None
    T_global: Optional[float] = None
    required_support: int = 0
    post_diameter: float = 0.0
    max_allowed_diameter: float = 0.0
    passes_exemplar: bool = False
    passes_support: bool = False
    passes_margin: bool = False
    passes_diameter: bool = False
    n_gates_passed: int = 0    # count of True passes_* values
    # Margin gate detail (None for old runs without margin logging)
    margin_gap: Optional[float] = None        # competitor_dist - dist_to_b; PASS if >= merge_margin
    margin_dist_to_b: Optional[float] = None  # distance from worst exemplar to cluster B
    margin_competitor_dist: Optional[float] = None  # distance to nearest competing cluster
    margin_competitor_id: Optional[int] = None      # ID of competing cluster
    # Cross-distance OR gate (spec-019)
    p25_cross_dist: Optional[float] = None   # p25 of ALL cross-cluster node-pair distances
    passes_cross: Optional[bool] = None      # p25_cross_dist <= merge_cross_threshold
    unique_support: Optional[int] = None     # greedy bipartite matching support count
    # ML prediction fields (None in heuristic mode; both set or both None)
    ml_prob: Optional[float] = None   # ML merge probability 0.0-1.0
    ml_pred: Optional[int] = None     # ML class prediction: 1=merge, 0=reject
    # Iteration this entry was evaluated in (0 = unknown / old log format)
    iteration: int = 0
    # spec-030: True only for the iteration's executed winner.  Distinguishes
    # MERGED (actually_merged=True) from PASSED (action="passed",
    # actually_merged=False, passed all gates but lost the iteration tie-break).
    actually_merged: bool = False


@dataclass
class MergeGroup:
    """A connected component of merge candidates, with UI-facing data attached.

    Wraps a CandidateGroup (pure algorithm output from merge.py) with the
    full MergeDecisionRow objects and face-count totals needed for rendering.
    """
    core: CandidateGroup               # algorithm output — cluster_ids, cohesion, confidence
    pairs: List[MergeDecisionRow]      # full pair data for rendering (gate badges, crops)
    total_faces: int                   # sum of unique cluster face counts in this group
    n_heuristic_merged: int            # pairs the algorithm auto-merged (action=="merged")
    n_heuristic_rejected: int          # pairs the algorithm rejected
    representative_pair: MergeDecisionRow  # pair with lowest exemplar_dist (for card preview)

    # Convenience delegates so callers don't have to reach into .core
    @property
    def group_id(self) -> int:
        return self.core.group_id

    @property
    def cluster_ids(self) -> List[int]:
        return self.core.cluster_ids

    @property
    def confidence(self) -> str:
        return self.core.confidence

    @property
    def cohesion(self) -> float:
        return self.core.cohesion

    @property
    def min_gates(self) -> int:
        return self.core.min_gates

    @property
    def max_gates(self) -> int:
        return self.core.max_gates


@dataclass
class MergeAnalysisView:
    """Analysis of merge decisions: gate bottlenecks, thresholds, near-misses."""

    n_clusters_base: int
    n_clusters_merged: int
    n_noise_base: int
    n_noise_merged: int
    merges: List[MergeDecisionRow]
    rejections: List[MergeDecisionRow]
    cluster_mapping: Dict[int, List[int]]   # merged_id -> [base_ids absorbed] (>1 only)

    # Threshold data from merge_metadata (None for old runs)
    cluster_thresholds: Optional[Dict[int, float]]
    global_threshold: Optional[float]

    # Gate analysis computed from rejections
    gate_rejection_counts: Dict[str, int]    # per gate: total rejections where gate failed
    gate_sole_blocker_counts: Dict[str, int] # per gate: rejections where ONLY this gate failed

    near_misses: List[MergeDecisionRow]      # rejections with n_gates_passed == 3

    # Config values from merge_metadata (None for old runs without config block)
    merge_margin: Optional[float] = None     # required margin gap to pass margin gate

    # Transitive grouping (computed from all candidates via group_merge_candidates)
    merge_groups: List["MergeGroup"] = field(default_factory=list)
    n_auto_approve: int = 0
    n_review: int = 0
    n_auto_reject: int = 0
    # ML mode metadata (None in heuristic mode)
    ml_threshold: Optional[float] = None  # threshold used when view was built in ML mode
    pair_features: Optional[Dict] = None  # keyed by (min_cid, max_cid) ClusterPairFeatures
    # Iteration visibility (SIGHTING-029)
    n_iterations: int = 0
    iter_timeline: List[Dict] = field(default_factory=list)
    all_rejection_rows: List["MergeDecisionRow"] = field(default_factory=list)
    pair_history: Dict = field(default_factory=dict)  # (min_a,max_b) -> List[MergeDecisionRow]

    @classmethod
    def compute(cls, result: PipelineResult) -> "MergeAnalysisView":
        cr = result.cluster_result
        mr = result.merged_cluster_result
        faces = result.faces

        n_merged = mr.n_clusters if mr else cr.n_clusters
        n_noise_merged = mr.n_noise if mr else cr.n_noise

        cluster_mapping = _build_cluster_mapping(cr, mr, faces) if mr else {}
        merges, all_rejection_rows = _parse_merge_log(result.merge_log or [], cr, faces)
        rejections = _latest_per_pair(all_rejection_rows)
        pair_history = _build_pair_history(all_rejection_rows)
        iter_timeline = _build_iter_timeline(result.merge_log or [])
        n_iterations = max((e["iteration"] for e in iter_timeline), default=0)

        gate_rejection_counts, gate_sole_blocker_counts = _compute_gate_stats(rejections)
        near_misses = [r for r in rejections if r.n_gates_passed == 3]

        metadata = result.merge_metadata or {}
        raw_thresholds = metadata.get("cluster_thresholds")
        cluster_thresholds = (
            {int(k): float(v) for k, v in raw_thresholds.items()}
            if raw_thresholds else None
        )
        global_threshold = (
            float(metadata["global_threshold"])
            if "global_threshold" in metadata else None
        )

        config_block = metadata.get("config", {})
        merge_margin = config_block.get("merge_margin")

        # Build transitive groups from all candidates (merges + rejections)
        all_rows = merges + rejections
        merge_groups = _build_merge_groups(all_rows)
        n_auto_approve = sum(1 for g in merge_groups if g.confidence == "auto_approve")
        n_review = sum(1 for g in merge_groups if g.confidence == "review")
        n_auto_reject = sum(1 for g in merge_groups if g.confidence == "auto_reject")

        return cls(
            n_clusters_base=cr.n_clusters,
            n_clusters_merged=n_merged,
            n_noise_base=cr.n_noise,
            n_noise_merged=n_noise_merged,
            merges=merges,
            rejections=rejections,
            cluster_mapping=cluster_mapping,
            cluster_thresholds=cluster_thresholds,
            global_threshold=global_threshold,
            gate_rejection_counts=gate_rejection_counts,
            gate_sole_blocker_counts=gate_sole_blocker_counts,
            near_misses=near_misses,
            merge_margin=merge_margin,
            merge_groups=merge_groups,
            n_auto_approve=n_auto_approve,
            n_review=n_review,
            n_auto_reject=n_auto_reject,
            n_iterations=n_iterations,
            iter_timeline=iter_timeline,
            all_rejection_rows=all_rejection_rows,
            pair_history=pair_history,
        )


# Keep alias for backward compatibility with any code that imports MergeComparisonView
MergeComparisonView = MergeAnalysisView


def _exemplar_face_ids(cluster_id: int, cr, faces: List[FaceRecord]) -> List[int]:
    """Return up to 3 exemplar face_ids for a cluster from the base result."""
    indices = cr.exemplars.get(cluster_id, cr.clusters.get(cluster_id, []))
    return [faces[i].face_id for i in indices[:3] if i < len(faces)]


def _parse_merge_log(
    merge_log: List[Dict],
    cr,
    faces: List[FaceRecord],
) -> Tuple[List[MergeDecisionRow], List[MergeDecisionRow]]:
    """Parse merge log into actual merges and ALL rejected pair entries (all iterations).

    Returns:
        (merges, all_rejection_rows) where all_rejection_rows preserves every
        iteration's entry for each rejected pair — use _latest_per_pair() for
        the deduplicated default display.
    """
    merges: List[MergeDecisionRow] = []
    all_rejection_rows: List[MergeDecisionRow] = []

    for entry in merge_log:
        cid_a = entry["cluster_a"]
        cid_b = entry["cluster_b"]
        pe = bool(entry.get("passes_exemplar", False))
        ps = bool(entry.get("passes_support", False))
        pm = bool(entry.get("passes_margin", False))
        pd_ = bool(entry.get("passes_diameter", False))
        n_passed = sum([pe, ps, pm, pd_])
        row = MergeDecisionRow(
            cluster_a=cid_a,
            cluster_b=cid_b,
            exemplar_dist=float(entry.get("exemplar_dist", 0.0)),
            threshold_used=float(entry.get("threshold_used", 0.0)),
            support=int(entry.get("support", 0)),
            action=entry.get("action", "rejected"),
            rejection_reason=entry.get("rejection_reason"),
            exemplar_face_ids_a=_exemplar_face_ids(cid_a, cr, faces),
            exemplar_face_ids_b=_exemplar_face_ids(cid_b, cr, faces),
            cluster_a_size=int(entry.get("cluster_a_size", 0)),
            cluster_b_size=int(entry.get("cluster_b_size", 0)),
            T_a=entry.get("T_a"),
            T_b=entry.get("T_b"),
            T_global=entry.get("T_global"),
            required_support=int(entry.get("required_support", 0)),
            post_diameter=float(entry.get("post_diameter", 0.0)),
            max_allowed_diameter=float(entry.get("max_allowed_diameter", 0.0)),
            passes_exemplar=pe,
            passes_support=ps,
            passes_margin=pm,
            passes_diameter=pd_,
            n_gates_passed=n_passed,
            margin_gap=entry.get("margin_gap"),
            margin_dist_to_b=entry.get("margin_dist_to_b"),
            margin_competitor_dist=entry.get("margin_competitor_dist"),
            margin_competitor_id=entry.get("margin_competitor_id"),
            p25_cross_dist=entry.get("p25_cross_dist"),
            passes_cross=entry.get("passes_cross"),
            unique_support=entry.get("unique_support"),
            iteration=int(entry.get("iteration", 0)),
            actually_merged=bool(entry.get("actually_merged", False)),
        )
        if entry.get("actually_merged"):
            merges.append(row)
        else:
            # "rejected" = failed gates, "passed" = passed gates but wasn't the best candidate
            if row.action == "passed":
                row.rejection_reason = "Passed all gates but was not the best candidate in this iteration"
            all_rejection_rows.append(row)

    return merges, all_rejection_rows


def _latest_per_pair(rows: List[MergeDecisionRow]) -> List[MergeDecisionRow]:
    """Return the last iteration's entry for each unique (cluster_a, cluster_b) pair."""
    latest: Dict[Tuple[int, int], MergeDecisionRow] = {}
    for row in rows:
        key = (min(row.cluster_a, row.cluster_b), max(row.cluster_a, row.cluster_b))
        if key not in latest or row.iteration >= latest[key].iteration:
            latest[key] = row
    return list(latest.values())


def _build_pair_history(
    rows: List[MergeDecisionRow],
) -> Dict[Tuple[int, int], List[MergeDecisionRow]]:
    """Group all rejection rows by pair key, sorted by iteration ascending."""
    history: Dict[Tuple[int, int], List[MergeDecisionRow]] = defaultdict(list)
    for row in rows:
        key = (min(row.cluster_a, row.cluster_b), max(row.cluster_a, row.cluster_b))
        history[key].append(row)
    return {k: sorted(v, key=lambda r: r.iteration) for k, v in history.items()}


def _build_iter_timeline(merge_log: List[Dict]) -> List[Dict]:
    """Build one entry per iteration summarising what was merged (or not).

    Each entry: {iteration, merged_a, merged_b, size_a, size_b, n_candidates}
    merged_a/merged_b are None for the final no-merge iteration.
    """
    from collections import Counter
    n_candidates = Counter(entry.get("iteration", 0) for entry in merge_log)
    merged_entries = {
        entry["iteration"]: entry
        for entry in merge_log
        if entry.get("actually_merged")
    }
    all_iters = sorted(n_candidates.keys())
    timeline = []
    for it in all_iters:
        m = merged_entries.get(it)
        timeline.append({
            "iteration":   it,
            "merged_a":    m["cluster_a"] if m else None,
            "merged_b":    m["cluster_b"] if m else None,
            "size_a":      m.get("cluster_a_size", 0) if m else None,
            "size_b":      m.get("cluster_b_size", 0) if m else None,
            "n_candidates": n_candidates[it],
        })
    return timeline


def _compute_gate_stats(
    rejections: List[MergeDecisionRow],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Compute per-gate rejection and sole-blocker counts from rejected pairs."""
    counts: Dict[str, int] = {g: 0 for g in _GATES}
    sole: Dict[str, int] = {g: 0 for g in _GATES}
    gate_pass_attr = {
        "exemplar": "passes_exemplar",
        "support": "passes_support",
        "margin": "passes_margin",
        "diameter": "passes_diameter",
    }
    for row in rejections:
        failed = [g for g, attr in gate_pass_attr.items() if not getattr(row, attr)]
        for g in failed:
            counts[g] += 1
        if len(failed) == 1:
            sole[failed[0]] += 1
    return counts, sole


def _build_merge_groups(all_rows: List[MergeDecisionRow]) -> List[MergeGroup]:
    """Build MergeGroup list from all candidate pairs (merges + rejections).

    Calls group_merge_candidates() for the transitive grouping algorithm,
    then attaches full MergeDecisionRow data for UI rendering.
    """
    if not all_rows:
        return []

    # Build lookup by pair key
    row_by_key: Dict[Tuple[int, int], MergeDecisionRow] = {}
    for row in all_rows:
        key = (min(row.cluster_a, row.cluster_b), max(row.cluster_a, row.cluster_b))
        row_by_key[key] = row

    candidate_pairs = list(row_by_key.keys())
    gate_counts = [row_by_key[k].n_gates_passed for k in candidate_pairs]

    core_groups = group_merge_candidates(candidate_pairs, gate_counts)

    # Build cluster -> face count from rows
    cluster_sizes: Dict[int, int] = {}
    for row in all_rows:
        cluster_sizes[row.cluster_a] = row.cluster_a_size
        cluster_sizes[row.cluster_b] = row.cluster_b_size

    merge_groups: List[MergeGroup] = []
    for core in core_groups:
        pairs = [row_by_key[k] for k in core.pair_keys if k in row_by_key]
        total_faces = sum(cluster_sizes.get(cid, 0) for cid in core.cluster_ids)
        n_merged = sum(1 for p in pairs if p.action == "merged")
        n_rejected = sum(1 for p in pairs if p.action in ("rejected", "passed"))
        rep = min(pairs, key=lambda p: p.exemplar_dist) if pairs else pairs[0]
        merge_groups.append(MergeGroup(
            core=core,
            pairs=pairs,
            total_faces=total_faces,
            n_heuristic_merged=n_merged,
            n_heuristic_rejected=n_rejected,
            representative_pair=rep,
        ))

    return merge_groups


def _prob_to_gate_count(prob: float) -> int:
    """Map ML probability to 0-4 gate-count equivalent for grouping."""
    if prob >= 0.8:
        return 4
    elif prob >= 0.6:
        return 3
    elif prob >= 0.4:
        return 2
    elif prob >= 0.3:
        return 1
    return 0


def compute_ml_merge_view(
    result: "PipelineResult",
    model_payload: Dict,
    threshold: float = 0.5,
    candidate_threshold: float = 0.45,
) -> MergeAnalysisView:
    """Compute ML-based merge analysis view for a pipeline result.

    Uses a trained ML classifier to score all candidate cluster pairs and
    returns a MergeAnalysisView with ml_prob/ml_pred populated on each row.

    Args:
        result: PipelineResult with cluster_result, faces, and output_dir.
        model_payload: Dict from MergeTrainer.load_model().
        threshold: Probability cutoff; >= threshold -> proposed_merge.
        candidate_threshold: Max exemplar distance for candidate discovery.

    Returns:
        MergeAnalysisView with ml_threshold set, ml_prob/ml_pred on all rows.

    Raises:
        ValueError: Feature version mismatch or no embeddings / no candidate pairs.
    """
    from face_cluster.features import (
        FeatureComputer,
        MergeFeatureContext,
        VERSION as FEATURE_VERSION,
    )
    from face_cluster.ml_trainer import MergeTrainer

    # --- Validate feature version ---
    payload_version = model_payload.get("metadata", {}).get("feature_version")
    if payload_version != FEATURE_VERSION:
        raise ValueError(
            f"Feature version mismatch: model requires V{payload_version}, "
            f"current is V{FEATURE_VERSION}"
        )

    # --- Build full n×n distance matrix from face embeddings ---
    faces = result.faces
    n = len(faces)
    emb_rows: List[np.ndarray] = []
    any_embedding = False
    for face in faces:
        emb = (
            face.embedding_normalized
            if face.embedding_normalized is not None
            else getattr(face, "embedding", None)
        )
        if emb is not None:
            normed = emb / (np.linalg.norm(emb) + 1e-9)
            emb_rows.append(normed.astype(np.float32))
            any_embedding = True
        else:
            emb_rows.append(np.zeros(512, dtype=np.float32))

    if not any_embedding:
        raise ValueError(
            "No face embeddings found in result. "
            "Re-run the pipeline to generate embeddings.npy."
        )

    emb_matrix = np.stack(emb_rows)  # (n, 512)
    sims = emb_matrix @ emb_matrix.T
    distance_matrix = np.clip(1.0 - sims, 0.0, 2.0).astype(np.float32)
    np.fill_diagonal(distance_matrix, 0.0)

    # --- Feature computation ---
    ctx = MergeFeatureContext(
        cluster_result=result.cluster_result,
        faces=faces,
        distance_matrix=distance_matrix,
    )
    fc = FeatureComputer()
    pair_features = fc.compute_all_pairs(ctx, candidate_threshold)

    if not pair_features:
        raise ValueError(
            "No candidate pairs found. Check that cluster_result has clusters."
        )

    df = fc.to_dataframe(pair_features)

    # --- ML prediction ---
    df_pred = MergeTrainer().predict(model_payload, df)

    # --- Build MergeDecisionRow objects ---
    cr = result.cluster_result
    all_rows: List[MergeDecisionRow] = []
    merges: List[MergeDecisionRow] = []
    rejections: List[MergeDecisionRow] = []

    for _, pred_row in df_pred.iterrows():
        cid_a = int(pred_row["cluster_a"])
        cid_b = int(pred_row["cluster_b"])
        ml_prob = float(pred_row.get("ml_prob", 0.5))
        if np.isnan(ml_prob):
            ml_prob = float(pred_row.get("ml_pred", 0))
        ml_pred = int(pred_row.get("ml_pred", 0))

        action = "proposed_merge" if ml_pred == 1 else "proposed_reject"
        gate_count = _prob_to_gate_count(ml_prob)

        key = (min(cid_a, cid_b), max(cid_a, cid_b))
        feat = pair_features.get(key)
        size_a = int(feat.size_a or 0) if feat else len(cr.clusters.get(cid_a, []))
        size_b = int(feat.size_b or 0) if feat else len(cr.clusters.get(cid_b, []))

        row = MergeDecisionRow(
            cluster_a=cid_a,
            cluster_b=cid_b,
            exemplar_dist=float(feat.min_exemplar_dist or 0.0) if feat else 0.0,
            threshold_used=threshold,
            support=0,
            action=action,
            rejection_reason=None if ml_pred == 1 else "ml_reject",
            exemplar_face_ids_a=_exemplar_face_ids(cid_a, cr, faces),
            exemplar_face_ids_b=_exemplar_face_ids(cid_b, cr, faces),
            cluster_a_size=size_a,
            cluster_b_size=size_b,
            n_gates_passed=gate_count,
            ml_prob=ml_prob,
            ml_pred=ml_pred,
        )
        all_rows.append(row)
        if ml_pred == 1:
            merges.append(row)
        else:
            rejections.append(row)

    # --- Grouping ---
    merge_groups = _build_merge_groups(all_rows)
    n_auto_approve = sum(1 for g in merge_groups if g.confidence == "auto_approve")
    n_review = sum(1 for g in merge_groups if g.confidence == "review")
    n_auto_reject = sum(1 for g in merge_groups if g.confidence == "auto_reject")
    near_misses = [r for r in all_rows if r.ml_prob is not None and 0.4 < r.ml_prob < 0.6]

    mr = result.merged_cluster_result
    n_merged = mr.n_clusters if mr else cr.n_clusters
    n_noise_merged = mr.n_noise if mr else cr.n_noise

    return MergeAnalysisView(
        n_clusters_base=cr.n_clusters,
        n_clusters_merged=n_merged,
        n_noise_base=cr.n_noise,
        n_noise_merged=n_noise_merged,
        merges=merges,
        rejections=rejections,
        cluster_mapping={},
        cluster_thresholds=None,
        global_threshold=None,
        gate_rejection_counts={},
        gate_sole_blocker_counts={},
        near_misses=near_misses,
        merge_groups=merge_groups,
        n_auto_approve=n_auto_approve,
        n_review=n_review,
        n_auto_reject=n_auto_reject,
        ml_threshold=threshold,
        pair_features=pair_features,
    )


def compute_pair_feature_contributions(
    pair_features: object,
    model_payload: Dict,
    top_n: int = 3,
) -> List[Tuple[str, float, str]]:
    """Compute per-feature contributions for a single candidate pair.

    Args:
        pair_features: ClusterPairFeatures instance (or dict with feature values).
        model_payload: Loaded model bundle from MergeTrainer.load_model().
        top_n: Maximum number of contributions to return.

    Returns:
        List of (feature_name, contribution_value, direction) sorted by
        abs(contribution_value) descending.  direction is "-> merge" if
        contribution is positive, "-> reject" if negative.
    """
    model = model_payload["model"]
    scaler = model_payload["scaler"]
    feature_names: List[str] = model_payload["feature_names"]
    model_type: str = model_payload.get("metadata", {}).get("model_type", "")

    # Build raw feature array
    if hasattr(pair_features, "to_array"):
        feat_array = pair_features.to_array(feature_names)
    elif isinstance(pair_features, dict):
        feat_array = np.array(
            [pair_features.get(name) or 0.0 for name in feature_names], dtype=float
        )
    else:
        feat_array = np.array(
            [getattr(pair_features, name, None) or 0.0 for name in feature_names],
            dtype=float,
        )

    scaled = scaler.transform([feat_array])[0]

    if model_type == "logistic_regression" and hasattr(model, "coef_"):
        contributions = model.coef_[0] * scaled
    elif model_type == "xgboost" and hasattr(model, "feature_importances_"):
        signs = np.sign(feat_array - scaler.mean_)
        contributions = model.feature_importances_ * signs
    elif model_type == "mlp" and hasattr(model, "coefs_"):
        signs = np.sign(feat_array - scaler.mean_)
        importances = np.abs(model.coefs_[0]).sum(axis=1)
        contributions = importances * signs
    else:
        # Fallback: global importance from metadata (if saved), else zeros
        imp_dict: Dict[str, float] = (
            model_payload.get("metadata", {}).get("feature_importance") or {}
        )
        contributions = np.array(
            [imp_dict.get(name, 0.0) for name in feature_names], dtype=float
        )

    result_entries: List[Tuple[str, float, str]] = []
    for name, contrib in zip(feature_names, contributions):
        direction = "-> merge" if contrib >= 0 else "-> reject"
        result_entries.append((name, float(contrib), direction))

    result_entries.sort(key=lambda x: -abs(x[1]))
    return result_entries[:top_n]


def _build_cluster_mapping(cr, mr, faces: List[FaceRecord]) -> Dict[int, List[int]]:
    """Map merged_cluster_id -> list of base_cluster_ids it absorbed (only multi-source)."""
    face_to_base = {f.face_id: int(cr.labels[i]) for i, f in enumerate(faces) if cr.labels[i] >= 0}
    face_to_merged = {f.face_id: int(mr.labels[i]) for i, f in enumerate(faces) if mr.labels[i] >= 0}

    mapping: Dict[int, set] = defaultdict(set)
    for fid, mid in face_to_merged.items():
        bid = face_to_base.get(fid, -1)
        if bid >= 0:
            mapping[mid].add(bid)

    return {mid: sorted(bids) for mid, bids in mapping.items() if len(bids) > 1}
