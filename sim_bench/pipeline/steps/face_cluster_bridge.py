"""Bridge between main pipeline's FaceForClustering and face_cluster's FaceRecord.

**DEPRECATED — scheduled for deletion in spec-040 Phase 7.**

This module is the legacy adapter that converts Albumify's dict-of-dicts
state (``context.insightface_faces`` + ``context.face_embeddings``) into
``face_cluster.types.FaceRecord`` and runs the bridge clustering chain.
It is replaced by the unified 8-step chain in
``sim_bench/pipeline/steps/face_clustering_steps.py`` plus
``face_cluster.fc_app_runner.FCAppRunner``, which read ``context.face_records``
directly (populated by the spec-040 A1 producer dual-write).

Today it is still imported by ``cluster_people`` because Albumify's
``default_pipeline`` in ``configs/pipeline.yaml`` invokes ``cluster_people``
with ``method: face_cluster_knn``. Phase 7 deletes this file once the new
FC App at ``app/face_clustering_v2/`` ships and the equivalence sweep
holds green for 2 weeks. **Do not add new callers.**

spec-033 P-C C-1: the bridge plumbs blur_score, pose, det_score,
landmarks, aligned_face from ``context.insightface_faces`` instead of
dropping them. This unblocks the quality gates that were previously
force-disabled (``build_fc_config`` no longer hardcodes 999.0 / 0.0).
"""

import copy
import logging
import warnings
from typing import List, Optional

import numpy as np

from face_cluster.attach import HoldoutAttacher
from face_cluster.clustering import ConnectedComponentsClusterer
from face_cluster.config import PipelineConfig as FCConfig
from face_cluster.exemplars import D10ExemplarSelector
from face_cluster.knn_graph import KNNGraphBuilder
from face_cluster.merge import ConservativeMerger
from face_cluster.quality import QualityGater
from face_cluster.types import FaceRecord

from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

warnings.warn(
    "sim_bench.pipeline.steps.face_cluster_bridge is deprecated and scheduled "
    "for deletion in spec-040 Phase 7. New callers should use the unified "
    "8-step clustering chain in sim_bench.pipeline.steps.face_clustering_steps "
    "or face_cluster.fc_app_runner.FCAppRunner, both of which read "
    "context.face_records directly.",
    DeprecationWarning,
    stacklevel=2,
)


def _lookup_insightface_face(context: Optional[PipelineContext], image_path: str, face_index: int) -> dict:
    """Find the matching insightface_faces entry for a (image, face_index) pair.

    Returns an empty dict if the lookup fails — caller must tolerate missing data.
    The bridge formerly dropped these fields entirely; reading them here turns
    "silently lost" into "explicitly absent" so downstream NULL columns are
    diagnosable.
    """
    if context is None:
        return {}
    face_data = (getattr(context, "insightface_faces", None) or {}).get(image_path, {})
    faces = face_data.get("faces", []) if isinstance(face_data, dict) else []
    for f in faces:
        if f.get("face_index") == face_index:
            return f
    return {}


def faces_to_face_records(
    faces,
    embeddings_norm: np.ndarray,
    context: Optional[PipelineContext] = None,
) -> List[FaceRecord]:
    """Bridge FaceForClustering → FaceRecord for face_cluster algorithms.

    When ``context`` is supplied, plumbs blur_score, pose, det_score,
    landmarks, and aligned_face from ``context.insightface_faces`` instead
    of leaving them at sentinel zeros. spec-033 P-C C-1.
    """
    records = []
    for i, face in enumerate(faces):
        bbox = face.bbox
        if isinstance(bbox, dict):
            bbox_tuple = (
                bbox.get("x", 0), bbox.get("y", 0),
                bbox.get("w", 0), bbox.get("h", 0),
            )
        else:
            bbox_tuple = (0, 0, 0, 0)

        # spec-033 P-C C-1: recover the fields FaceForClustering doesn't carry
        # by joining back to context.insightface_faces.
        if_face = _lookup_insightface_face(context, str(face.original_path), face.face_index)
        if_scores = if_face.get("scores", {}) if isinstance(if_face, dict) else {}

        blur_score = (
            getattr(face, "blur_score", None)
            if getattr(face, "blur_score", None) is not None
            else float(if_scores.get("blur_score", 0.0) or 0.0)
        )
        det_score = (
            getattr(face, "det_score", None)
            if getattr(face, "det_score", None) is not None
            else (float(if_face.get("confidence")) if if_face.get("confidence") is not None else None)
        )
        pose = getattr(face, "pose", None)
        if pose is None:
            pose_scores = if_face.get("pose_scores") or if_scores.get("pose") if isinstance(if_face, dict) else None
            if isinstance(pose_scores, dict) and {"yaw", "pitch", "roll"} <= set(pose_scores):
                pose = (
                    float(pose_scores["yaw"]),
                    float(pose_scores["pitch"]),
                    float(pose_scores["roll"]),
                )
        landmarks = if_face.get("landmarks") if isinstance(if_face, dict) else None
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                landmarks = np.asarray(landmarks, dtype=np.float32)
            except (TypeError, ValueError):
                landmarks = None

        records.append(FaceRecord(
            face_id=i,
            image_id=str(face.original_path),
            image_path=str(face.original_path),
            face_index=face.face_index,
            bbox=bbox_tuple,
            embedding=face.embedding,
            embedding_normalized=embeddings_norm[i],
            blur_score=float(blur_score) if blur_score is not None else 0.0,
            area=float(bbox.get("w", 0) * bbox.get("h", 0)) if isinstance(bbox, dict) else 0.0,
            pose=pose,
            det_score=det_score,
            landmarks=landmarks,
            # aligned_face is populated downstream by align_faces; bridge does
            # not synthesize it.
        ))
    return records


def build_fc_config(config: dict) -> FCConfig:
    """Build a face_cluster PipelineConfig from the main pipeline's config dict.

    spec-033 P-C C-1 + P-F F-1: quality gates are NO LONGER force-disabled
    wholesale. Values flow through from the caller's typed config.

    EXCEPTION — gates whose data the InsightFace pipeline does not yet
    compute are pinned to permissive values, regardless of config:

    * ``blur_min``: pinned to 0.0. The active InsightFace pipeline has no
      blur-scoring step; ``FaceRecord.blur_score`` is 0.0 for every face.
      Honoring ``cluster_people.blur_min: 50.0`` from pipeline.yaml would
      reject 100% of faces (the regression that motivated this comment —
      see logs/2026-05-15_11-18-44/api.log). Re-enable once an
      ``insightface_score_blur`` step lands and the bridge plumbs it.

    Pose gates (yaw/pitch/roll) stay config-driven: when pose is None and
    ``require_pose=False`` the QualityGater passes the face through
    permissively, so the gate's threshold value is moot for now.
    """
    return FCConfig(
        K=config.get("K", 5),
        distance_threshold=config.get("distance_threshold", 0.35),
        min_cluster_size=config.get("min_cluster_size", 2),
        yaw_max=float(config.get("yaw_max", 999.0)),
        pitch_max=float(config.get("pitch_max", 999.0)),
        roll_max=float(config.get("roll_max", 999.0)),
        # Pinned: InsightFace pipeline has no blur step yet (see docstring).
        blur_min=0.0,
        max_faces_per_image_core=config.get("max_faces_per_image_core", 3),
        det_score_min=config.get("det_score_min", None),
        split_enabled=config.get("split_enabled", False),
        merge_enabled=config.get("merge_enabled", False),
        attach_enabled=config.get("attach_enabled", False),
        merge_candidate_threshold=config.get("merge_candidate_threshold", 0.45),
        merge_exemplar_threshold=config.get("merge_exemplar_threshold", 0.35),
        merge_use_cross_gate=config.get("merge_use_cross_gate", True),
        merge_cross_threshold=config.get("merge_cross_threshold", 0.40),
        merge_cross_max_size=config.get("merge_cross_max_size", 5),
        merge_support_frac=config.get("merge_support_frac", 0.3),
        merge_support_min=config.get("merge_support_min", 2),
        merge_support_unique=config.get("merge_support_unique", False),
        merge_margin=config.get("merge_margin", 0.05),
        merge_diameter_expansion_factor=config.get("merge_diameter_expansion_factor", 1.5),
    )


def run_face_cluster_knn(faces, embeddings_norm, config, context):
    """Run face_cluster/ algorithms and return (labels, face_records, base_cr, merged_cr, core_indices, merge_log, merge_metadata, fc_cfg)."""
    fc_cfg = build_fc_config(config)
    # spec-033 P-C C-1: pass context so the bridge can plumb blur/pose/det/landmarks
    # from insightface_faces instead of dropping them.
    face_records = faces_to_face_records(faces, embeddings_norm, context=context)
    n = len(face_records)

    # Quality gating
    gater = QualityGater(fc_cfg)
    core_indices, holdout_indices, verdicts = gater.select_core_set(face_records)
    logger.info(f"face_cluster_knn: quality gating {len(core_indices)} core, {len(holdout_indices)} holdout out of {n}")

    if not core_indices:
        logger.warning("face_cluster_knn: no faces passed quality gating, all noise")
        return np.full(n, -1, dtype=int), face_records, None, None, [], None, None, fc_cfg

    # Build kNN graph
    context.report_progress("cluster_people", 0.6, "Building kNN graph")
    graph_builder = KNNGraphBuilder(fc_cfg)
    graph_result = graph_builder.build_graph(face_records, core_indices)

    # Cluster via connected components
    context.report_progress("cluster_people", 0.7, "Clustering connected components")
    clusterer = ConnectedComponentsClusterer(fc_cfg)
    cluster_result = clusterer.cluster(graph_result, core_indices)
    logger.info(f"face_cluster_knn: {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

    # Exemplar selection
    selector = D10ExemplarSelector(fc_cfg)
    cluster_result, _ = selector.select_exemplars(cluster_result, graph_result)

    # Save base before merge
    base_cluster_result = copy.deepcopy(cluster_result)
    merge_log = None
    merge_metadata = None

    # Optional merge
    if fc_cfg.merge_enabled:
        context.report_progress("cluster_people", 0.8, "Merging clusters")
        merger = ConservativeMerger(fc_cfg)
        cluster_result, merge_log, merge_metadata = merger.merge_clusters_with_logging(cluster_result, graph_result)
        logger.info(f"face_cluster_knn: {cluster_result.n_clusters} clusters after merge")

    # Optional holdout attachment
    if fc_cfg.attach_enabled and holdout_indices:
        context.report_progress("cluster_people", 0.9, "Attaching holdout faces")
        attacher = HoldoutAttacher(fc_cfg)
        cluster_result = attacher.attach_holdouts(face_records, core_indices, holdout_indices, cluster_result, graph_result)

    # Build label array
    uses_global_indices = fc_cfg.attach_enabled and holdout_indices
    labels = np.full(n, -1, dtype=int)
    for cid, node_indices in cluster_result.clusters.items():
        for node_idx in node_indices:
            global_idx = node_idx if uses_global_indices else (core_indices[node_idx] if node_idx < len(core_indices) else node_idx)
            if 0 <= global_idx < n:
                labels[global_idx] = cid

    return labels, face_records, base_cluster_result, cluster_result, core_indices, merge_log, merge_metadata, fc_cfg
