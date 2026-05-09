"""Bridge between main pipeline's FaceForClustering and face_cluster's FaceRecord.

Runs the face_cluster_knn algorithm: quality gating → kNN graph →
connected components → exemplar selection → optional merge/attach.
"""

import copy
import logging
from typing import List

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


def faces_to_face_records(faces, embeddings_norm: np.ndarray) -> List[FaceRecord]:
    """Bridge FaceForClustering → FaceRecord for face_cluster algorithms."""
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

        records.append(FaceRecord(
            face_id=i,
            image_id=str(face.original_path),
            image_path=str(face.original_path),
            face_index=face.face_index,
            bbox=bbox_tuple,
            embedding=face.embedding,
            embedding_normalized=embeddings_norm[i],
            blur_score=getattr(face, "blur_score", 0.0),
            area=float(bbox.get("w", 0) * bbox.get("h", 0)) if isinstance(bbox, dict) else 0.0,
            pose=getattr(face, "pose", None),
            det_score=getattr(face, "det_score", None),
        ))
    return records


def build_fc_config(config: dict) -> FCConfig:
    """Build a face_cluster PipelineConfig from the main pipeline's config dict.

    Quality gates are FORCE-DISABLED because the main pipeline doesn't
    compute blur_score or pose on FaceForClustering objects.
    """
    return FCConfig(
        K=config.get("K", 5),
        distance_threshold=config.get("distance_threshold", 0.35),
        min_cluster_size=config.get("min_cluster_size", 2),
        yaw_max=999.0,
        pitch_max=999.0,
        roll_max=999.0,
        blur_min=0.0,
        max_faces_per_image_core=config.get("max_faces_per_image_core", 3),
        det_score_min=None,
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
    face_records = faces_to_face_records(faces, embeddings_norm)
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
