"""ClusteringRunner - Execute clustering using sim_bench algorithms."""

import logging
from typing import Dict, List, Any

import numpy as np

from sim_bench.clustering.base import load_clustering_method
from app.face_clustering_debug.models.schemas import (
    ClusterInfo,
    ClusteringRequest,
    MergeDecision,
    AttachDecision,
    ClusteringResult,
)

logger = logging.getLogger(__name__)


# Parameter definitions for each algorithm (used to generate UI sliders)
# All threshold defaults are in cosine distance space: distance = 1 - cosine_similarity
# Converted from Euclidean using: t_c = (t_e²) / 2
ALGORITHM_PARAMS = {
    "hdbscan": {
        "pca_dim": {"type": "int", "min": 32, "max": 512, "default": None, "optional": True},
        "min_cluster_size": {"type": "int", "min": 2, "max": 20, "default": 5},
        "min_samples": {"type": "int", "min": 1, "max": 10, "default": 2},
        "cluster_selection_epsilon": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.0},
    },
    "hybrid_hdbscan_knn": {
        "pca_dim": {"type": "int", "min": 32, "max": 512, "default": None, "optional": True},
        "min_cluster_size": {"type": "int", "min": 2, "max": 10, "default": 2},
        "min_samples": {"type": "int", "min": 1, "max": 5, "default": 2},
        "cluster_selection_epsilon": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.045},
        "knn_k": {"type": "int", "min": 1, "max": 10, "default": 3},
        "threshold_percentile": {"type": "int", "min": 50, "max": 99, "default": 90},
        "threshold_floor": {"type": "float", "min": 0.02, "max": 0.3, "default": 0.125},
        "threshold_ceiling": {"type": "float", "min": 0.2, "max": 0.6, "default": 0.405},
        "max_exemplars": {"type": "int", "min": 3, "max": 20, "default": 10},
        "merge_min_pairs": {"type": "int", "min": 1, "max": 10, "default": 3},
        "merge_min_distinct": {"type": "int", "min": 1, "max": 5, "default": 2},
        "attach_min_exemplars": {"type": "int", "min": 1, "max": 5, "default": 2},
        "max_iterations": {"type": "int", "min": 1, "max": 20, "default": 10},
    },
    "hybrid_closest_face": {
        "pca_dim": {"type": "int", "min": 32, "max": 512, "default": None, "optional": True},
        "min_cluster_size": {"type": "int", "min": 2, "max": 10, "default": 2},
        "min_samples": {"type": "int", "min": 1, "max": 5, "default": 2},
        "cluster_selection_epsilon": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.045},
        "knn_k": {"type": "int", "min": 1, "max": 10, "default": 3},
        "threshold_percentile": {"type": "int", "min": 50, "max": 99, "default": 90},
        "threshold_floor": {"type": "float", "min": 0.02, "max": 0.2, "default": 0.045},
        "threshold_ceiling": {"type": "float", "min": 0.2, "max": 0.6, "default": 0.405},
        "max_exemplars": {"type": "int", "min": 3, "max": 20, "default": 10},
        "merge_min_faces": {"type": "int", "min": 1, "max": 10, "default": 2},
        "early_exit_multiplier": {"type": "float", "min": 1.0, "max": 5.0, "default": 2.0},
        "merge_threshold_multiplier": {"type": "float", "min": 1.0, "max": 2.0, "default": 1.5},
        "attach_min_neighbors": {"type": "int", "min": 1, "max": 5, "default": 1},
    },
    # Variants of hybrid_hdbscan_knn with different threshold/merge/attach strategies
    "hybrid_hdbscan_knn_Tcore2all": {
        "pca_dim": {"type": "int", "min": 32, "max": 512, "default": None, "optional": True},
        "min_cluster_size": {"type": "int", "min": 2, "max": 10, "default": 2},
        "min_samples": {"type": "int", "min": 1, "max": 5, "default": 2},
        "cluster_selection_epsilon": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.045},
        "knn_k": {"type": "int", "min": 1, "max": 10, "default": 3},
        "threshold_percentile": {"type": "int", "min": 50, "max": 99, "default": 95},  # Higher default
        "threshold_floor": {"type": "float", "min": 0.02, "max": 0.3, "default": 0.125},
        "threshold_ceiling": {"type": "float", "min": 0.2, "max": 0.6, "default": 0.405},
        "max_exemplars": {"type": "int", "min": 3, "max": 20, "default": 10},
        "merge_min_pairs": {"type": "int", "min": 1, "max": 10, "default": 3},
        "merge_min_distinct": {"type": "int", "min": 1, "max": 5, "default": 2},
        "attach_min_exemplars": {"type": "int", "min": 1, "max": 5, "default": 2},
        "max_iterations": {"type": "int", "min": 1, "max": 20, "default": 10},
    },
    "hybrid_hdbscan_knn_merge_twotier": {
        "pca_dim": {"type": "int", "min": 32, "max": 512, "default": None, "optional": True},
        "min_cluster_size": {"type": "int", "min": 2, "max": 10, "default": 2},
        "min_samples": {"type": "int", "min": 1, "max": 5, "default": 2},
        "cluster_selection_epsilon": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.045},
        "knn_k": {"type": "int", "min": 1, "max": 10, "default": 3},
        "threshold_percentile": {"type": "int", "min": 50, "max": 99, "default": 90},
        "threshold_floor": {"type": "float", "min": 0.02, "max": 0.3, "default": 0.125},
        "threshold_ceiling": {"type": "float", "min": 0.2, "max": 0.6, "default": 0.405},
        "max_exemplars": {"type": "int", "min": 3, "max": 20, "default": 10},
        "merge_min_pairs": {"type": "int", "min": 1, "max": 10, "default": 3},
        "merge_min_distinct": {"type": "int", "min": 1, "max": 5, "default": 2},
        "merge_secondary_min_pairs": {"type": "int", "min": 3, "max": 10, "default": 5},  # New param
        "attach_min_exemplars": {"type": "int", "min": 1, "max": 5, "default": 2},
        "max_iterations": {"type": "int", "min": 1, "max": 20, "default": 10},
    },
    "hybrid_hdbscan_knn_attach_strong1": {
        "pca_dim": {"type": "int", "min": 32, "max": 512, "default": None, "optional": True},
        "min_cluster_size": {"type": "int", "min": 2, "max": 10, "default": 2},
        "min_samples": {"type": "int", "min": 1, "max": 5, "default": 2},
        "cluster_selection_epsilon": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.045},
        "knn_k": {"type": "int", "min": 1, "max": 10, "default": 3},
        "threshold_percentile": {"type": "int", "min": 50, "max": 99, "default": 90},
        "threshold_floor": {"type": "float", "min": 0.02, "max": 0.3, "default": 0.125},
        "threshold_ceiling": {"type": "float", "min": 0.2, "max": 0.6, "default": 0.405},
        "max_exemplars": {"type": "int", "min": 3, "max": 20, "default": 10},
        "merge_min_pairs": {"type": "int", "min": 1, "max": 10, "default": 3},
        "merge_min_distinct": {"type": "int", "min": 1, "max": 5, "default": 2},
        "attach_min_exemplars": {"type": "int", "min": 1, "max": 5, "default": 2},
        "attach_strong1_multiplier": {"type": "float", "min": 0.5, "max": 1.0, "default": 0.8},  # New param
        "max_iterations": {"type": "int", "min": 1, "max": 20, "default": 10},
    },
}


class ClusteringRunner:
    """Execute clustering using sim_bench algorithms."""

    @staticmethod
    def get_available_algorithms() -> List[str]:
        """Return list of available clustering algorithms."""
        return list(ALGORITHM_PARAMS.keys())

    @staticmethod
    def get_algorithm_params(algorithm: str) -> Dict[str, Dict[str, Any]]:
        """Return parameter definitions for the given algorithm."""
        return ALGORITHM_PARAMS.get(algorithm, {})

    @staticmethod
    def run(request: ClusteringRequest) -> ClusteringResult:
        """Run clustering and return structured result.

        Args:
            request: ClusteringRequest with algorithm, params, embeddings, faces

        Returns:
            ClusteringResult with labels, clusters, and debug data
        """
        config = {"algorithm": request.algorithm, "params": request.params}

        logger.info(f"Running clustering: algorithm={request.algorithm}, n_faces={len(request.embeddings)}")
        clusterer = load_clustering_method(config)
        labels, stats = clusterer.cluster(request.embeddings, collect_debug_data=request.collect_debug_data)

        clusters = ClusteringRunner._parse_clusters(labels, stats)
        merge_decisions = ClusteringRunner._parse_merge_decisions(stats)
        attach_decisions = ClusteringRunner._parse_attach_decisions(stats)

        return ClusteringResult(
            labels=labels,
            embeddings=request.embeddings,
            faces=request.faces,
            clusters=clusters,
            merge_decisions=merge_decisions,
            attach_decisions=attach_decisions,
            algorithm=request.algorithm,
            params=request.params,
            n_clusters=stats.get("n_clusters", len(clusters)),
            n_noise=stats.get("n_noise", int(np.sum(labels == -1))),
        )

    @staticmethod
    def _parse_clusters(labels: np.ndarray, stats: Dict[str, Any]) -> List[ClusterInfo]:
        """Parse cluster info from stats dict. Handles both int and str keys."""
        debug = stats.get("debug", {})
        thresholds = debug.get("cluster_thresholds", {})
        exemplars = debug.get("cluster_exemplars", {})
        d3_stats = debug.get("cluster_d3_stats", {})

        clusters = []
        for label in sorted(set(labels) - {-1}):
            key = int(label)
            face_indices = [i for i, l in enumerate(labels) if l == label]
            d3 = d3_stats.get(key, d3_stats.get(str(key), {}))

            clusters.append(ClusterInfo(
                cluster_id=key,
                face_indices=face_indices,
                threshold=thresholds.get(key, thresholds.get(str(key), 0.0)),
                exemplar_indices=exemplars.get(key, exemplars.get(str(key), [])),
                q1=d3.get("q1", 0.0),
                q3=d3.get("q3", 0.0),
                iqr=d3.get("iqr", 0.0),
                raw_threshold=d3.get("raw_threshold", 0.0),
            ))

        return clusters

    @staticmethod
    def _parse_merge_decisions(stats: Dict[str, Any]) -> List[MergeDecision]:
        """Parse merge decisions from stats dict.

        Handles both hybrid_hdbscan_knn (exemplar-based) and hybrid_closest_face (face-based).
        """
        debug = stats.get("debug", {})
        decisions = []

        for d in debug.get("merge_decisions", []):
            decisions.append(MergeDecision(
                cluster_a=d.get("cluster_a", 0),
                cluster_b=d.get("cluster_b", 0),
                threshold_a=d.get("threshold_a", 0.0),
                threshold_b=d.get("threshold_b", 0.0),
                merged=d.get("merged", False),
                reason=d.get("reason", ""),
                # Common
                min_distance=d.get("min_distance", d.get("min_exemplar_dist", 0.0)),
                cross_distances=d.get("cross_distances", d.get("exemplar_cross_distances")),
                # hybrid_hdbscan_knn specific
                threshold_used=d.get("threshold", 0.0),
                pairs_within_threshold=d.get("n_pairs_within", 0),
                exemplars_a_involved=d.get("exemplars_a_involved", 0),
                exemplars_b_involved=d.get("exemplars_b_involved", 0),
                min_dists_a=d.get("min_dists_a", d.get("exemplar_min_dists_a")),
                min_dists_b=d.get("min_dists_b", d.get("exemplar_min_dists_b")),
                # hybrid_closest_face specific
                d3_cross_a=d.get("d3_cross_a"),
                d3_cross_b=d.get("d3_cross_b"),
                fits_a=d.get("fits_a", 0),
                fits_b=d.get("fits_b", 0),
                n_fits_total=d.get("n_fits_total", 0),
                effective_threshold_a=d.get("effective_threshold_a", 0.0),
                effective_threshold_b=d.get("effective_threshold_b", 0.0),
                merge_threshold_multiplier=d.get("merge_threshold_multiplier", 1.0),
                merge_min_faces=d.get("merge_min_faces", 2),
                early_exit_threshold=d.get("early_exit_threshold", 0.0),
            ))

        return decisions

    @staticmethod
    def _parse_attach_decisions(stats: Dict[str, Any]) -> List[AttachDecision]:
        """Parse attachment decisions from stats dict."""
        debug = stats.get("debug", {})
        decisions = []

        for d in debug.get("attach_decisions", []):
            decisions.append(AttachDecision(
                face_index=d.get("face_idx", 0),
                attached_to=d.get("attached_to"),
                reason=d.get("reason", ""),
                candidates=d.get("candidates", []),
            ))

        return decisions
