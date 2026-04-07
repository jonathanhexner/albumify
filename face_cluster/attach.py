"""Attach holdout faces to clusters using vote+margin strategy."""

import logging
from typing import List
import numpy as np

from face_cluster.types import FaceRecord, ClusterResult, GraphResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


class HoldoutAttacher:
    """Attach holdout faces to clusters using vote+margin strategy.

    For each holdout face:
    1. Compute distance to each cluster (using exemplars)
    2. Find best and second-best cluster
    3. Check:
       - d_best <= attach_distance_threshold
       - d_best + margin <= d_second (separation margin)
       - Among K_attach nearest core faces, >= vote_min belong to best cluster
    4. If all conditions pass, attach to best cluster; else mark as noise
    """

    def __init__(self, config: PipelineConfig):
        """Initialize holdout attacher.

        Args:
            config: Pipeline configuration with attachment parameters
        """
        self.config = config

    def attach_holdouts(
        self,
        faces: List[FaceRecord],
        core_indices: List[int],
        holdout_indices: List[int],
        cluster_result: ClusterResult,
        graph_result: GraphResult
    ) -> ClusterResult:
        """Attach holdout faces to clusters.

        Args:
            faces: All face records
            core_indices: Indices of core faces
            holdout_indices: Indices of holdout faces
            cluster_result: Cluster result from core clustering
            graph_result: Graph result with distance matrix

        Returns:
            Updated ClusterResult with holdout faces assigned
        """
        if not self.config.attach_enabled:
            logger.info("Holdout attachment disabled")
            return cluster_result

        if len(holdout_indices) == 0:
            logger.info("No holdout faces to attach")
            return cluster_result

        # Build full distance matrix (core + holdout)
        all_indices = core_indices + holdout_indices
        all_embeddings = np.array([faces[i].embedding_normalized for i in all_indices])
        full_distance_matrix = self._compute_distance_matrix(all_embeddings)

        # Map core indices to positions in all_indices
        core_positions = list(range(len(core_indices)))
        holdout_positions = list(range(len(core_indices), len(all_indices)))

        # Get core labels
        core_labels = cluster_result.labels

        # Create extended labels array
        extended_labels = np.full(len(all_indices), -1, dtype=np.int32)
        extended_labels[:len(core_indices)] = core_labels

        n_attached = 0

        for holdout_pos in holdout_positions:
            holdout_idx = all_indices[holdout_pos]
            holdout_face = faces[holdout_idx]

            # Compute distance to each cluster
            cluster_dists = {}
            for cluster_id, cluster_nodes in cluster_result.clusters.items():
                # Use exemplars if available, else use all nodes
                if cluster_id in cluster_result.exemplars and len(cluster_result.exemplars[cluster_id]) > 0:
                    representative_nodes = cluster_result.exemplars[cluster_id]
                else:
                    representative_nodes = cluster_nodes

                # Get distances from holdout to representatives
                dists = [
                    full_distance_matrix[holdout_pos, node]
                    for node in representative_nodes
                ]
                cluster_dists[cluster_id] = min(dists)

            if len(cluster_dists) == 0:
                # No clusters, mark as noise
                continue

            # Find best and second-best cluster
            sorted_clusters = sorted(cluster_dists.items(), key=lambda x: x[1])
            best_cluster_id, d_best = sorted_clusters[0]

            d_second = sorted_clusters[1][1] if len(sorted_clusters) > 1 else np.inf

            # Check distance threshold
            if d_best > self.config.attach_distance_threshold:
                logger.debug(
                    f"Holdout {holdout_face.face_id}: d_best {d_best:.3f} > "
                    f"threshold {self.config.attach_distance_threshold:.3f}"
                )
                continue

            # Check margin
            if d_best + self.config.margin > d_second:
                logger.debug(
                    f"Holdout {holdout_face.face_id}: insufficient margin "
                    f"({d_best:.3f} + {self.config.margin:.3f} > {d_second:.3f})"
                )
                continue

            # Check neighbor voting
            # Get K_attach nearest core faces
            holdout_to_core_dists = full_distance_matrix[holdout_pos, :len(core_indices)]
            nearest_core_indices = np.argsort(holdout_to_core_dists)[:self.config.K_attach]

            # Count votes for best cluster
            votes = 0
            for core_pos in nearest_core_indices:
                if core_labels[core_pos] == best_cluster_id:
                    votes += 1

            if votes < self.config.vote_min:
                logger.debug(
                    f"Holdout {holdout_face.face_id}: insufficient votes "
                    f"({votes} < {self.config.vote_min})"
                )
                continue

            # All conditions passed - attach to cluster
            extended_labels[holdout_pos] = best_cluster_id
            n_attached += 1

            logger.debug(
                f"Attached holdout {holdout_face.face_id} to cluster {best_cluster_id} "
                f"(d={d_best:.3f}, votes={votes})"
            )

        logger.info(
            f"Attached {n_attached}/{len(holdout_indices)} holdout faces to clusters"
        )

        # Update cluster result with extended labels
        # Rebuild clusters dict with new assignments
        new_clusters = {}
        for i, label in enumerate(extended_labels):
            if label >= 0:
                if label not in new_clusters:
                    new_clusters[label] = []
                new_clusters[label].append(all_indices[i])

        # Update cluster stats (sizes changed)
        new_cluster_stats = {}
        for cluster_id, nodes in new_clusters.items():
            new_cluster_stats[cluster_id] = {
                'size': len(nodes),
                # Keep original stats from core clustering
                **cluster_result.cluster_stats.get(cluster_id, {})
            }

        n_noise = int(np.sum(extended_labels == -1))

        return ClusterResult(
            labels=extended_labels,
            clusters=new_clusters,
            cluster_stats=new_cluster_stats,
            exemplars=cluster_result.exemplars,
            n_clusters=cluster_result.n_clusters,
            n_noise=n_noise
        )

    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine distance matrix.

        Args:
            embeddings: n x d array of normalized embeddings

        Returns:
            n x n distance matrix
        """
        similarity = embeddings @ embeddings.T
        distance = np.clip(1.0 - similarity, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        return distance
