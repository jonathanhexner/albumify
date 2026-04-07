"""Cluster analysis and visualization using ClusterSnapshot."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from face_cluster.types import FaceRecord, ClusterResult

logger = logging.getLogger(__name__)


@dataclass
class ClusterSnapshot:
    """Unified snapshot of clustering state for analysis and visualization.

    Create after any clustering stage (initial, merge, split, etc.) to enable
    consistent analysis without managing multiple variables.

    Attributes:
        faces: All face records
        core_indices: Indices into faces that were used for clustering
        labels: Cluster labels for core faces (-1 = noise)
        distance_matrix: Pairwise distance matrix for core faces
        clusters: Dict mapping cluster_id -> list of node indices
        cluster_stats: Dict mapping cluster_id -> statistics
        exemplars: Dict mapping cluster_id -> exemplar node indices
        cluster_thresholds: Per-cluster adaptive thresholds (for merge analysis)
        merge_candidates: Proposed merge pairs with evidence
        split_candidates: Clusters that might need splitting
        stage: Label for this snapshot (e.g., "initial", "after_merge")
        config: Pipeline configuration used
    """
    # Core data
    faces: List[FaceRecord]
    core_indices: List[int]
    labels: np.ndarray
    distance_matrix: np.ndarray

    # Clustering results
    clusters: Dict[int, List[int]]
    cluster_stats: Dict[int, Dict[str, float]]
    exemplars: Dict[int, List[int]] = field(default_factory=dict)

    # Decision metadata (optional, for sensitivity analysis)
    cluster_thresholds: Optional[Dict[int, float]] = None
    merge_candidates: Optional[List[Tuple[int, int, float, Dict]]] = None
    split_candidates: Optional[List[Tuple[int, float]]] = None

    # Metadata
    stage: str = "unknown"
    config: Optional[object] = None

    @classmethod
    def from_result(
        cls,
        cluster_result: ClusterResult,
        faces: List[FaceRecord],
        core_indices: List[int],
        distance_matrix: np.ndarray,
        stage: str = "unknown",
        config: Optional[object] = None,
        cluster_thresholds: Optional[Dict[int, float]] = None,
        merge_candidates: Optional[List[Tuple[int, int, float, Dict]]] = None,
        split_candidates: Optional[List[Tuple[int, float]]] = None,
    ) -> 'ClusterSnapshot':
        """Create snapshot from ClusterResult.

        Args:
            cluster_result: Result from clustering/merge/split stage
            faces: All face records
            core_indices: Indices of faces used for clustering
            distance_matrix: Pairwise distance matrix for core faces
            stage: Label for this snapshot
            config: Pipeline configuration
            cluster_thresholds: Per-cluster adaptive thresholds (from merger)
            merge_candidates: Proposed merge pairs (from merger)
            split_candidates: Clusters that might split (from splitter)

        Returns:
            ClusterSnapshot instance
        """
        return cls(
            faces=faces,
            core_indices=core_indices,
            labels=cluster_result.labels,
            distance_matrix=distance_matrix,
            clusters=cluster_result.clusters,
            cluster_stats=cluster_result.cluster_stats,
            exemplars=cluster_result.exemplars,
            cluster_thresholds=cluster_thresholds,
            merge_candidates=merge_candidates,
            split_candidates=split_candidates,
            stage=stage,
            config=config,
        )

    @property
    def n_clusters(self) -> int:
        """Number of clusters (excluding noise)."""
        return len(self.clusters)

    @property
    def n_noise(self) -> int:
        """Number of noise points."""
        return int((self.labels == -1).sum())

    @property
    def n_core(self) -> int:
        """Number of core faces."""
        return len(self.core_indices)

    @property
    def n_total(self) -> int:
        """Total number of faces."""
        return len(self.faces)

    def get_source_images(self, cluster_id: int) -> Dict[str, int]:
        """Get unique source images for a cluster.

        Args:
            cluster_id: Cluster ID to analyze

        Returns:
            Dict mapping image_id -> face_count, sorted by count descending
        """
        if cluster_id not in self.clusters:
            return {}

        cluster_nodes = self.clusters[cluster_id]
        cluster_face_indices = [self.core_indices[n] for n in cluster_nodes]

        image_counts = Counter(
            self.faces[idx].image_id for idx in cluster_face_indices
        )

        return dict(sorted(image_counts.items(), key=lambda x: x[1], reverse=True))

    def print_cluster_sources(self, cluster_id: int, max_images: int = 10):
        """Print source image breakdown for a cluster.

        Args:
            cluster_id: Cluster ID to analyze
            max_images: Maximum number of images to show
        """
        if cluster_id not in self.clusters:
            logger.warning(f"Cluster {cluster_id} not found")
            return

        sources = self.get_source_images(cluster_id)
        cluster_size = len(self.clusters[cluster_id])

        logger.info(f"Cluster {cluster_id} ({cluster_size} faces from {len(sources)} images):")
        for i, (image_id, count) in enumerate(sources.items()):
            if i >= max_images:
                logger.debug(f"  ... and {len(sources) - max_images} more images")
                break
            logger.debug(f"  {image_id}: {count} face(s)")

    def print_summary(self):
        """Print comprehensive summary of clustering state."""
        logger.info("\n" + "="*60)
        logger.info(f"CLUSTER SNAPSHOT: {self.stage}")
        logger.info("="*60)

        logger.info(f"\nFaces:")
        logger.info(f"  Total: {self.n_total}")
        logger.info(f"  Core: {self.n_core}")
        logger.info(f"  Holdout: {self.n_total - self.n_core}")

        logger.info(f"\nClusters:")
        logger.info(f"  Count: {self.n_clusters}")
        logger.info(f"  Noise: {self.n_noise}")

        if self.n_clusters > 0:
            sizes = [len(nodes) for nodes in self.clusters.values()]
            logger.info(f"  Size range: {min(sizes)} - {max(sizes)} faces")
            logger.info(f"  Mean size: {np.mean(sizes):.1f}")
            logger.info(f"  Median size: {np.median(sizes):.1f}")

        if self.cluster_stats:
            diameters = [stats.get('diameter', 0) for stats in self.cluster_stats.values()]
            if diameters:
                logger.info(f"\nCluster diameters:")
                logger.info(f"  Min: {min(diameters):.3f}")
                logger.info(f"  Median: {np.median(diameters):.3f}")
                logger.info(f"  Max: {max(diameters):.3f}")

        if self.exemplars:
            exemplar_counts = [len(exs) for exs in self.exemplars.values()]
            logger.info(f"\nExemplars:")
            logger.info(f"  Total: {sum(exemplar_counts)}")
            logger.info(f"  Per cluster: {np.mean(exemplar_counts):.1f} avg")

        logger.info("\n" + "="*60)

    def plot_overview(
        self,
        max_clusters: int = 10,
        max_faces_per_cluster: int = 20,
        figsize: Tuple[int, int] = (15, 10),
        show_labels: bool = True
    ):
        """Plot overview of top clusters with face grids.

        Args:
            max_clusters: Maximum number of clusters to show
            max_faces_per_cluster: Maximum faces per cluster
            figsize: Figure size
            show_labels: Whether to show image_id labels
        """
        if self.n_clusters == 0:
            logger.warning("No clusters to display")
            return

        # Sort clusters by size
        sorted_clusters = sorted(
            self.clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:max_clusters]

        n_clusters_shown = len(sorted_clusters)
        fig, axes = plt.subplots(
            n_clusters_shown, 1,
            figsize=(figsize[0], figsize[1] * n_clusters_shown / 10)
        )
        if n_clusters_shown == 1:
            axes = [axes]

        for ax, (cluster_id, cluster_nodes) in zip(axes, sorted_clusters):
            # Get faces for this cluster
            cluster_face_indices = [self.core_indices[n] for n in cluster_nodes]
            cluster_faces = [self.faces[idx] for idx in cluster_face_indices]

            # Limit number of faces shown
            faces_to_show = cluster_faces[:max_faces_per_cluster]

            # Get cluster stats
            stats = self.cluster_stats.get(cluster_id, {})
            diameter = stats.get('diameter', 0)

            # Create grid
            n_faces = len(faces_to_show)
            n_cols = min(10, n_faces)

            # Prepare images
            images = []
            for face in faces_to_show:
                if face.aligned_face is not None:
                    images.append(face.aligned_face)
                else:
                    images.append(np.zeros((112, 112, 3), dtype=np.uint8))

            if not images:
                continue

            # Concatenate images horizontally
            grid = np.concatenate(images[:n_cols], axis=1)

            ax.imshow(grid)
            ax.axis('off')

            # Title with stats
            title = f"Cluster {cluster_id}: {len(cluster_faces)} faces"
            if diameter > 0:
                title += f" (diameter={diameter:.3f})"

            # Add threshold info if available
            if self.cluster_thresholds and cluster_id in self.cluster_thresholds:
                threshold = self.cluster_thresholds[cluster_id]
                title += f" [T={threshold:.3f}]"

            ax.set_title(title, fontsize=10, pad=5)

            # Show image sources
            if show_labels:
                sources = self.get_source_images(cluster_id)
                source_str = ", ".join(f"{img}:{cnt}" for img, cnt in list(sources.items())[:3])
                if len(sources) > 3:
                    source_str += f" +{len(sources)-3} more"
                ax.text(
                    0.5, -0.05, source_str,
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=8, style='italic'
                )

        plt.tight_layout()
        plt.suptitle(f"{self.stage} - Top {n_clusters_shown} Clusters", y=1.0, fontsize=12)
        plt.show()

    def plot_widest_clusters(
        self,
        top_k: int = 3,
        show_histogram: bool = True
    ):
        """Plot distance distribution for widest clusters.

        Args:
            top_k: Number of widest clusters to analyze
            show_histogram: Whether to show distance histograms
        """
        if not self.cluster_stats:
            logger.warning("No cluster statistics available")
            return

        # Find widest clusters
        widest = sorted(
            self.cluster_stats.items(),
            key=lambda x: x[1].get('diameter', 0),
            reverse=True
        )[:top_k]

        if not widest:
            logger.warning("No clusters with statistics")
            return

        logger.info(f"\nTop {len(widest)} widest clusters:")
        for cluster_id, stats in widest:
            diameter = stats.get('diameter', 0)
            size = len(self.clusters[cluster_id])
            logger.debug(f"  Cluster {cluster_id}: diameter={diameter:.3f}, size={size}")

            # Show source images
            sources = self.get_source_images(cluster_id)
            logger.debug(f"    From {len(sources)} images: {list(sources.keys())[:5]}")

        if show_histogram:
            fig, axes = plt.subplots(1, len(widest), figsize=(5*len(widest), 4))
            if len(widest) == 1:
                axes = [axes]

            for ax, (cluster_id, stats) in zip(axes, widest):
                cluster_nodes = self.clusters[cluster_id]

                # Get all pairwise distances within cluster
                distances = []
                for i, node_i in enumerate(cluster_nodes):
                    for node_j in cluster_nodes[i+1:]:
                        dist = self.distance_matrix[node_i, node_j]
                        distances.append(dist)

                if distances:
                    ax.hist(distances, bins=20, edgecolor='black', alpha=0.7)
                    ax.axvline(
                        stats.get('diameter', 0),
                        color='red', linestyle='--',
                        label=f"Diameter={stats.get('diameter', 0):.3f}"
                    )
                    ax.set_xlabel('Distance')
                    ax.set_ylabel('Count')
                    ax.set_title(f"Cluster {cluster_id} ({len(cluster_nodes)} faces)")
                    ax.legend()

            plt.tight_layout()
            plt.show()

    def get_close_clusters_df(self, top_k: int = 10):
        """Get DataFrame of closest cluster pairs.

        Args:
            top_k: Number of pairs to include

        Returns:
            pandas.DataFrame with cluster pairs and distances
        """
        if self.n_clusters < 2:
            return None

        import pandas as pd

        cluster_ids = sorted(self.clusters.keys())
        pairs = []

        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                nodes_c1 = self.clusters[c1]
                nodes_c2 = self.clusters[c2]
                min_dist = min(
                    self.distance_matrix[n1, n2]
                    for n1 in nodes_c1
                    for n2 in nodes_c2
                )
                pairs.append((c1, c2, min_dist))

        pairs.sort(key=lambda x: x[2])

        rows = []
        for c1, c2, min_dist in pairs[:top_k]:
            row = {
                'C1': c1,
                'C2': c2,
                'Size1': len(self.clusters[c1]),
                'Size2': len(self.clusters[c2]),
                'Min_Dist': round(min_dist, 3),
            }

            if self.cluster_thresholds:
                T_A = self.cluster_thresholds.get(c1, 0)
                T_B = self.cluster_thresholds.get(c2, 0)
                alpha = self.config.merge_threshold_alpha if self.config else 0.7
                global_pct = self.config.merge_global_percentile if self.config else 50
                T_global = np.percentile(list(self.cluster_thresholds.values()), global_pct)
                T_merge = alpha * max(T_A, T_B) + (1 - alpha) * T_global
                row['T_merge'] = round(T_merge, 3)
                row['Gap'] = round(min_dist - T_merge, 3)

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_close_clusters(self, top_k: int = 5):
        """Show DataFrame of closest cluster pairs.

        Args:
            top_k: Number of close pairs to show
        """
        df = self.get_close_clusters_df(top_k)
        if df is not None and len(df) > 0:
            logger.info("\nClosest Cluster Pairs:")
            logger.info(df.to_string(index=False))

    def get_cluster_distances(self, sort_by: str = 'Exemplar_Dist'):
        """Get distance matrix between all cluster pairs.

        This is the PRIMARY method for understanding why clusters didn't merge.
        Shows exemplar distances (used for merge proposals) and pairwise statistics.

        Args:
            sort_by: Column to sort by (default: 'Exemplar_Dist')

        Returns:
            pandas.DataFrame with columns:
                - C1, C2: Cluster IDs
                - Size1, Size2: Cluster sizes
                - Exemplar_Dist: Min distance between exemplars (used for merge proposal threshold)
                - Min_Dist: Min distance between any two faces
                - Mean_Dist: Mean distance across all face pairs
                - Max_Dist: Max distance between any two faces
        """
        if self.n_clusters < 2:
            return None

        import pandas as pd

        cluster_ids = sorted(self.clusters.keys())
        rows = []

        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                nodes_c1 = self.clusters[c1]
                nodes_c2 = self.clusters[c2]
                exemplars_c1 = self.exemplars.get(c1, nodes_c1)
                exemplars_c2 = self.exemplars.get(c2, nodes_c2)

                # Exemplar distance (used for merge proposal)
                exemplar_dist = min(
                    self.distance_matrix[e1, e2]
                    for e1 in exemplars_c1
                    for e2 in exemplars_c2
                )

                # All pairwise distances between clusters
                all_dists = [
                    self.distance_matrix[n1, n2]
                    for n1 in nodes_c1
                    for n2 in nodes_c2
                ]

                rows.append({
                    'C1': c1,
                    'C2': c2,
                    'Size1': len(nodes_c1),
                    'Size2': len(nodes_c2),
                    'Exemplar_Dist': round(exemplar_dist, 3),
                    'Min_Dist': round(min(all_dists), 3),
                    'Mean_Dist': round(np.mean(all_dists), 3),
                    'Max_Dist': round(max(all_dists), 3),
                })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values(sort_by)
        return df

    def get_merge_decisions_df(self):
        """Get DataFrame showing merge decisions and thresholds.

        Returns:
            pandas.DataFrame with merge candidates, thresholds, and rejection reasons
        """
        if not self.cluster_thresholds or not self.merge_candidates:
            return None

        import pandas as pd

        global_pct = self.config.merge_global_percentile if self.config else 50
        global_threshold = np.percentile(list(self.cluster_thresholds.values()), global_pct)
        alpha = self.config.merge_threshold_alpha if self.config else 0.7

        rows = []
        for c1, c2, exemplar_dist, evidence in self.merge_candidates[:20]:
            T_A = self.cluster_thresholds.get(c1, 0)
            T_B = self.cluster_thresholds.get(c2, 0)
            T_local = max(T_A, T_B)
            T_merge = alpha * T_local + (1 - alpha) * global_threshold
            size_A = len(self.clusters.get(c1, []))
            size_B = len(self.clusters.get(c2, []))

            valid = evidence.get('valid', False)
            rejection = []
            if not valid:
                if not evidence.get('passes_exemplar', False):
                    rejection.append("Exemplar")
                if not evidence.get('passes_support', False):
                    rejection.append("Support")
                if not evidence.get('passes_margin', False):
                    rejection.append("Margin")
                if not evidence.get('passes_diameter', False):
                    rejection.append("Diameter")

            rows.append({
                'C1': c1,
                'C2': c2,
                'Size1': size_A,
                'Size2': size_B,
                'Exemplar_Dist': round(exemplar_dist, 3),
                'T_A': round(T_A, 3),
                'T_B': round(T_B, 3),
                'T_local': round(T_local, 3),
                'T_global': round(global_threshold, 3),
                'T_merge': round(T_merge, 3),
                'Gap': round(exemplar_dist - T_merge, 3),
                'Merged': valid,
                'Failed': ', '.join(rejection) if rejection else '-'
            })

        return pd.DataFrame(rows)

    def plot_decision_boundaries(self):
        """Show merge decision DataFrame and threshold plots."""
        if not self.cluster_thresholds:
            logger.warning("No cluster thresholds available")
            return

        df = self.get_merge_decisions_df()
        if df is not None and len(df) > 0:
            logger.info("\nMerge Decisions:")
            logger.info(df.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Per-cluster thresholds
        ax = axes[0]
        cluster_ids = sorted(self.cluster_thresholds.keys())
        thresholds = [self.cluster_thresholds[cid] for cid in cluster_ids]
        sizes = [len(self.clusters[cid]) for cid in cluster_ids]

        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_ids)))
        bars = ax.bar(range(len(cluster_ids)), thresholds, color=colors)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Adaptive Threshold')
        ax.set_title('Per-Cluster Merge Thresholds')
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([f"{cid}\n({sz})" for cid, sz in zip(cluster_ids, sizes)], fontsize=8)

        # Add global threshold line if available
        if thresholds:
            global_pct = self.config.merge_global_percentile if self.config else 50
            global_threshold = np.percentile(thresholds, global_pct)
            pct_label = 'median' if global_pct == 50 else f'P{global_pct}'
            ax.axhline(
                global_threshold,
                color='red', linestyle='--',
                label=f'Global ({pct_label})={global_threshold:.3f}'
            )
            ax.legend()

        # Plot 2: Merge candidates vs threshold
        ax = axes[1]
        if self.merge_candidates:
            candidate_dists = [mc[2] for mc in self.merge_candidates[:20]]
            candidate_labels = [f"{mc[0]}-{mc[1]}" for mc in self.merge_candidates[:20]]

            ax.barh(range(len(candidate_dists)), candidate_dists, color='lightblue')
            ax.set_yticks(range(len(candidate_dists)))
            ax.set_yticklabels(candidate_labels, fontsize=8)
            ax.set_xlabel('Exemplar Distance')
            ax.set_title('Merge Candidates (Exemplar Distance)')
            ax.invert_yaxis()

            # Add global threshold line
            if thresholds:
                global_pct = self.config.merge_global_percentile if self.config else 50
                global_threshold = np.percentile(thresholds, global_pct)
                pct_label = 'median' if global_pct == 50 else f'P{global_pct}'
                ax.axvline(
                    global_threshold,
                    color='red', linestyle='--',
                    label=f'Global ({pct_label})={global_threshold:.3f}'
                )
                ax.legend()
        else:
            ax.text(0.5, 0.5, "No merge candidates", ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Merge Candidates')

        plt.tight_layout()
        plt.show()

    def compare_with(self, other: 'ClusterSnapshot', show_diff: bool = True):
        """Compare two snapshots (e.g., before/after merge).

        Args:
            other: Another ClusterSnapshot to compare with
            show_diff: Whether to print detailed differences
        """
        logger.info("\n" + "="*60)
        logger.info(f"COMPARISON: {self.stage} vs {other.stage}")
        logger.info("="*60)

        logger.info(f"\nClusters:")
        logger.info(f"  {self.stage}: {self.n_clusters} clusters, {self.n_noise} noise")
        logger.info(f"  {other.stage}: {other.n_clusters} clusters, {other.n_noise} noise")
        logger.info(f"  Change: {other.n_clusters - self.n_clusters:+d} clusters, {other.n_noise - self.n_noise:+d} noise")

        if show_diff and self.n_clusters > 0 and other.n_clusters > 0:
            # Find which clusters merged
            logger.info(f"\nCluster size distribution:")

            self_sizes = sorted([len(nodes) for nodes in self.clusters.values()], reverse=True)
            other_sizes = sorted([len(nodes) for nodes in other.clusters.values()], reverse=True)

            logger.info(f"  {self.stage}: {self_sizes[:10]}")
            logger.info(f"  {other.stage}: {other_sizes[:10]}")

        logger.info("\n" + "="*60)
