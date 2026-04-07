"""Cluster Connected Components step - Cluster faces via graph connected components.

Finds connected components in the mutual k-NN graph to form initial clusters.
"""

import logging
import time
from typing import Dict, Any

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from face_cluster.clustering import ConnectedComponentsClusterer
from face_cluster.config import PipelineConfig as FaceClusterConfig

logger = logging.getLogger(__name__)


@register_step
class ClusterConnectedComponentsStep(BaseStep):
    """Cluster faces using connected components on mutual k-NN graph.

    Forms initial clusters by finding connected components in the graph.
    Small components (< min_cluster_size) are marked as noise.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="cluster_connected_components",
            display_name="Cluster Connected Components",
            description="Cluster faces using connected components on kNN graph",
            category="people",
            requires={"knn_graph_result", "core_indices"},
            produces={"initial_clusters"},
            depends_on=["build_knn_graph"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_cluster_size": {
                        "type": "integer",
                        "default": 2,
                        "description": "Minimum cluster size (smaller = noise)"
                    },
                    "split_enabled": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable cluster splitting for wide clusters (Phase 2)"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Cluster core faces using connected components.

        Args:
            context: Pipeline context with knn_graph_result
            config: Step configuration with min_cluster_size
        """
        start_time = time.time()

        graph_result = context.knn_graph_result
        core_indices = context.core_indices

        if graph_result is None or not core_indices:
            logger.warning("No graph result or core faces to cluster")
            context.initial_clusters = None
            context.report_progress("cluster_connected_components", 1.0, "No faces to cluster")
            return

        logger.info(f"Clustering {len(core_indices)} faces via connected components")

        # Create face_cluster config
        fc_config = FaceClusterConfig(
            min_cluster_size=config.get('min_cluster_size', 2),
            split_enabled=config.get('split_enabled', False)
        )

        # Run clustering
        context.report_progress("cluster_connected_components", 0.5, "Finding connected components")

        clusterer = ConnectedComponentsClusterer(fc_config)
        cluster_result = clusterer.cluster(graph_result, core_indices)

        # Store result in context
        context.initial_clusters = cluster_result

        duration = time.time() - start_time

        # Log with timing
        logger.info("=" * 60)
        logger.info("CLUSTER_CONNECTED_COMPONENTS: Stage completed")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Total faces: {len(core_indices)}")
        logger.info(f"Clusters: {cluster_result.n_clusters}")
        logger.info(f"Noise points: {cluster_result.n_noise}")

        if cluster_result.n_clusters > 0:
            cluster_sizes = [len(nodes) for nodes in cluster_result.clusters.values()]
            logger.info(f"Cluster sizes: min={min(cluster_sizes)}, "
                       f"max={max(cluster_sizes)}, "
                       f"mean={sum(cluster_sizes)/len(cluster_sizes):.1f}")

            # Log cluster statistics
            for cluster_id, stats in cluster_result.cluster_stats.items():
                logger.info(f"Cluster {cluster_id}: size={stats['size']}, "
                           f"diameter={stats['diameter']:.3f}, "
                           f"median_dist={stats['median_dist']:.3f}")

        logger.info("=" * 60)

        context.report_progress(
            "cluster_connected_components", 1.0,
            f"Clustered: {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise"
        )
