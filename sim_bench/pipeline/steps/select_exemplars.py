"""Select Exemplars step - Select representative faces for each cluster.

Uses d10-based density metric to select high-quality representative faces
(exemplars) for each cluster. These are used for cluster comparison and merging.
"""

import logging
import time
from typing import Dict, Any

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from face_cluster.exemplars import D10ExemplarSelector
from face_cluster.config import PipelineConfig as FaceClusterConfig

logger = logging.getLogger(__name__)


@register_step
class SelectExemplarsStep(BaseStep):
    """Select exemplar faces for each cluster using d10 density metric.

    Exemplars are high-density faces that represent the cluster well.
    Used for cluster-to-cluster distance computation and merge decisions.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="select_exemplars",
            display_name="Select Exemplars",
            description="Select representative faces (exemplars) for each cluster",
            category="people",
            requires={"initial_clusters", "knn_graph_result"},
            produces={"initial_clusters"},  # Updates initial_clusters with exemplars
            depends_on=["cluster_connected_components"],
            config_schema={
                "type": "object",
                "properties": {
                    "d10_k": {
                        "type": "integer",
                        "default": 10,
                        "description": "K for d10 density metric (distance to Kth neighbor)"
                    },
                    "exemplars_d10_threshold": {
                        "type": "number",
                        "default": 0.25,
                        "description": "Maximum d10 value for exemplar candidates"
                    },
                    "exemplar_suppression_radius": {
                        "type": "number",
                        "default": 0.15,
                        "description": "Minimum distance between selected exemplars"
                    },
                    "N_exemplars_max": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of exemplars per cluster"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Select exemplars for each cluster.

        Args:
            context: Pipeline context with initial_clusters and knn_graph_result
            config: Step configuration with d10 parameters
        """
        start_time = time.time()

        cluster_result = context.initial_clusters
        graph_result = context.knn_graph_result

        if cluster_result is None or graph_result is None:
            logger.warning("No cluster result or graph result to select exemplars from")
            context.report_progress("select_exemplars", 1.0, "No clusters")
            return

        if cluster_result.n_clusters == 0:
            logger.warning("No clusters to select exemplars from")
            context.report_progress("select_exemplars", 1.0, "No clusters")
            return

        logger.info(f"Selecting exemplars for {cluster_result.n_clusters} clusters")

        # Create face_cluster config
        fc_config = FaceClusterConfig(
            d10_k=config.get('d10_k', 10),
            exemplars_d10_threshold=config.get('exemplars_d10_threshold', 0.25),
            exemplar_suppression_radius=config.get('exemplar_suppression_radius', 0.15),
            N_exemplars_max=config.get('N_exemplars_max', 5)
        )

        # Select exemplars
        context.report_progress("select_exemplars", 0.5, "Selecting exemplars")

        selector = D10ExemplarSelector(fc_config)
        cluster_result, _ = selector.select_exemplars(cluster_result, graph_result)

        # Update context (cluster_result is modified in-place, but reassign for clarity)
        context.initial_clusters = cluster_result

        duration = time.time() - start_time

        # Log with timing
        logger.info("=" * 60)
        logger.info("SELECT_EXEMPLARS: Stage completed")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Clusters: {cluster_result.n_clusters}")

        total_exemplars = sum(len(exs) for exs in cluster_result.exemplars.values())
        avg_exemplars = total_exemplars / cluster_result.n_clusters if cluster_result.n_clusters > 0 else 0

        logger.info(f"Total exemplars: {total_exemplars}")
        logger.info(f"Avg exemplars per cluster: {avg_exemplars:.1f}")

        for cluster_id, exemplar_indices in cluster_result.exemplars.items():
            logger.info(f"Cluster {cluster_id}: {len(exemplar_indices)} exemplars")

        logger.info("=" * 60)

        context.report_progress(
            "select_exemplars", 1.0,
            f"Selected {total_exemplars} exemplars for {cluster_result.n_clusters} clusters"
        )
