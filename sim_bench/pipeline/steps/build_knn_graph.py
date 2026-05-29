"""Build KNN Graph step - Build mutual k-NN graph for face clustering.

Constructs a mutual k-NN graph from face embeddings, where edges are created
only if two faces are mutual nearest neighbors and their distance is below threshold.
"""

import logging
import time
from typing import Dict, Any

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from face_cluster.knn_graph import KNNGraphBuilder
from face_cluster.config import PipelineConfig as FaceClusterConfig

logger = logging.getLogger(__name__)


@register_step
class BuildKNNGraphStep(BaseStep):
    """Build mutual k-NN graph for face clustering.

    Uses mutual k-NN + distance threshold to create edges between similar faces.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="build_knn_graph",
            display_name="Build KNN Graph",
            description="Build mutual k-NN graph with distance threshold for clustering",
            category="people",
            requires={"face_records", "core_indices"},
            produces={"knn_graph_result"},
            depends_on=["quality_gate"],
            config_schema={
                "type": "object",
                "properties": {
                    "K": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of nearest neighbors for kNN graph"
                    },
                    "distance_threshold": {
                        "type": "number",
                        "default": 0.35,
                        "description": "Maximum distance for creating edges (cosine distance)"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Build mutual k-NN graph from core face embeddings.

        Args:
            context: Pipeline context with face_records and core_indices
            config: Step configuration with K and distance_threshold
        """
        start_time = time.time()

        face_records = context.face_records
        core_indices = context.core_indices

        if not core_indices:
            logger.warning("No core faces to build graph from")
            context.knn_graph_result = None
            context.report_progress("build_knn_graph", 1.0, "No core faces")
            return

        # Validation: Check embeddings are normalized
        embeddings = np.array([face_records[i].embedding for i in core_indices])
        norms = np.linalg.norm(embeddings, axis=1)
        assert all(0.9 < norm < 1.1 for norm in norms), \
            f"Embeddings not normalized - check embedding model (norms: {norms[:5]})"

        logger.info(f"Building kNN graph for {len(core_indices)} core faces")

        # Create face_cluster config
        K = config.get('K', 5)
        distance_threshold = config.get('distance_threshold', 0.35)

        fc_config = FaceClusterConfig(
            K=K,
            distance_threshold=distance_threshold
        )

        # Build graph
        context.report_progress("build_knn_graph", 0.5, f"Building kNN graph (K={K})")

        builder = KNNGraphBuilder(fc_config)
        graph_result = builder.build_graph(face_records, core_indices)

        # Store result in context
        context.knn_graph_result = graph_result

        duration = time.time() - start_time

        # Log with timing
        logger.info("=" * 60)
        logger.info("BUILD_KNN_GRAPH: Stage completed")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Nodes: {len(core_indices)}")
        logger.info(f"Edges: {len(graph_result.edges)}")
        logger.info(f"K: {K}")
        logger.info(f"Distance threshold: {distance_threshold:.3f}")
        if len(graph_result.edges) > 0:
            edge_dists = [e[2] for e in graph_result.edges]
            logger.info(f"Edge distances: min={min(edge_dists):.3f}, "
                       f"max={max(edge_dists):.3f}, "
                       f"median={np.median(edge_dists):.3f}")
        logger.info("=" * 60)

        context.report_progress(
            "build_knn_graph", 1.0,
            f"Built graph: {len(core_indices)} nodes, {len(graph_result.edges)} edges"
        )
