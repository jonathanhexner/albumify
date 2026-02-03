"""Cluster scenes step - group similar images using HDBSCAN."""

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step


@register_step
class ClusterScenesStep(BaseStep):
    """Cluster images by scene similarity using HDBSCAN."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="cluster_scenes",
            display_name="Cluster Scenes",
            description="Group visually similar images into clusters using HDBSCAN on scene embeddings.",
            category="clustering",
            requires={"scene_embeddings"},
            produces={"scene_clusters", "scene_cluster_labels"},
            depends_on=["extract_scene_embedding"],
            config_schema={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["hdbscan", "dbscan", "kmeans"],
                        "default": "hdbscan",
                        "description": "Clustering algorithm"
                    },
                    "min_cluster_size": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 2,
                        "description": "Minimum images per cluster"
                    }
                }
            }
        )
        self._clusterer = None

    def _get_clusterer(self, config: dict):
        from sim_bench.clustering.base import load_clustering_method
        clustering_config = {
            "algorithm": config.get("method", config.get("algorithm", "hdbscan")),
            "params": {
                "min_cluster_size": config.get("min_cluster_size", 2),
                "min_samples": config.get("min_samples", 2),
                "metric": config.get("metric", "cosine"),
                "cluster_selection_epsilon": config.get("cluster_selection_epsilon", 0.0),
                "cluster_selection_method": config.get("cluster_selection_method", "eom"),
            },
            "output": {}
        }
        return load_clustering_method(clustering_config)

    def process(self, context: PipelineContext, config: dict) -> None:
        if not context.scene_embeddings:
            context.report_progress("cluster_scenes", 1.0, "No embeddings to cluster")
            return

        image_paths = list(context.scene_embeddings.keys())
        features = np.array([context.scene_embeddings[p] for p in image_paths])

        context.report_progress("cluster_scenes", 0.2, f"Clustering {len(image_paths)} images")

        clusterer = self._get_clusterer(config)
        labels, stats = clusterer.cluster(features)

        clusters: dict[int, list[str]] = {}
        for path, label in zip(image_paths, labels):
            label_int = int(label)
            context.scene_cluster_labels[path] = label_int
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(path)

        context.scene_clusters = clusters

        num_clusters = len([k for k in clusters.keys() if k >= 0])
        noise_count = len(clusters.get(-1, []))

        context.report_progress(
            "cluster_scenes",
            1.0,
            f"Created {num_clusters} clusters ({noise_count} noise images)"
        )
