"""Cluster People step — group faces by identity.

Delegates to face_cluster_bridge.py for the face_cluster_knn algorithm
and face_cluster_export.py for standalone app export.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.clustering_labels import NOISE_LABEL, is_noise
from sim_bench.pipeline.context import PipelineContext, StepDecision
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.steps.configs._validate import validate_step_config
from sim_bench.pipeline.steps.face_cluster_bridge import run_face_cluster_knn
from sim_bench.pipeline.steps.face_cluster_export import export_for_analysis

logger = logging.getLogger(__name__)


@dataclass
class FaceForClustering:
    """Lightweight face object for clustering."""
    original_path: Path
    face_index: int
    embedding: np.ndarray
    bbox: Dict[str, Any] = field(default_factory=dict)
    cluster_id: int = -1


@register_step
class ClusterPeopleStep(BaseStep):
    """Cluster faces by identity using face embeddings."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="cluster_people",
            display_name="Cluster People",
            description="Group faces by identity using agglomerative clustering on face embeddings.",
            category="people",
            requires={"face_embeddings"},
            produces={"people_clusters"},
            depends_on=["extract_face_embeddings"],
            config_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["hdbscan", "hdbscan_pca", "hybrid_hdbscan_knn", "mutual_knn", "agglomerative", "face_cluster_knn"],
                        "default": "hybrid_hdbscan_knn",
                        "description": "Clustering algorithm"
                    },
                    "distance_threshold": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Distance threshold for agglomerative clustering"
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Fixed number of clusters (optional)"
                    }
                }
            }
        )

    def _collect_faces_with_embeddings(self, context: PipelineContext) -> List[FaceForClustering]:
        """Collect all faces with embeddings from context."""
        faces_with_embeddings = []

        logger.info(f"Collecting faces: MediaPipe faces={len(context.faces)}, "
                   f"InsightFace faces={len(context.insightface_faces) if hasattr(context, 'insightface_faces') else 0}, "
                   f"face_embeddings={len(context.face_embeddings)}")

        if context.face_embeddings:
            sample_keys = list(context.face_embeddings.keys())[:3]
            logger.info(f"Sample embedding keys: {sample_keys}")

        # MediaPipe faces
        if context.faces:
            for img_path, faces in context.faces.items():
                for face in faces:
                    path_str = str(face.original_path).replace('\\', '/')
                    cache_key = f"{path_str}:face_{face.face_index}"
                    embedding = context.face_embeddings.get(cache_key)
                    if embedding is not None:
                        face.embedding = embedding
                        faces_with_embeddings.append(face)
                    elif hasattr(face, 'embedding') and face.embedding is not None:
                        faces_with_embeddings.append(face)

        # InsightFace faces
        if hasattr(context, 'insightface_faces') and context.insightface_faces:
            for img_path, face_data in context.insightface_faces.items():
                path_str = str(img_path).replace('\\', '/')
                for face_info in face_data.get('faces', []):
                    face_index = face_info.get('face_index', 0)
                    cache_key = f"{path_str}:face_{face_index}"
                    embedding = context.face_embeddings.get(cache_key)
                    if embedding is not None:
                        face = FaceForClustering(
                            original_path=Path(img_path),
                            face_index=face_index,
                            embedding=embedding,
                            bbox=face_info.get('bbox', {})
                        )
                        faces_with_embeddings.append(face)

        logger.info(f"Total faces with embeddings: {len(faces_with_embeddings)}")
        return faces_with_embeddings

    def process(self, context: PipelineContext, config: dict) -> None:
        """Cluster faces by identity."""
        # spec-033 P-G: typo'd key raises ValidationError.
        validate_step_config("cluster_people", config)
        faces_with_embeddings = self._collect_faces_with_embeddings(context)

        if not faces_with_embeddings:
            context.report_progress("cluster_people", 1.0, "No faces with embeddings to cluster")
            return

        logger.info(f"Clustering {len(faces_with_embeddings)} faces with embeddings")

        if len(faces_with_embeddings) == 1:
            faces_with_embeddings[0].cluster_id = 0
            context.people_clusters = {0: [faces_with_embeddings[0]]}
            context.report_progress("cluster_people", 1.0, "Single face detected (1 person)")
            return

        embeddings = np.array([f.embedding for f in faces_with_embeddings])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_normalized = embeddings / norms

        method = config.get("method", "hdbscan")
        context.report_progress("cluster_people", 0.5, f"Running {method} clustering")

        labels = self._dispatch_clustering(method, config, embeddings, embeddings_normalized,
                                           faces_with_embeddings, context)

        # Assign cluster IDs and build people_clusters
        for face, label in zip(faces_with_embeddings, labels):
            face.cluster_id = int(label)

        clusters = {}
        noise_count = 0
        for face in faces_with_embeddings:
            if is_noise(face.cluster_id):
                noise_count += 1
                continue
            if face.cluster_id not in clusters:
                clusters[face.cluster_id] = []
            clusters[face.cluster_id].append(face)

        if noise_count:
            logger.info(f"Excluded {noise_count} noise faces (label={NOISE_LABEL}) from people clusters")

        # Emit per-face StepDecision records
        cfg = {"method": method}
        for face in faces_with_embeddings:
            face_key = f"{face.original_path}:face_{face.face_index}"
            if is_noise(face.cluster_id):
                decision, reason = "noise", "Not assigned to any person (noise)"
            else:
                cluster_size = len(clusters.get(face.cluster_id, []))
                decision = f"person_{face.cluster_id}"
                reason = f"Assigned to person cluster {face.cluster_id} ({cluster_size} faces)"
            context.step_decisions.append(StepDecision(
                item_id=face_key, item_type="face", step="cluster_people",
                decision=decision, reason=reason, config_used=cfg,
                metrics={"cluster_id": face.cluster_id},
            ))

        context.people_clusters = clusters
        num_people = len(clusters)
        clustered_faces = len(faces_with_embeddings) - noise_count
        avg = clustered_faces / num_people if num_people > 0 else 0

        context.report_progress(
            "cluster_people", 1.0,
            f"Clustered {len(faces_with_embeddings)} faces into {num_people} identities (avg {avg:.1f} faces/person)"
        )

    def _dispatch_clustering(self, method, config, embeddings, embeddings_normalized,
                              faces_with_embeddings, context):
        """Dispatch to the appropriate clustering algorithm."""
        if method == "hdbscan":
            return self._run_hdbscan(config, embeddings_normalized)
        elif method == "hdbscan_pca":
            return self._run_hdbscan_pca(config, embeddings)
        elif method == "hybrid_hdbscan_knn":
            return self._run_hybrid(config, embeddings)
        elif method == "mutual_knn":
            return self._run_mutual_knn(config, embeddings)
        elif method == "agglomerative":
            return self._run_agglomerative(config, embeddings)
        elif method == "face_cluster_knn":
            return self._run_face_cluster_knn(faces_with_embeddings, embeddings_normalized, config, context)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    @staticmethod
    def _run_hdbscan(config, embeddings_normalized):
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.get("min_cluster_size", 2),
            min_samples=config.get("min_samples", 2),
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=config.get("cluster_selection_epsilon", 0.3),
        )
        labels = clusterer.fit_predict(embeddings_normalized)
        logger.info(f"HDBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels

    @staticmethod
    def _run_hdbscan_pca(config, embeddings):
        # SIGHTING-120: PCA+HDBSCAN is the `hdbscan` factory algorithm with a `pca_dim`
        # param — the standalone `hdbscan_pca` algorithm was retired as a duplicate.
        # The two are behaviour-identical (verified: exact-match labels, ARI=1.000);
        # we just map pca_components -> pca_dim.
        from sim_bench.clustering.base import load_clustering_method
        clustering_config = {
            'algorithm': 'hdbscan',
            'params': {
                'pca_dim': config.get('pca_components', 128),
                'metric': 'cosine',
                'min_cluster_size': config.get('min_cluster_size', 2),
                'min_samples': config.get('min_samples', 2),
                'cluster_selection_epsilon': config.get('cluster_selection_epsilon', 0.3),
            }
        }
        labels, stats = load_clustering_method(clustering_config).cluster(embeddings)
        logger.info(f"HDBSCAN+PCA: {stats['n_clusters']} clusters")
        return labels

    @staticmethod
    def _run_hybrid(config, embeddings):
        from sim_bench.clustering.base import load_clustering_method
        clustering_config = {
            'algorithm': 'hybrid_hdbscan_knn',
            'params': {
                'min_cluster_size': config.get('min_cluster_size', 2),
                'min_samples': config.get('min_samples', 2),
                'cluster_selection_epsilon': config.get('cluster_selection_epsilon', 0.045),
                'knn_k': config.get('knn_k', 3),
                'threshold_floor': config.get('threshold_floor', 0.125),
                'threshold_ceiling': config.get('threshold_ceiling', 0.405),
                'max_exemplars': config.get('max_exemplars', 10),
                'attach_min_exemplars': config.get('attach_min_exemplars', 2),
                'merge_min_pairs': config.get('merge_min_pairs', 3),
            }
        }
        labels, stats = load_clustering_method(clustering_config).cluster(embeddings)
        logger.info(f"HybridHDBSCANKNN: {stats['n_clusters']} clusters")
        return labels

    @staticmethod
    def _run_mutual_knn(config, embeddings):
        from sim_bench.clustering.base import load_clustering_method
        clustering_config = {
            'algorithm': 'mutual_knn',
            'params': {
                'k': config.get('k', 10),
                'similarity_threshold': config.get('similarity_threshold', 0.70),
            }
        }
        labels, stats = load_clustering_method(clustering_config).cluster(embeddings)
        logger.info(f"Mutual KNN: {stats['n_clusters']} clusters")
        return labels

    @staticmethod
    def _run_agglomerative(config, embeddings):
        from sklearn.cluster import AgglomerativeClustering
        n_clusters = config.get("n_clusters")
        distance_threshold = config.get("distance_threshold", 0.5)
        if n_clusters is not None:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        else:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric='cosine', linkage='average')
        labels = clustering.fit_predict(embeddings)
        logger.info(f"Agglomerative: {len(set(labels))} clusters")
        return labels

    def _run_face_cluster_knn(self, faces, embeddings_norm, config, context):
        """Run face_cluster_knn and optionally export artifacts."""
        labels, face_records, base_cr, merged_cr, core_indices, merge_log, merge_metadata, fc_cfg = \
            run_face_cluster_knn(faces, embeddings_norm, config, context)

        if config.get("export_for_analysis", False) and base_cr is not None:
            export_for_analysis(
                face_records, base_cr, merged_cr, core_indices,
                fc_cfg, merge_log, merge_metadata, context,
            )

        return labels
