"""Cluster People step - group faces by identity."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

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
                        "enum": ["agglomerative"],
                        "default": "agglomerative",
                        "description": "Clustering algorithm"
                    },
                    "distance_threshold": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Distance threshold for agglomerative clustering (lower = stricter)"
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Fixed number of clusters (optional, overrides distance_threshold)"
                    }
                }
            }
        )

    def _collect_faces_with_embeddings(self, context: PipelineContext) -> List[FaceForClustering]:
        """
        Collect all faces with embeddings from context.

        Works with both MediaPipe (context.faces) and InsightFace (context.insightface_faces).
        Embeddings are looked up from context.face_embeddings using cache key format.
        """
        faces_with_embeddings = []

        # Debug: log what's available
        logger.info(f"Collecting faces: MediaPipe faces={len(context.faces)}, "
                   f"InsightFace faces={len(context.insightface_faces) if hasattr(context, 'insightface_faces') else 0}, "
                   f"face_embeddings={len(context.face_embeddings)}")

        if context.face_embeddings:
            sample_keys = list(context.face_embeddings.keys())[:3]
            logger.info(f"Sample embedding keys: {sample_keys}")

        # Try MediaPipe faces first (context.faces)
        if context.faces:
            logger.info(f"Processing {len(context.faces)} MediaPipe face entries")
            for img_path, faces in context.faces.items():
                for face in faces:
                    # Generate cache key to look up embedding (normalize path)
                    path_str = str(face.original_path).replace('\\', '/')
                    cache_key = f"{path_str}:face_{face.face_index}"
                    embedding = context.face_embeddings.get(cache_key)

                    if embedding is not None:
                        face.embedding = embedding
                        faces_with_embeddings.append(face)
                    elif hasattr(face, 'embedding') and face.embedding is not None:
                        faces_with_embeddings.append(face)

        # Try InsightFace faces (context.insightface_faces)
        if hasattr(context, 'insightface_faces') and context.insightface_faces:
            logger.info(f"Processing {len(context.insightface_faces)} InsightFace face entries")
            matched = 0
            unmatched = 0
            for img_path, face_data in context.insightface_faces.items():
                # Normalize path for consistent cache key lookup
                path_str = str(img_path).replace('\\', '/')
                for face_info in face_data.get('faces', []):
                    face_index = face_info.get('face_index', 0)

                    # Generate cache key to look up embedding
                    cache_key = f"{path_str}:face_{face_index}"
                    embedding = context.face_embeddings.get(cache_key)

                    if embedding is not None:
                        matched += 1
                        face = FaceForClustering(
                            original_path=Path(img_path),
                            face_index=face_index,
                            embedding=embedding,
                            bbox=face_info.get('bbox', {})
                        )
                        faces_with_embeddings.append(face)
                    else:
                        unmatched += 1

            logger.info(f"InsightFace: matched {matched} embeddings, unmatched {unmatched}")

        logger.info(f"Total faces with embeddings: {len(faces_with_embeddings)}")
        return faces_with_embeddings

    def process(self, context: PipelineContext, config: dict) -> None:
        """Cluster faces by identity."""
        # Collect all faces with embeddings (works with both MediaPipe and InsightFace)
        faces_with_embeddings = self._collect_faces_with_embeddings(context)

        if not faces_with_embeddings:
            context.report_progress("cluster_people", 1.0, "No faces with embeddings to cluster")
            return

        logger.info(f"Clustering {len(faces_with_embeddings)} faces with embeddings")

        if len(faces_with_embeddings) == 1:
            # Single face - assign to cluster 0
            faces_with_embeddings[0].cluster_id = 0
            context.people_clusters = {0: [faces_with_embeddings[0]]}
            context.report_progress("cluster_people", 1.0, "Single face detected (1 person)")
            return

        # Prepare embeddings for clustering
        embeddings = np.array([f.embedding for f in faces_with_embeddings])

        # Normalize embeddings for cosine similarity (HDBSCAN uses euclidean on normalized = cosine)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings / norms

        # Run clustering
        method = config.get("method", "hdbscan")

        if method == "hdbscan":
            import hdbscan

            min_cluster_size = config.get("min_cluster_size", 2)
            min_samples = config.get("min_samples", min_cluster_size)
            # cluster_selection_epsilon: merge clusters within this distance
            # Higher = more merging = fewer clusters (reduces over-segmentation)
            cluster_selection_epsilon = config.get("cluster_selection_epsilon", 0.3)

            context.report_progress("cluster_people", 0.5, "Running HDBSCAN clustering")

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',  # On normalized vectors, euclidean ≈ cosine
                cluster_selection_method='eom',  # Excess of Mass (better for varying density)
                cluster_selection_epsilon=cluster_selection_epsilon,  # Merge nearby clusters
            )
            labels = clusterer.fit_predict(embeddings_normalized)

            # Count noise points (label -1)
            noise_count = np.sum(labels == -1)
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"HDBSCAN: {num_clusters} clusters, {noise_count} noise points, epsilon={cluster_selection_epsilon}")

        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            distance_threshold = config.get("distance_threshold", 0.5)
            n_clusters = config.get("n_clusters")

            if n_clusters is not None:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric='cosine',
                    linkage='average'
                )

            context.report_progress("cluster_people", 0.5, "Running agglomerative clustering")
            labels = clustering.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Assign cluster IDs to faces
        for face, label in zip(faces_with_embeddings, labels):
            face.cluster_id = int(label)
        
        # Group faces by cluster
        clusters = {}
        for face in faces_with_embeddings:
            if face.cluster_id not in clusters:
                clusters[face.cluster_id] = []
            clusters[face.cluster_id].append(face)
        
        # Store in context
        context.people_clusters = clusters
        
        num_people = len(clusters)
        avg_faces_per_person = len(faces_with_embeddings) / num_people if num_people > 0 else 0
        
        context.report_progress(
            "cluster_people", 1.0,
            f"Clustered {len(faces_with_embeddings)} faces into {num_people} identities "
            f"(avg {avg_faces_per_person:.1f} faces/person)"
        )
