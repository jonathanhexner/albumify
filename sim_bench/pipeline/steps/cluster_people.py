"""Cluster People step - group faces by identity."""

import logging
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


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

    def process(self, context: PipelineContext, config: dict) -> None:
        """Cluster faces by identity."""
        # Collect all faces with embeddings from context.faces
        # (context.faces is keyed by image path, each value is a list of Face objects)
        faces_with_embeddings = []

        for img_path, faces in context.faces.items():
            embeddings_for_img = context.face_embeddings.get(img_path, [])
            for i, face in enumerate(faces):
                # Get corresponding embedding if available
                if i < len(embeddings_for_img) and embeddings_for_img[i] is not None:
                    face.embedding = embeddings_for_img[i]
                    faces_with_embeddings.append(face)
                elif hasattr(face, 'embedding') and face.embedding is not None:
                    faces_with_embeddings.append(face)

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
        
        # Run clustering
        method = config.get("method", "agglomerative")
        distance_threshold = config.get("distance_threshold", 0.5)
        n_clusters = config.get("n_clusters")
        
        if method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            
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
