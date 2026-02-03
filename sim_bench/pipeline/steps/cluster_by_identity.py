"""Cluster by identity step - sub-cluster scenes by face count and identity."""

import logging
from collections import defaultdict
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


def _get_face_count_bucket(num_faces: int) -> str:
    """Get face count bucket: '0', '1', '2', '3+'."""
    if num_faces == 0:
        return "0"
    elif num_faces == 1:
        return "1"
    elif num_faces == 2:
        return "2"
    else:
        return "3+"


def _get_identity_signature(face_embeddings: list[np.ndarray], threshold: float) -> tuple:
    """
    Create an identity signature from face embeddings.

    For multi-face images, we create a signature based on which known identities
    are present. Returns a tuple of sorted identity indices.
    """
    if not face_embeddings:
        return ()

    # For now, return a hash of the embeddings
    # This will be refined when we have global identity clustering
    signatures = []
    for emb in face_embeddings:
        if emb is not None:
            # Simple quantization to create comparable signatures
            sig = tuple((emb[:8] * 10).astype(int).tolist())
            signatures.append(sig)
    return tuple(sorted(signatures))


@register_step
class ClusterByIdentityStep(BaseStep):
    """
    Sub-cluster scene clusters by face count and face identity.

    This step takes scene clusters and creates sub-clusters based on:
    1. Face count (0, 1, 2, 3+ faces)
    2. Face identity (using ArcFace embeddings)

    Key insight: An image with Mom+Dad is a SEPARATE cluster from
    an image with just Mom. This ensures we select best photos for
    each combination of people.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="cluster_by_identity",
            display_name="Cluster by Identity",
            description="Sub-cluster scenes by face count and face identity (Mom+Dad separate from Mom alone).",
            category="clustering",
            requires={"scene_clusters"},  # faces is optional - handles images without faces
            produces={"face_subclusters"},
            depends_on=["cluster_scenes", "extract_face_embeddings"],
            config_schema={
                "type": "object",
                "properties": {
                    "distance_threshold": {
                        "type": "number",
                        "default": 0.6,
                        "description": "ArcFace distance threshold for same person"
                    },
                    "min_face_area_ratio": {
                        "type": "number",
                        "default": 0.03,
                        "description": "Minimum face area ratio to be considered significant"
                    },
                    "group_by_count": {
                        "type": "boolean",
                        "default": True,
                        "description": "Separate clusters by face count"
                    },
                    "group_by_identity": {
                        "type": "boolean",
                        "default": True,
                        "description": "Separate clusters by face identity"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Create sub-clusters within each scene cluster."""
        distance_threshold = config.get("distance_threshold", 0.6)
        min_face_area_ratio = config.get("min_face_area_ratio", 0.03)
        group_by_count = config.get("group_by_count", True)
        group_by_identity = config.get("group_by_identity", True)

        if not context.scene_clusters:
            context.report_progress("cluster_by_identity", 1.0, "No scene clusters to sub-cluster")
            return

        # face_clusters: scene_id -> subcluster_id -> list of image paths
        face_subclusters = {}
        total_subclusters = 0

        for scene_id, image_paths in context.scene_clusters.items():
            context.report_progress(
                "cluster_by_identity",
                0.5 * (scene_id + 1) / len(context.scene_clusters),
                f"Processing scene {scene_id}"
            )

            # Group images by face count first
            by_face_count = defaultdict(list)

            for img_path in image_paths:
                # Get significant faces for this image
                faces = context.faces.get(img_path, [])
                significant_faces = [f for f in faces if self._is_face_significant(f, min_face_area_ratio)]

                if group_by_count:
                    bucket = _get_face_count_bucket(len(significant_faces))
                else:
                    bucket = "all"

                # Get face embeddings for identity clustering
                embeddings = context.face_embeddings.get(img_path, [])

                by_face_count[bucket].append({
                    "path": img_path,
                    "face_count": len(significant_faces),
                    "embeddings": embeddings[:len(significant_faces)] if embeddings else []
                })

            # Now sub-cluster by identity within each face count bucket
            scene_subclusters = {}
            subcluster_id = 0

            for bucket, images in by_face_count.items():
                if not group_by_identity or bucket == "0":
                    # No faces or identity grouping disabled - single subcluster
                    scene_subclusters[subcluster_id] = {
                        "face_count": bucket,
                        "images": [img["path"] for img in images],
                        "has_faces": bucket != "0"
                    }
                    subcluster_id += 1
                else:
                    # Group by identity signature
                    by_identity = defaultdict(list)

                    for img in images:
                        if img["embeddings"]:
                            sig = self._compute_identity_signature(
                                img["embeddings"],
                                distance_threshold
                            )
                        else:
                            sig = "unknown"
                        by_identity[sig].append(img["path"])

                    for identity_sig, paths in by_identity.items():
                        scene_subclusters[subcluster_id] = {
                            "face_count": bucket,
                            "images": paths,
                            "has_faces": True,
                            "identity": str(identity_sig)
                        }
                        subcluster_id += 1

            face_subclusters[scene_id] = scene_subclusters
            total_subclusters += len(scene_subclusters)

        # Store in context
        context.face_clusters = face_subclusters

        context.report_progress(
            "cluster_by_identity",
            1.0,
            f"Created {total_subclusters} sub-clusters from {len(context.scene_clusters)} scenes"
        )

    def _is_face_significant(self, face, min_area_ratio: float) -> bool:
        """Check if a face is significant enough to count."""
        # Face objects may have different structures depending on detector
        if hasattr(face, 'area_ratio'):
            return face.area_ratio >= min_area_ratio
        if hasattr(face, 'bbox'):
            # Estimate area ratio from bbox if available
            return True  # Assume significant if we have a bbox
        if isinstance(face, dict):
            return face.get('area_ratio', 0.05) >= min_area_ratio
        return True  # Default to significant

    def _compute_identity_signature(self, embeddings: list[np.ndarray], threshold: float) -> tuple:
        """
        Compute a hashable identity signature for a set of face embeddings.

        This allows grouping images with the same people together.
        """
        if not embeddings:
            return ()

        # Quantize embeddings to create comparable signatures
        signatures = []
        for emb in embeddings:
            if emb is not None and len(emb) > 0:
                # Use first 16 dimensions, quantized
                sig = tuple(np.round(emb[:16] * 5).astype(int).tolist())
                signatures.append(sig)

        # Sort for consistent ordering
        return tuple(sorted(signatures))
