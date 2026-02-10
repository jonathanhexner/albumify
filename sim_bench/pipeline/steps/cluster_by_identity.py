"""Cluster by identity step - sub-cluster scenes by face count and identity."""

import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional

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


def _build_face_to_person_lookup(people_clusters: dict) -> Dict[Tuple[str, int], int]:
    """
    Build a lookup from (image_path, face_index) -> person_id.

    Uses the global people_clusters from cluster_people step.
    Normalizes paths to forward slashes for consistent lookups.
    """
    lookup = {}
    for person_id, faces in people_clusters.items():
        for face in faces:
            # Normalize path to forward slashes
            path = str(face.original_path).replace('\\', '/')
            key = (path, face.face_index)
            lookup[key] = person_id
    return lookup


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
            depends_on=["cluster_scenes", "cluster_people"],  # Uses global person IDs
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
        min_face_area_ratio = config.get("min_face_area_ratio", 0.03)
        group_by_count = config.get("group_by_count", True)
        group_by_identity = config.get("group_by_identity", True)

        if not context.scene_clusters:
            context.report_progress("cluster_by_identity", 1.0, "No scene clusters to sub-cluster")
            return

        # Build lookup from (image_path, face_index) -> person_id
        # This uses the global clustering from cluster_people step
        face_to_person = {}
        if context.people_clusters:
            face_to_person = _build_face_to_person_lookup(context.people_clusters)
            logger.info(f"Built face-to-person lookup with {len(face_to_person)} entries")
        else:
            logger.warning("No people_clusters available - identity grouping will be limited")

        # face_clusters: scene_id -> subcluster_id -> list of image paths
        face_subclusters = {}
        total_subclusters = 0

        for scene_id, image_paths in context.scene_clusters.items():
            context.report_progress(
                "cluster_by_identity",
                0.5 * (scene_id + 1) / len(context.scene_clusters),
                f"Processing scene {scene_id}"
            )

            # Group images by face count first, then by identity
            by_face_count = defaultdict(list)

            for img_path in image_paths:
                # Normalize path for lookups
                path_normalized = img_path.replace('\\', '/')

                # Get significant faces for this image - check both MediaPipe and InsightFace
                faces = context.faces.get(img_path, [])
                significant_faces = [f for f in faces if self._is_face_significant(f, min_face_area_ratio)]

                # Also check InsightFace faces
                insightface_faces = []
                if hasattr(context, 'insightface_faces') and context.insightface_faces:
                    face_data = context.insightface_faces.get(path_normalized, {})
                    if not face_data:
                        face_data = context.insightface_faces.get(img_path, {})
                    insightface_faces = face_data.get('faces', [])

                # Use whichever has faces
                num_faces = len(significant_faces) if significant_faces else len(insightface_faces)

                if group_by_count:
                    bucket = _get_face_count_bucket(num_faces)
                else:
                    bucket = "all"

                # Look up person IDs for each face using global clustering
                person_ids = []

                # MediaPipe faces
                for face in significant_faces:
                    face_path = str(face.original_path).replace('\\', '/')
                    key = (face_path, face.face_index)
                    person_id = face_to_person.get(key)
                    if person_id is not None:
                        person_ids.append(person_id)

                # InsightFace faces
                for face_info in insightface_faces:
                    face_index = face_info.get('face_index', 0)
                    key = (path_normalized, face_index)
                    person_id = face_to_person.get(key)
                    if person_id is not None and person_id not in person_ids:
                        person_ids.append(person_id)

                by_face_count[bucket].append({
                    "path": img_path,
                    "face_count": num_faces,
                    "person_ids": person_ids
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
                    # Group by person ID combination (using global clustering)
                    by_identity = defaultdict(list)

                    for img in images:
                        if img["person_ids"]:
                            # Sort person IDs for consistent grouping
                            sig = tuple(sorted(img["person_ids"]))
                        else:
                            sig = ("unknown",)
                        by_identity[sig].append(img["path"])

                    for identity_sig, paths in by_identity.items():
                        # Create human-readable identity label
                        if identity_sig == ("unknown",):
                            identity_label = "unknown"
                        else:
                            identity_label = "+".join(f"Person_{pid}" for pid in identity_sig)

                        scene_subclusters[subcluster_id] = {
                            "face_count": bucket,
                            "images": paths,
                            "has_faces": True,
                            "identity": identity_label,
                            "person_ids": list(identity_sig) if identity_sig != ("unknown",) else []
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
        if hasattr(face, 'face_ratio'):
            return face.face_ratio >= min_area_ratio
        if hasattr(face, 'area_ratio'):
            return face.area_ratio >= min_area_ratio
        if hasattr(face, 'bbox'):
            # Estimate area ratio from bbox if available
            return True  # Assume significant if we have a bbox
        if isinstance(face, dict):
            return face.get('area_ratio', 0.05) >= min_area_ratio
        return True  # Default to significant
