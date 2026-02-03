"""Select Best Per Person step - choose best face/image for each detected person."""

import logging

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


@register_step
class SelectBestPerPersonStep(BaseStep):
    """Select the best face image for each detected person.

    Uses face quality scores (pose, eyes, smile) to rank faces
    and select the best representative image for each person.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="select_best_per_person",
            display_name="Select Best Per Person",
            description="Select the best face image for each detected person based on quality scores.",
            category="selection",
            requires={"people_clusters"},
            produces={"people_thumbnails", "people_best_images"},
            depends_on=["cluster_people"],
            config_schema={
                "type": "object",
                "properties": {
                    "pose_weight": {
                        "type": "number",
                        "default": 0.4,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Weight for frontal pose score"
                    },
                    "eyes_weight": {
                        "type": "number",
                        "default": 0.3,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Weight for eyes open score"
                    },
                    "smile_weight": {
                        "type": "number",
                        "default": 0.2,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Weight for smile score"
                    },
                    "sharpness_weight": {
                        "type": "number",
                        "default": 0.1,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Weight for face sharpness"
                    },
                    "require_eyes_open": {
                        "type": "boolean",
                        "default": True,
                        "description": "Filter out faces with closed eyes"
                    },
                    "prefer_smiling": {
                        "type": "boolean",
                        "default": False,
                        "description": "Prefer smiling faces when available"
                    }
                }
            }
        )

    def _compute_face_score(self, face, config: dict) -> float:
        """Compute composite score for a face."""
        pose_weight = config.get("pose_weight", 0.4)
        eyes_weight = config.get("eyes_weight", 0.3)
        smile_weight = config.get("smile_weight", 0.2)
        sharpness_weight = config.get("sharpness_weight", 0.1)

        # Get scores from face (default to 0.5 if not available)
        pose_score = 0.5
        if hasattr(face, 'pose') and face.pose:
            pose_score = face.pose.frontal_score

        eyes_score = face.eyes_open_score if face.eyes_open_score is not None else 0.5
        smile_score = face.smile_score if face.smile_score is not None else 0.5

        sharpness_score = 0.5
        if hasattr(face, 'quality') and face.quality:
            sharpness_score = face.quality.sharpness_score

        # Compute weighted score
        total_weight = pose_weight + eyes_weight + smile_weight + sharpness_weight
        if total_weight == 0:
            return 0.5

        composite = (
            pose_score * pose_weight +
            eyes_score * eyes_weight +
            smile_score * smile_weight +
            sharpness_score * sharpness_weight
        ) / total_weight

        return composite

    def process(self, context: PipelineContext, config: dict) -> None:
        """Select best face for each person cluster."""
        if not context.people_clusters:
            context.report_progress(
                "select_best_per_person", 1.0,
                "No people clusters to process"
            )
            return

        require_eyes_open = config.get("require_eyes_open", True)
        prefer_smiling = config.get("prefer_smiling", False)

        thumbnails = {}
        best_images = {}

        total_people = len(context.people_clusters)

        for i, (cluster_id, faces) in enumerate(context.people_clusters.items()):
            if not faces:
                continue

            # Filter faces if needed
            candidates = list(faces)

            if require_eyes_open:
                eyes_open_candidates = [
                    f for f in candidates
                    if getattr(f, 'both_eyes_open', True) is not False
                ]
                # Only filter if we have some candidates left
                if eyes_open_candidates:
                    candidates = eyes_open_candidates

            if prefer_smiling:
                smiling_candidates = [
                    f for f in candidates
                    if getattr(f, 'is_smiling', False)
                ]
                # Only filter if we have some candidates left
                if smiling_candidates:
                    candidates = smiling_candidates

            # Score remaining candidates
            scored_faces = [
                (face, self._compute_face_score(face, config))
                for face in candidates
            ]

            # Sort by score descending
            scored_faces.sort(key=lambda x: x[1], reverse=True)

            # Best face becomes thumbnail
            if scored_faces:
                best_face, best_score = scored_faces[0]
                thumbnails[cluster_id] = best_face
                best_images[cluster_id] = {
                    'image_path': str(best_face.original_path),
                    'face_index': best_face.face_index,
                    'score': best_score,
                    'bbox': list(best_face.bbox) if best_face.bbox is not None else None
                }

                logger.debug(
                    f"Person {cluster_id}: best face from {best_face.original_path} "
                    f"(score={best_score:.3f})"
                )

            progress = (i + 1) / total_people
            context.report_progress(
                "select_best_per_person", progress,
                f"Selecting best for person {i + 1}/{total_people}"
            )

        # Store results
        context.people_thumbnails = thumbnails
        context.people_best_images = best_images

        context.report_progress(
            "select_best_per_person", 1.0,
            f"Selected best faces for {len(thumbnails)} people"
        )

        logger.info(f"Selected best faces for {len(thumbnails)} people")
