"""Filter Portraits step - select images with dominant faces."""

import logging
from pathlib import Path

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


@register_step
class FilterPortraitsStep(BaseStep):
    """Filter images to keep only those with face-dominant composition."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="filter_portraits",
            display_name="Filter Portraits",
            description="Keep only images with face-dominant composition (portraits).",
            category="filtering",
            requires={"faces"},
            produces={"active_images"},
            depends_on=["detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_face_ratio": {
                        "type": "number",
                        "default": 0.02,
                        "description": "Minimum face size as ratio of image area"
                    },
                    "require_frontal_pose": {
                        "type": "boolean",
                        "default": False,
                        "description": "Only keep images with frontal-facing faces"
                    },
                    "min_pose_score": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Minimum frontal pose score (if require_frontal_pose=True)"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Filter to portrait images."""
        if not context.faces:
            context.report_progress("filter_portraits", 1.0, "No faces detected, no filtering applied")
            return
        
        min_face_ratio = config.get("min_face_ratio", 0.02)
        require_frontal = config.get("require_frontal_pose", False)
        min_pose_score = config.get("min_pose_score", 0.5)
        
        portrait_images = set()
        
        for image_path_str, faces_list in context.faces.items():
            if not faces_list:
                continue
            
            # Check each face
            for face in faces_list:
                # Size check
                if face.face_ratio < min_face_ratio:
                    continue
                
                # Pose check (if required)
                if require_frontal:
                    if not face.pose or face.pose.frontal_score < min_pose_score:
                        continue
                
                # This image has a valid face
                portrait_images.add(Path(image_path_str))
                break
        
        # Update active images
        context.active_images = portrait_images
        
        filtered_count = len(context.image_paths) - len(portrait_images)
        context.report_progress(
            "filter_portraits", 1.0,
            f"Kept {len(portrait_images)} portraits, filtered {filtered_count} images"
        )
