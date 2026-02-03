"""Filter Best Faces step - select highest quality faces per image."""

import logging
from pathlib import Path

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


@register_step
class FilterBestFacesStep(BaseStep):
    """Filter to keep only the best quality face from each image."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="filter_best_faces",
            display_name="Filter Best Faces",
            description="Keep only the highest quality face from each image (for face clustering).",
            category="people",
            requires={"faces"},
            produces={"all_faces"},
            depends_on=["detect_faces", "score_face_quality"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_quality_score": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Minimum overall quality score to keep a face"
                    },
                    "require_eyes_open": {
                        "type": "boolean",
                        "default": False,
                        "description": "Only keep faces with both eyes open"
                    },
                    "min_face_ratio": {
                        "type": "number",
                        "default": 0.02,
                        "description": "Minimum face size ratio"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Filter to best faces."""
        if not context.faces:
            context.report_progress("filter_best_faces", 1.0, "No faces to filter")
            return
        
        min_quality = config.get("min_quality_score", 0.3)
        require_eyes = config.get("require_eyes_open", False)
        min_face_ratio = config.get("min_face_ratio", 0.02)
        
        best_faces = []
        
        for image_path_str, faces_list in context.faces.items():
            if not faces_list:
                continue
            
            # Filter by criteria
            valid_faces = []
            for face in faces_list:
                # Size check
                if face.face_ratio < min_face_ratio:
                    continue
                
                # Quality check
                if not face.quality or face.quality.overall < min_quality:
                    continue
                
                # Eyes check
                if require_eyes and not face.both_eyes_open:
                    continue
                
                valid_faces.append(face)
            
            # Select best from valid faces
            if valid_faces:
                best_face = max(valid_faces, key=lambda f: f.quality.overall)
                best_faces.append(best_face)
        
        # Store in context
        context.all_faces = best_faces
        
        total_faces = sum(len(faces) for faces in context.faces.values())
        context.report_progress(
            "filter_best_faces", 1.0,
            f"Selected {len(best_faces)} best faces from {total_faces} total"
        )
