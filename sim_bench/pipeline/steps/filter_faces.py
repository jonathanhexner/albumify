"""Filter Faces step - remove small and low-confidence faces."""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.utils.image_cache import get_image_cache

logger = logging.getLogger(__name__)


def compute_inter_eye_distance(landmarks: List[List[float]]) -> float:
    """Compute distance between left and right eye from 5-point landmarks.

    Landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    if not landmarks or len(landmarks) < 2:
        return 0.0

    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    return math.sqrt(dx * dx + dy * dy)


@register_step
class FilterFacesStep(BaseStep):
    """Filter out small and low-confidence faces.

    Faces that don't pass the filter are marked with `filter_passed=False`.
    They are kept in the data structure for debugging but ignored by downstream steps.

    Filter criteria:
    - min_confidence: Detection confidence threshold
    - min_bbox_ratio: Minimum bbox_width / image_width
    - min_relative_size: Minimum bbox_width / max_bbox_width in same image
    - min_eye_ratio: Minimum inter_eye_distance / image_width
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="filter_faces",
            display_name="Filter Small Faces",
            description="Filter out faces that are too small or have low detection confidence.",
            category="filtering",
            requires={"insightface_faces"},
            produces={"insightface_faces"},
            depends_on=["insightface_detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_confidence": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Minimum detection confidence"
                    },
                    "min_bbox_ratio": {
                        "type": "number",
                        "default": 0.02,
                        "description": "Minimum bbox_width / image_width"
                    },
                    "min_relative_size": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Minimum bbox_width / max_bbox_width in image"
                    },
                    "min_eye_ratio": {
                        "type": "number",
                        "default": 0.01,
                        "description": "Minimum inter_eye_distance / image_width"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Filter faces based on size and confidence criteria."""
        min_confidence = config.get("min_confidence", 0.5)
        min_bbox_ratio = config.get("min_bbox_ratio", 0.02)
        min_relative_size = config.get("min_relative_size", 0.3)
        min_eye_ratio = config.get("min_eye_ratio", 0.01)

        cache = get_image_cache()

        total_faces = 0
        filtered_count = 0

        for image_path, face_data in context.insightface_faces.items():
            faces = face_data.get("faces", [])
            if not faces:
                continue

            # Get image dimensions
            try:
                image_width, image_height = cache.get_dimensions(image_path)
            except Exception as e:
                logger.warning(f"Could not get dimensions for {image_path}: {e}")
                # Mark all faces as filtered
                for face in faces:
                    face["filter_passed"] = False
                    face["filter_scores"] = {}
                    face["filter_reason"] = "image_error"
                continue

            # Find max bbox width in this image
            max_bbox_width = max(
                (f.get("bbox", {}).get("w_px", 0) for f in faces),
                default=0
            )

            for face in faces:
                total_faces += 1
                bbox = face.get("bbox", {})
                bbox_width = bbox.get("w_px", 0)
                confidence = face.get("confidence", 0)
                landmarks = face.get("landmarks", [])

                inter_eye = compute_inter_eye_distance(landmarks)

                # Compute filter scores
                filter_scores = {
                    "confidence": confidence,
                    "bbox_ratio": bbox_width / image_width if image_width > 0 else 0,
                    "relative_size": bbox_width / max_bbox_width if max_bbox_width > 0 else 0,
                    "eye_ratio": inter_eye / image_width if image_width > 0 else 0,
                    "bbox_width": bbox_width,
                    "inter_eye": inter_eye,
                    "image_width": image_width,
                    "max_bbox_width": max_bbox_width,
                }

                # Check thresholds
                passes = True
                fail_reasons = []

                if filter_scores["confidence"] < min_confidence:
                    passes = False
                    fail_reasons.append(f"confidence={filter_scores['confidence']:.2f}<{min_confidence}")

                if filter_scores["bbox_ratio"] < min_bbox_ratio:
                    passes = False
                    fail_reasons.append(f"bbox_ratio={filter_scores['bbox_ratio']:.3f}<{min_bbox_ratio}")

                if filter_scores["relative_size"] < min_relative_size:
                    passes = False
                    fail_reasons.append(f"relative_size={filter_scores['relative_size']:.2f}<{min_relative_size}")

                if filter_scores["eye_ratio"] < min_eye_ratio:
                    passes = False
                    fail_reasons.append(f"eye_ratio={filter_scores['eye_ratio']:.4f}<{min_eye_ratio}")

                face["filter_scores"] = filter_scores
                face["filter_passed"] = passes
                face["filter_reason"] = ", ".join(fail_reasons) if fail_reasons else "passed"

                if not passes:
                    filtered_count += 1

            # Store count of filtered faces for this image
            face_data["filter_stats"] = {
                "total": len(faces),
                "passed": len(faces) - sum(1 for f in faces if not f.get("filter_passed", True)),
                "filtered": sum(1 for f in faces if not f.get("filter_passed", True)),
            }

        passed_count = total_faces - filtered_count
        context.report_progress(
            "filter_faces",
            1.0,
            f"Filtered {filtered_count}/{total_faces} faces, {passed_count} passed"
        )

        # Detailed logging for debugging
        logger.info("=" * 60)
        logger.info("FILTER_FACES STEP COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Config: min_confidence={min_confidence}, min_bbox_ratio={min_bbox_ratio}, "
                   f"min_relative_size={min_relative_size}, min_eye_ratio={min_eye_ratio}")
        logger.info(f"Results: {passed_count}/{total_faces} faces passed, {filtered_count} filtered")

        # Count failures by criterion
        fail_by_confidence = 0
        fail_by_bbox = 0
        fail_by_relative = 0
        fail_by_eye = 0

        for image_path, face_data in context.insightface_faces.items():
            for face in face_data.get("faces", []):
                reason = face.get("filter_reason", "")
                if "confidence" in reason:
                    fail_by_confidence += 1
                if "bbox_ratio" in reason:
                    fail_by_bbox += 1
                if "relative_size" in reason:
                    fail_by_relative += 1
                if "eye_ratio" in reason:
                    fail_by_eye += 1

        logger.info(f"Failures by criterion: confidence={fail_by_confidence}, "
                   f"bbox_ratio={fail_by_bbox}, relative_size={fail_by_relative}, eye_ratio={fail_by_eye}")

        # Log sample face scores
        sample_count = 0
        for image_path, face_data in context.insightface_faces.items():
            if sample_count >= 3:
                break
            for face in face_data.get("faces", []):
                if sample_count >= 3:
                    break
                scores = face.get("filter_scores", {})
                passed = face.get("filter_passed", "?")
                logger.info(f"  Sample face: passed={passed}, confidence={scores.get('confidence', '?'):.2f}, "
                           f"bbox_ratio={scores.get('bbox_ratio', '?'):.3f}, "
                           f"relative_size={scores.get('relative_size', '?'):.2f}")
                sample_count += 1

        logger.info("=" * 60)
