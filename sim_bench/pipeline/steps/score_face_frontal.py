"""Score Face Frontal step - compute frontal score and mark faces as clusterable."""

import logging
import math
from typing import Any, Dict, List, Tuple

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.utils.image_cache import get_image_cache

logger = logging.getLogger(__name__)


def compute_distance(p1: List[float], p2: List[float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def compute_roll_angle(landmarks: List[List[float]]) -> float:
    """Compute roll angle from eye landmarks.

    Returns angle in degrees. Positive = clockwise tilt.
    """
    if not landmarks or len(landmarks) < 2:
        return 0.0

    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]

    roll_angle = math.atan2(dy, dx) * 180 / math.pi
    return roll_angle


def compute_centrality(
    bbox: Dict[str, float],
    image_width: int,
    image_height: int
) -> float:
    """Compute how centered the face is in the image.

    Returns a score from 0.0 (edge) to 1.0 (center).
    """
    bbox_x = bbox.get("x_px", 0)
    bbox_y = bbox.get("y_px", 0)
    bbox_w = bbox.get("w_px", 0)
    bbox_h = bbox.get("h_px", 0)

    # Face center
    face_cx = bbox_x + bbox_w / 2
    face_cy = bbox_y + bbox_h / 2

    # Image center
    img_cx = image_width / 2
    img_cy = image_height / 2

    # Normalize distance to [0, 1] where 0 = at center, 1 = at corner
    max_dist = math.sqrt(img_cx * img_cx + img_cy * img_cy)
    dist = math.sqrt((face_cx - img_cx) ** 2 + (face_cy - img_cy) ** 2)

    normalized_dist = dist / max_dist if max_dist > 0 else 0

    # Convert to centrality score (1 = center, 0 = corner)
    centrality = 1.0 - normalized_dist

    return max(0.0, min(1.0, centrality))


def compute_frontal_scores(
    landmarks: List[List[float]],
    bbox: Dict[str, float]
) -> Dict[str, float]:
    """Compute frontal score from 5-point face landmarks.

    Landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]

    Two metrics for detecting yaw (profile view):
    1. Inter-eye / bbox width ratio: Lower = more profile
    2. Nose-eye asymmetry: Higher = more profile

    Returns dict with all intermediate scores.
    """
    if not landmarks or len(landmarks) < 3:
        return {
            "frontal_score": 0.0,
            "eye_bbox_ratio": 0.0,
            "eye_score": 0.0,
            "asymmetry": 0.0,
            "asym_score": 0.0,
            "error": "insufficient_landmarks"
        }

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    bbox_width = bbox.get("w_px", 0)
    if bbox_width <= 0:
        return {
            "frontal_score": 0.0,
            "eye_bbox_ratio": 0.0,
            "eye_score": 0.0,
            "asymmetry": 0.0,
            "asym_score": 0.0,
            "error": "invalid_bbox"
        }

    # Metric 1: Inter-eye / bbox width ratio
    # Frontal: ~0.25-0.30, Profile: ~0.15 or less
    inter_eye = compute_distance(left_eye, right_eye)
    eye_bbox_ratio = inter_eye / bbox_width

    # Normalize to [0, 1]: 0.25+ is ideal frontal, 0.15 is profile
    eye_score = max(0.0, min(1.0, (eye_bbox_ratio - 0.15) / (0.25 - 0.15)))

    # Metric 2: Nose-eye asymmetry ratio
    # Frontal: ~1.0, Profile: 2.0+
    dist_nose_left = compute_distance(nose, left_eye)
    dist_nose_right = compute_distance(nose, right_eye)

    min_dist = min(dist_nose_left, dist_nose_right)
    max_dist = max(dist_nose_left, dist_nose_right)

    asymmetry = max_dist / (min_dist + 1e-6)

    # Normalize to [0, 1]: 1.0 is ideal, 2.0+ is profile
    asym_score = max(0.0, min(1.0, 1.0 - (asymmetry - 1.0) / 1.0))

    # Combine scores (weighted average)
    frontal_score = (eye_score + asym_score) / 2

    return {
        "frontal_score": frontal_score,
        "eye_bbox_ratio": eye_bbox_ratio,
        "eye_score": eye_score,
        "asymmetry": asymmetry,
        "asym_score": asym_score,
    }


@register_step
class ScoreFaceFrontalStep(BaseStep):
    """Compute frontal score and mark faces as clusterable.

    This step analyzes face landmarks to determine if a face is frontal
    (suitable for clustering) or profile/rotated (excluded from clustering).

    Non-frontal faces are kept in the data but marked `is_clusterable=False`.
    Their `frontal_score` is used for penalty calculation in select_best.

    Also computes:
    - roll_angle: Head tilt angle (used for alignment before embedding)
    - centrality: How centered the face is (used for penalty weighting)
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_face_frontal",
            display_name="Score Face Frontal",
            description="Compute frontal score and mark faces as clusterable based on landmark geometry.",
            category="analysis",
            requires={"insightface_faces"},
            produces={"insightface_faces"},
            depends_on=["filter_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_frontal_score": {
                        "type": "number",
                        "default": 0.4,
                        "description": "Minimum frontal score for clustering"
                    },
                    "min_eye_bbox_ratio": {
                        "type": "number",
                        "default": 0.20,
                        "description": "Minimum inter_eye/bbox_width (below = profile)"
                    },
                    "max_asymmetry": {
                        "type": "number",
                        "default": 1.8,
                        "description": "Maximum nose-eye asymmetry (above = profile)"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Compute frontal scores and mark faces as clusterable."""
        min_frontal_score = config.get("min_frontal_score", 0.4)

        cache = get_image_cache()

        total_faces = 0
        clusterable_count = 0
        non_clusterable_count = 0
        skipped_filtered = 0

        for image_path, face_data in context.insightface_faces.items():
            faces = face_data.get("faces", [])
            if not faces:
                continue

            # Get image dimensions for centrality calculation
            try:
                image_width, image_height = cache.get_dimensions(image_path)
            except Exception as e:
                logger.warning(f"Could not get dimensions for {image_path}: {e}")
                image_width, image_height = 0, 0

            for face in faces:
                # Skip faces that didn't pass filtering
                if not face.get("filter_passed", True):
                    skipped_filtered += 1
                    face["frontal_scores"] = {}
                    face["frontal_score"] = 0.0
                    face["is_clusterable"] = False
                    face["roll_angle"] = 0.0
                    face["centrality"] = 0.0
                    continue

                total_faces += 1

                landmarks = face.get("landmarks", [])
                bbox = face.get("bbox", {})

                # Compute frontal scores
                frontal_scores = compute_frontal_scores(landmarks, bbox)
                face["frontal_scores"] = frontal_scores
                face["frontal_score"] = frontal_scores["frontal_score"]

                # Compute roll angle
                roll_angle = compute_roll_angle(landmarks)
                face["roll_angle"] = roll_angle

                # Compute centrality
                if image_width > 0 and image_height > 0:
                    centrality = compute_centrality(bbox, image_width, image_height)
                else:
                    centrality = 0.5  # Default to middle
                face["centrality"] = centrality

                # Determine if clusterable
                is_clusterable = frontal_scores["frontal_score"] >= min_frontal_score
                face["is_clusterable"] = is_clusterable

                if is_clusterable:
                    clusterable_count += 1
                else:
                    non_clusterable_count += 1

            # Store stats for this image
            face_data["frontal_stats"] = {
                "total": len([f for f in faces if f.get("filter_passed", True)]),
                "clusterable": sum(1 for f in faces if f.get("is_clusterable", False)),
                "non_clusterable": sum(
                    1 for f in faces
                    if f.get("filter_passed", True) and not f.get("is_clusterable", False)
                ),
            }

        context.report_progress(
            "score_face_frontal",
            1.0,
            f"Scored {total_faces} faces: {clusterable_count} clusterable, "
            f"{non_clusterable_count} non-clusterable, {skipped_filtered} skipped (filtered)"
        )

        # Detailed logging for debugging
        logger.info("=" * 60)
        logger.info("SCORE_FACE_FRONTAL STEP COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Config: min_frontal_score={min_frontal_score}")
        logger.info(f"Results: {clusterable_count}/{total_faces} clusterable, "
                   f"{non_clusterable_count} non-clusterable, {skipped_filtered} skipped (pre-filtered)")

        # Log frontal score distribution
        frontal_scores = []
        for image_path, face_data in context.insightface_faces.items():
            for face in face_data.get("faces", []):
                if face.get("filter_passed", True):
                    fs = face.get("frontal_score")
                    if fs is not None:
                        frontal_scores.append(fs)

        if frontal_scores:
            avg_score = sum(frontal_scores) / len(frontal_scores)
            min_score = min(frontal_scores)
            max_score = max(frontal_scores)
            logger.info(f"Frontal score distribution: min={min_score:.2f}, max={max_score:.2f}, avg={avg_score:.2f}")

        # Log sample face scores
        sample_count = 0
        for image_path, face_data in context.insightface_faces.items():
            if sample_count >= 3:
                break
            for face in face_data.get("faces", []):
                if sample_count >= 3:
                    break
                if not face.get("filter_passed", True):
                    continue
                frontal_data = face.get("frontal_scores", {})
                logger.info(f"  Sample face: clusterable={face.get('is_clusterable', '?')}, "
                           f"frontal_score={face.get('frontal_score', '?'):.2f}, "
                           f"eye_bbox_ratio={frontal_data.get('eye_bbox_ratio', '?'):.3f}, "
                           f"asymmetry={frontal_data.get('asymmetry', '?'):.2f}, "
                           f"roll={face.get('roll_angle', '?'):.1f}deg")
                sample_count += 1

        logger.info("=" * 60)
