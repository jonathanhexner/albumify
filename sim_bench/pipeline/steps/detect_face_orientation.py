"""Detect Face Orientation step - determines rotation needed to make face upright.

This step analyzes 5-point facial landmarks to determine if the face is:
- Upright (0°): eyes above nose, nose above mouth
- Upside down (180°): eyes below nose, nose below mouth
- Rotated 90° CW (90°): eyes on right, mouth on left
- Rotated 90° CCW (270°): eyes on left, mouth on right

Unlike roll_angle (eye-line tilt), this detects the actual face orientation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


def detect_face_orientation(landmarks: List[List[float]]) -> int:
    """Determine face orientation from 5-point landmark positions.

    Normal face: eyes at top, nose below eyes, mouth below nose
    Upside-down: eyes at bottom, nose above eyes, mouth above nose
    Rotated 90° CW: eyes on right, mouth on left
    Rotated 90° CCW: eyes on left, mouth on right

    Args:
        landmarks: 5-point landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
                   Each point is [x, y] in pixel coordinates.

    Returns:
        Degrees to rotate clockwise to make face upright: 0, 90, 180, or 270
    """
    if not landmarks or len(landmarks) < 5:
        logger.warning(f"Invalid landmarks for orientation detection: {len(landmarks) if landmarks else 0} points")
        return 0

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[:5]

    # Compute centers
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    nose_x, nose_y = nose[0], nose[1]

    # Compute vectors from eyes to nose and nose to mouth
    # In normal face: eyes->nose goes DOWN (positive y), nose->mouth goes DOWN
    eyes_to_nose_y = nose_y - eye_center_y
    eyes_to_nose_x = nose_x - eye_center_x
    nose_to_mouth_y = mouth_center_y - nose_y
    nose_to_mouth_x = mouth_center_x - nose_x

    # Vertical relationships (image coords: y increases downward)
    eyes_above_nose = eyes_to_nose_y > 0  # eyes have smaller y than nose
    nose_above_mouth = nose_to_mouth_y > 0  # nose has smaller y than mouth

    # Check for upright vs upside-down using vertical dominance
    vertical_extent = abs(eyes_to_nose_y) + abs(nose_to_mouth_y)
    horizontal_extent = abs(eyes_to_nose_x) + abs(nose_to_mouth_x)

    # Use ratio to determine if face is vertical or horizontal orientation
    if vertical_extent < 1e-6 and horizontal_extent < 1e-6:
        logger.warning("Face landmarks are too close together for orientation detection")
        return 0

    is_vertical = vertical_extent >= horizontal_extent

    if is_vertical:
        # Face is roughly vertical (upright or upside-down)
        if eyes_above_nose and nose_above_mouth:
            return 0  # Upright - no rotation needed
        elif not eyes_above_nose and not nose_above_mouth:
            return 180  # Upside down - rotate 180°
        else:
            # Mixed - could be partial rotation, default to checking overall direction
            if not eyes_above_nose:
                return 180
            return 0
    else:
        # Face is roughly horizontal (rotated 90° CW or CCW)
        # When face is tilted 90° CCW (top of head pointing LEFT):
        #   - Eyes are stacked vertically
        #   - Mouth is to the RIGHT of eyes (larger x)
        #   - Need to rotate 90° CW to fix
        # When face is tilted 90° CW (top of head pointing RIGHT):
        #   - Eyes are stacked vertically
        #   - Mouth is to the LEFT of eyes (smaller x)
        #   - Need to rotate 270° CW (= 90° CCW) to fix

        mouth_right_of_eyes = mouth_center_x > eye_center_x

        if mouth_right_of_eyes:
            # Mouth on right, eyes on left -> face tilted 90° CCW -> rotate 90° CW to fix
            return 90
        else:
            # Mouth on left, eyes on right -> face tilted 90° CW -> rotate 270° CW to fix
            return 270


def compute_orientation_confidence(landmarks: List[List[float]], orientation: int) -> float:
    """Compute confidence score for detected orientation.

    Higher scores indicate more confident detection based on how well
    the landmark positions match expected layout for that orientation.

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not landmarks or len(landmarks) < 5:
        return 0.0

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[:5]

    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    nose_y = nose[1]

    # Compute expected relationships for detected orientation
    if orientation == 0:
        # Upright: eyes above nose above mouth (y values should increase)
        score = 0.0
        if eye_center_y < nose_y:
            score += 0.5
        if nose_y < mouth_center_y:
            score += 0.5
        return score
    elif orientation == 180:
        # Upside down: eyes below nose below mouth
        score = 0.0
        if eye_center_y > nose_y:
            score += 0.5
        if nose_y > mouth_center_y:
            score += 0.5
        return score
    else:
        # For 90/270 rotations, check horizontal relationships
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2

        if orientation == 90:
            # Eyes should be to the right of mouth
            return 1.0 if eye_center_x > mouth_center_x else 0.5
        else:  # 270
            # Eyes should be to the left of mouth
            return 1.0 if eye_center_x < mouth_center_x else 0.5


@register_step
class DetectFaceOrientationStep(BaseStep):
    """Detect face orientation and add rotation angles to context.

    Analyzes 5-point landmarks to determine if faces need rotation
    to be upright before alignment. Stores orientation_angle for each face.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="detect_face_orientation",
            display_name="Detect Face Orientation",
            description="Detect if faces are upright, upside-down, or rotated 90°.",
            category="people",
            requires={"insightface_faces"},
            produces={"face_orientations"},
            depends_on=["insightface_detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "log_orientations": {
                        "type": "boolean",
                        "default": True,
                        "description": "Log detected orientations for debugging"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Detect orientation for all faces in context."""
        if not hasattr(context, 'insightface_faces') or not context.insightface_faces:
            logger.info("No faces found in context - skipping orientation detection")
            return

        log_orientations = config.get("log_orientations", True)

        orientation_counts = {0: 0, 90: 0, 180: 0, 270: 0}
        total_faces = 0

        for image_path, face_data in context.insightface_faces.items():
            for face_info in face_data.get('faces', []):
                landmarks = face_info.get('landmarks')

                if not landmarks or len(landmarks) < 5:
                    face_info['orientation_angle'] = 0
                    face_info['orientation_confidence'] = 0.0
                    continue

                orientation = detect_face_orientation(landmarks)
                confidence = compute_orientation_confidence(landmarks, orientation)

                face_info['orientation_angle'] = orientation
                face_info['orientation_confidence'] = confidence

                orientation_counts[orientation] += 1
                total_faces += 1

                if log_orientations and orientation != 0:
                    face_idx = face_info.get('face_index', '?')
                    logger.info(
                        f"Face {face_idx} in {image_path}: orientation={orientation}° "
                        f"(confidence={confidence:.2f})"
                    )

        # Log summary
        logger.info("=" * 60)
        logger.info("DETECT_FACE_ORIENTATION: Summary")
        logger.info("=" * 60)
        logger.info(f"Total faces processed: {total_faces}")
        logger.info(f"  Upright (0°):      {orientation_counts[0]}")
        logger.info(f"  Rotated 90° CW:    {orientation_counts[90]}")
        logger.info(f"  Upside-down (180°):{orientation_counts[180]}")
        logger.info(f"  Rotated 90° CCW:   {orientation_counts[270]}")
        logger.info("=" * 60)

        # Store summary in context for downstream steps
        context.face_orientations = orientation_counts
