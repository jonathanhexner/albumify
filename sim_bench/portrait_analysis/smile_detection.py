"""
Smile detection using mouth landmarks.
"""

import logging
from typing import Dict, Tuple, Any

from sim_bench.portrait_analysis.types import SmileState
from sim_bench.portrait_analysis.eye_state import euclidean_distance, get_landmark_coords

logger = logging.getLogger(__name__)

# Mouth landmark indices
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_TOP = 0
MOUTH_BOTTOM = 17
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def detect_smile(
    face_landmarks,
    image_shape: Tuple[int, int],
    width_threshold: float,
    elevation_threshold: float
) -> Dict[str, Any]:
    """
    Detect smile based on mouth landmarks.

    Args:
        face_landmarks: MediaPipe face mesh landmarks
        image_shape: Image dimensions (height, width)
        width_threshold: Threshold for mouth width ratio
        elevation_threshold: Threshold for mouth corner elevation

    Returns:
        Dict with smile detection results
    """
    left_corner = get_landmark_coords(face_landmarks.landmark[MOUTH_LEFT_CORNER], image_shape)
    right_corner = get_landmark_coords(face_landmarks.landmark[MOUTH_RIGHT_CORNER], image_shape)
    mouth_top = get_landmark_coords(face_landmarks.landmark[MOUTH_TOP], image_shape)
    mouth_bottom = get_landmark_coords(face_landmarks.landmark[MOUTH_BOTTOM], image_shape)

    mouth_width = euclidean_distance(left_corner, right_corner)

    left_eye_outer = get_landmark_coords(face_landmarks.landmark[LEFT_EYE_OUTER], image_shape)
    right_eye_outer = get_landmark_coords(face_landmarks.landmark[RIGHT_EYE_OUTER], image_shape)
    face_width = euclidean_distance(left_eye_outer, right_eye_outer) * 1.5

    mouth_width_ratio = mouth_width / face_width if face_width > 0 else 0.0

    mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
    left_elevation = (mouth_center_y - left_corner[1]) / image_shape[0]
    right_elevation = (mouth_center_y - right_corner[1]) / image_shape[0]
    avg_elevation = (left_elevation + right_elevation) / 2

    lip_separation = euclidean_distance(mouth_top, mouth_bottom) / image_shape[0]

    smile_score = (
        mouth_width_ratio * 0.5 +
        max(0, avg_elevation) * 0.3 +
        lip_separation * 0.2
    )
    smile_score = min(1.0, max(0.0, smile_score))

    is_smiling = mouth_width_ratio > width_threshold and avg_elevation > elevation_threshold

    return {
        'is_smiling': is_smiling,
        'smile_score': smile_score,
        'mouth_width_ratio': mouth_width_ratio,
        'corner_elevation': avg_elevation,
        'lip_separation': lip_separation,
        'mouth_coords': {
            'left_corner': left_corner,
            'right_corner': right_corner,
            'top': mouth_top,
            'bottom': mouth_bottom
        }
    }


def create_smile_state(smile_data: Dict[str, Any]) -> SmileState:
    """Create SmileState dataclass from detection result."""
    return SmileState(
        is_smiling=smile_data['is_smiling'],
        smile_score=smile_data['smile_score'],
        mouth_width_ratio=smile_data['mouth_width_ratio'],
        corner_elevation=smile_data['corner_elevation']
    )
