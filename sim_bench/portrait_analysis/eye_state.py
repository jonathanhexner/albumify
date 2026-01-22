"""
Eye state detection using Eye Aspect Ratio (EAR).
"""

import logging
from typing import Dict, List, Tuple, Any

import numpy as np

from sim_bench.portrait_analysis.types import EyeState

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices for eyes
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_landmark_coords(landmark, image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Convert normalized landmark coordinates to pixel coordinates."""
    h, w = image_shape[:2]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return (x, y)


def calculate_eye_aspect_ratio(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_landmarks: List of 6 eye landmark coordinates [(x,y), ...]

    Returns:
        EAR value (typically 0.15-0.25 for open eyes, <0.15 for closed)
    """
    v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    h = euclidean_distance(eye_landmarks[0], eye_landmarks[3])

    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def detect_eye_state(
    face_landmarks,
    image_shape: Tuple[int, int],
    ear_threshold: float
) -> Dict[str, Any]:
    """
    Detect eye state (open/closed) from face mesh landmarks.

    Args:
        face_landmarks: MediaPipe face mesh landmarks
        image_shape: Image dimensions (height, width)
        ear_threshold: EAR threshold for open/closed decision

    Returns:
        Dict with eye state information
    """
    left_eye_coords = [
        get_landmark_coords(face_landmarks.landmark[i], image_shape)
        for i in LEFT_EYE_INDICES
    ]
    right_eye_coords = [
        get_landmark_coords(face_landmarks.landmark[i], image_shape)
        for i in RIGHT_EYE_INDICES
    ]

    left_ear = calculate_eye_aspect_ratio(left_eye_coords)
    right_ear = calculate_eye_aspect_ratio(right_eye_coords)

    return {
        'left_eye_open': left_ear >= ear_threshold,
        'right_eye_open': right_ear >= ear_threshold,
        'left_ear': left_ear,
        'right_ear': right_ear,
        'left_eye_coords': left_eye_coords,
        'right_eye_coords': right_eye_coords
    }


def create_eye_state(eye_data: Dict[str, Any]) -> EyeState:
    """Create EyeState dataclass from detection result."""
    return EyeState(
        left_eye_open=eye_data['left_eye_open'],
        right_eye_open=eye_data['right_eye_open'],
        left_ear=eye_data['left_ear'],
        right_ear=eye_data['right_ear']
    )
