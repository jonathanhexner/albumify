"""
Data types for portrait analysis results.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EyeState:
    """Eye state detection result."""
    left_eye_open: bool
    right_eye_open: bool
    left_ear: float  # Eye Aspect Ratio
    right_ear: float
    both_eyes_open: bool = field(init=False)

    def __post_init__(self):
        self.both_eyes_open = self.left_eye_open and self.right_eye_open


@dataclass
class SmileState:
    """Smile detection result."""
    is_smiling: bool
    smile_score: float  # 0-1 normalized
    mouth_width_ratio: float
    corner_elevation: float


@dataclass
class PortraitMetrics:
    """Complete portrait analysis result for an image."""
    image_path: str
    has_face: bool
    num_faces: int
    is_portrait: bool
    face_ratio: Optional[float]
    center_offset: Optional[float]
    eye_state: Optional[EyeState]
    smile_state: Optional[SmileState]
    confidence: float
