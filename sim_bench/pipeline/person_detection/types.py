"""Person detection data types."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: float
    y: float
    w: float
    h: float
    x_px: int
    y_px: int
    w_px: int
    h_px: int


@dataclass
class PersonDetection:
    """Person detection result with body orientation."""
    bbox: BoundingBox
    confidence: float
    keypoints: np.ndarray  # (17, 3) - x, y, confidence
    body_facing_score: float  # 0-1, where 1.0 = front-facing
    keypoint_confidence: float  # Average confidence of key keypoints
    
    def is_reliable(self, min_confidence: float = 0.5) -> bool:
        """Check if detection is reliable based on keypoint confidence."""
        return self.keypoint_confidence >= min_confidence
