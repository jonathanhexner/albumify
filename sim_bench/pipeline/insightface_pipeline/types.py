"""InsightFace detection data types."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from sim_bench.pipeline.person_detection.types import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class InsightFaceDetection:
    """Face detection result from InsightFace."""
    original_path: Path
    face_index: int
    bbox: BoundingBox
    confidence: float
    landmarks: np.ndarray  # (5, 2) - left_eye, right_eye, nose, mouth_left, mouth_right
    person_bbox: Optional[BoundingBox]  # Associated person
    face_occluded: bool  # Person exists but face not found
    
    def get_key(self) -> str:
        """Get unique key for this face detection."""
        return f"{self.original_path}:face_{self.face_index}"
