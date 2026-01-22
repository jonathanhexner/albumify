"""
Data types for the unified Model Hub.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ImageMetrics:
    """
    Unified image metrics from all models.

    Combines quality scores, aesthetic scores, and portrait metrics
    into a single data structure.
    """
    image_path: str

    # Technical quality (IQA - 0 to 1 scale)
    iqa_score: Optional[float] = None
    sharpness: Optional[float] = None
    exposure: Optional[float] = None
    colorfulness: Optional[float] = None
    contrast: Optional[float] = None

    # Aesthetic quality (AVA - 1 to 10 scale)
    ava_score: Optional[float] = None

    # Portrait metrics (MediaPipe)
    has_face: bool = False
    num_faces: int = 0
    is_portrait: bool = False
    eyes_open: Optional[bool] = None
    is_smiling: Optional[bool] = None
    smile_score: Optional[float] = None
    eye_aspect_ratio: Optional[float] = None

    # Scene embedding for clustering
    scene_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    # Cluster assignment (populated after clustering)
    cluster_id: Optional[int] = None

    def get_composite_score(
        self,
        ava_weight: float = 0.5,
        iqa_weight: float = 0.2,
        portrait_weight: float = 0.3
    ) -> float:
        """
        Calculate weighted composite score.

        Args:
            ava_weight: Weight for aesthetic score
            iqa_weight: Weight for technical quality
            portrait_weight: Weight for portrait metrics

        Returns:
            Composite score (0-1 scale)
        """
        ava_norm = ((self.ava_score or 5.0) - 1) / 9 if self.ava_score else 0.5
        iqa = self.iqa_score or 0.5
        portrait = self._compute_portrait_score()

        return ava_weight * ava_norm + iqa_weight * iqa + portrait_weight * portrait

    def _compute_portrait_score(self) -> float:
        """Compute portrait-specific bonus score."""
        if not self.is_portrait:
            return 0.5

        score = 0.5
        if self.eyes_open:
            score += 0.25
        if self.is_smiling and self.smile_score:
            score += 0.25 * self.smile_score

        return min(1.0, score)
