"""Body orientation inference from keypoints using strategy pattern."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from sim_bench.pipeline.person_detection.types import PersonDetection

logger = logging.getLogger(__name__)


class BodyOrientationStrategy(ABC):
    """Base class for body orientation inference."""
    
    @abstractmethod
    def compute_facing_score(self, person: PersonDetection) -> float:
        """Compute body facing score (0-1, where 1.0 = front-facing)."""
        pass


class ShoulderHipOrientationStrategy(BodyOrientationStrategy):
    """Infer body orientation from shoulder and hip keypoints."""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_keypoint_confidence = config.get('keypoint_confidence_threshold', 0.5)
        logger.info(f"ShoulderHipOrientationStrategy initialized with min_confidence={self.min_keypoint_confidence}")
    
    def compute_facing_score(self, person: PersonDetection) -> float:
        """
        Compute facing score using shoulder and hip symmetry.
        
        Front-facing: shoulders and hips visible symmetrically
        Side: asymmetric visibility
        Back: shoulders close together or low confidence
        """
        keypoints = person.keypoints
        
        # Keypoint indices (COCO format)
        left_shoulder_idx = 5
        right_shoulder_idx = 6
        left_hip_idx = 11
        right_hip_idx = 12
        
        # Extract keypoints
        left_shoulder = keypoints[left_shoulder_idx]
        right_shoulder = keypoints[right_shoulder_idx]
        left_hip = keypoints[left_hip_idx]
        right_hip = keypoints[right_hip_idx]
        
        # Check confidence (return neutral score for unreliable detections)
        min_conf = min(left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2])
        reliable = min_conf >= self.min_keypoint_confidence
        
        # Strategy pattern: delegate to confidence-based scorer
        scorer = ReliableScorer() if reliable else UnreliableScorer()
        return scorer.compute_score(left_shoulder, right_shoulder, left_hip, right_hip)


class ReliableScorer:
    """Score computation for reliable keypoints."""
    
    def compute_score(self, left_shoulder, right_shoulder, left_hip, right_hip) -> float:
        """Compute score based on shoulder/hip symmetry."""
        # Shoulder width
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Hip width
        hip_width = abs(right_hip[0] - left_hip[0])
        
        # Symmetry score (normalized)
        symmetry = min(shoulder_width, hip_width) / (max(shoulder_width, hip_width) + 1e-6)
        
        # Front-facing: high symmetry (both shoulders/hips visible)
        # Back: low symmetry (shoulders close together)
        return float(symmetry)


class UnreliableScorer:
    """Neutral score for unreliable keypoints."""
    
    def compute_score(self, *args) -> float:
        """Return neutral score when keypoints are unreliable."""
        return 0.5  # Neutral - don't penalize or reward


class BodyOrientationFactory:
    """Factory for creating body orientation strategies."""
    
    _strategies = {
        'shoulder_hip': ShoulderHipOrientationStrategy,
    }
    
    @classmethod
    def create(cls, strategy_name: str, config: Dict[str, Any]) -> BodyOrientationStrategy:
        """Create body orientation strategy by name."""
        strategy_class = cls._strategies.get(strategy_name, ShoulderHipOrientationStrategy)
        return strategy_class(config)
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register new body orientation strategy."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered body orientation strategy: {name}")
