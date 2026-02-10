"""Pluggable scoring strategies for image quality assessment."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PenaltyComputer:
    """Compute penalties from context data."""
    
    def __init__(self, config: Dict[str, Any]):
        penalties_config = config.get('penalties', {})
        self.body_penalty_weight = penalties_config.get('body_orientation', 0.1)
        self.occlusion_penalty_weight = penalties_config.get('face_occlusion', 0.2)
        self.eyes_penalty_weight = penalties_config.get('eyes_closed', 0.15)
        self.smile_penalty_weight = penalties_config.get('no_smile', 0.1)
        self.pose_penalty_weight = penalties_config.get('face_turned', 0.15)
    
    def compute_body_penalty(self, person_data: Dict[str, Any]) -> float:
        """Compute body orientation penalty."""
        body_facing_score = person_data.get('body_facing_score', 1.0)
        return self.body_penalty_weight * (1.0 - body_facing_score)
    
    def compute_occlusion_penalty(self) -> float:
        """Compute face occlusion penalty."""
        return self.occlusion_penalty_weight
    
    def compute_eyes_penalty(self, eyes_score: float) -> float:
        """Compute eyes closed penalty."""
        return self.eyes_penalty_weight * (1.0 - eyes_score)
    
    def compute_smile_penalty(self, smile_score: float) -> float:
        """Compute no-smile penalty."""
        return self.smile_penalty_weight * (1.0 - smile_score)
    
    def compute_pose_penalty(self, pose_score: float) -> float:
        """Compute face turned away penalty."""
        return self.pose_penalty_weight * (1.0 - pose_score)


class ScoringStrategy(ABC):
    """Base class for image scoring strategies."""
    
    @abstractmethod
    def compute_score(self, image_path: str, context: PipelineContext, config: Dict[str, Any]) -> float:
        """Compute final score for an image."""
        pass


class InsightFacePenaltyScoring(ScoringStrategy):
    """Penalty-based scoring using InsightFace attributes."""
    
    def compute_score(self, image_path: str, context: PipelineContext, config: Dict[str, Any]) -> float:
        """Apply penalties to base AVA score."""
        # Get base score (AVA) - already normalized to 0-1 at storage time
        base_score = context.ava_scores.get(image_path, 0.5)
        
        # Create penalty computer
        penalty_computer = PenaltyComputer(config)
        
        # Compute penalties
        penalties = self._compute_all_penalties(image_path, context, penalty_computer)
        
        # Apply penalties
        final_score = base_score - penalties
        return max(0.0, min(1.0, final_score))
    
    def _compute_all_penalties(self, image_path: str, context: PipelineContext, penalty_computer: PenaltyComputer) -> float:
        """Compute sum of all penalties."""
        total_penalties = 0.0

        # Person penalties
        persons = getattr(context, 'persons', {})
        person_data = persons.get(image_path, {})
        person_detected = person_data.get('person_detected', False)
        
        # Strategy: delegate to person penalty computer or no-person penalty computer
        penalty_strategy = PersonPenaltyStrategy() if person_detected else NoPersonPenaltyStrategy()
        total_penalties += penalty_strategy.compute(image_path, context, penalty_computer)
        
        return total_penalties


class PersonPenaltyStrategy:
    """Compute penalties when person is detected."""

    def compute(self, image_path: str, context: PipelineContext, penalty_computer: PenaltyComputer) -> float:
        """Compute penalties for person-related attributes."""
        penalties = 0.0

        persons = getattr(context, 'persons', {})
        person_data = persons.get(image_path, {})

        # Body orientation penalty
        penalties += penalty_computer.compute_body_penalty(person_data)

        # Face penalties (delegate to face strategy)
        insightface_faces = getattr(context, 'insightface_faces', {})
        faces = insightface_faces.get(image_path, {}).get('faces', [])
        face_strategy = FacePenaltyStrategy() if faces else OccludedFacePenaltyStrategy()
        penalties += face_strategy.compute(image_path, context, penalty_computer)
        
        return penalties


class NoPersonPenaltyStrategy:
    """No penalties when person not detected."""
    
    def compute(self, image_path: str, context: PipelineContext, penalty_computer: PenaltyComputer) -> float:
        """No person-related penalties."""
        return 0.0


class FacePenaltyStrategy:
    """Compute penalties when face is visible."""

    def compute(self, image_path: str, context: PipelineContext, penalty_computer: PenaltyComputer) -> float:
        """Compute face attribute penalties."""
        penalties = 0.0

        insightface_faces = getattr(context, 'insightface_faces', {})
        faces = insightface_faces.get(image_path, {}).get('faces', [])
        face = faces[0] if faces else None
        face_key = f"{image_path}:face_{face.get('face_index', 0)}" if face else None
        
        # Eyes penalty (using standard attribute name)
        eyes_score = context.face_eyes_scores.get(face_key, 0.5)
        penalties += penalty_computer.compute_eyes_penalty(eyes_score)
        
        # Smile penalty (using standard attribute name)
        smile_score = context.face_smile_scores.get(face_key, 0.5)
        penalties += penalty_computer.compute_smile_penalty(smile_score)
        
        # Pose penalty (using standard attribute name)
        pose_score = context.face_pose_scores.get(face_key, 0.5)
        penalties += penalty_computer.compute_pose_penalty(pose_score)
        
        return penalties


class OccludedFacePenaltyStrategy:
    """Compute penalty when face is occluded."""
    
    def compute(self, image_path: str, context: PipelineContext, penalty_computer: PenaltyComputer) -> float:
        """Return occlusion penalty."""
        return penalty_computer.compute_occlusion_penalty()


class ScoringStrategyFactory:
    """Factory for creating scoring strategies."""
    
    _strategies = {
        'insightface_penalty': InsightFacePenaltyScoring,
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> ScoringStrategy:
        """Create scoring strategy by name."""
        strategy_class = cls._strategies.get(strategy_name, InsightFacePenaltyScoring)
        logger.info(f"Created scoring strategy: {strategy_name}")
        return strategy_class()
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a new scoring strategy."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered scoring strategy: {name}")
