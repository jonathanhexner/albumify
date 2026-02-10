"""Person penalty computation for portrait quality scoring."""

import logging
from typing import Dict, Optional

from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PersonPenaltyConfig:
    """Configuration for person penalty computation."""

    def __init__(
        self,
        face_occlusion: float = -0.3,
        body_not_facing: float = -0.1,
        eyes_closed: float = -0.15,
        not_smiling: float = -0.05,
        face_turned: float = -0.1,
        max_penalty: float = -0.7,
        body_facing_threshold: float = 0.5,
        eyes_threshold: float = 0.5,
        smile_threshold: float = 0.5,
        pose_threshold: float = 0.5,
    ):
        """
        Initialize penalty configuration.

        Args:
            face_occlusion: Penalty for no face detected
            body_not_facing: Penalty for body not facing camera
            eyes_closed: Penalty for eyes closed
            not_smiling: Penalty for not smiling
            face_turned: Penalty for face turned away
            max_penalty: Maximum cumulative penalty cap
            body_facing_threshold: Threshold for body facing (below = penalty)
            eyes_threshold: Threshold for eyes open (below = penalty)
            smile_threshold: Threshold for smiling (below = penalty)
            pose_threshold: Threshold for face pose (below = penalty)
        """
        self.face_occlusion = face_occlusion
        self.body_not_facing = body_not_facing
        self.eyes_closed = eyes_closed
        self.not_smiling = not_smiling
        self.face_turned = face_turned
        self.max_penalty = max_penalty
        self.body_facing_threshold = body_facing_threshold
        self.eyes_threshold = eyes_threshold
        self.smile_threshold = smile_threshold
        self.pose_threshold = pose_threshold


class PersonPenaltyComputer:
    """Compute additive person penalties for portrait images."""

    def __init__(self, config: PersonPenaltyConfig):
        """
        Initialize penalty computer.

        Args:
            config: Penalty configuration
        """
        self.config = config
        logger.info(
            f"PersonPenaltyComputer initialized: "
            f"face_occlusion={config.face_occlusion}, "
            f"body={config.body_not_facing}, "
            f"eyes={config.eyes_closed}, "
            f"smile={config.not_smiling}, "
            f"pose={config.face_turned}, "
            f"max_penalty={config.max_penalty}"
        )

    def compute_penalty(self, image_path: str, context: PipelineContext) -> float:
        """
        Compute additive person penalty for image.

        Args:
            image_path: Path to the image
            context: Pipeline context with person/face scores

        Returns:
            Penalty value in [max_penalty, 0.0], where 0.0 = no penalty
        """
        person_detected = self._check_person_detected(image_path, context)
        if not person_detected:
            return 0.0

        face_detected = self._check_face_detected(image_path, context)
        if not face_detected:
            return self.config.face_occlusion

        penalty = self._accumulate_penalties(image_path, context)
        capped_penalty = max(penalty, self.config.max_penalty)
        return capped_penalty

    def _check_person_detected(
        self, image_path: str, context: PipelineContext
    ) -> bool:
        """Check if person is detected in image."""
        path_normalized = image_path.replace('\\', '/')
        persons = context.persons.get(path_normalized, {})
        if not persons:
            persons = context.persons.get(image_path, {})
        return persons.get("person_detected", False)

    def _check_face_detected(self, image_path: str, context: PipelineContext) -> bool:
        """Check if any face is detected in image."""
        path_normalized = image_path.replace('\\', '/')
        # Check if any face scores exist for this image
        for key in context.face_eyes_scores:
            if key.startswith(path_normalized + ":face_"):
                return True
        for key in context.face_smile_scores:
            if key.startswith(path_normalized + ":face_"):
                return True
        for key in context.face_pose_scores:
            if key.startswith(path_normalized + ":face_"):
                return True
        return False

    def _get_face_count(self, image_path: str, context: PipelineContext) -> int:
        """Get number of detected faces for image."""
        path_normalized = image_path.replace('\\', '/')
        # Check InsightFace faces
        if hasattr(context, 'insightface_faces') and context.insightface_faces:
            face_data = context.insightface_faces.get(path_normalized, {})
            if not face_data:
                face_data = context.insightface_faces.get(image_path, {})
            return len(face_data.get('faces', []))
        # Check MediaPipe faces
        faces = context.faces.get(image_path, [])
        return len(faces)

    def _accumulate_penalties(
        self, image_path: str, context: PipelineContext
    ) -> float:
        """Accumulate all applicable penalties using WORST score from all faces."""
        total_penalty = 0.0

        total_penalty += self._compute_body_penalty(image_path, context)
        # For face-specific penalties, find the WORST score across all faces
        total_penalty += self._compute_eyes_penalty(image_path, context)
        total_penalty += self._compute_smile_penalty(image_path, context)
        total_penalty += self._compute_pose_penalty(image_path, context)

        return total_penalty

    def _compute_body_penalty(
        self, image_path: str, context: PipelineContext
    ) -> float:
        """Compute penalty for body not facing camera."""
        path_normalized = image_path.replace('\\', '/')
        persons = context.persons.get(path_normalized, {})
        if not persons:
            persons = context.persons.get(image_path, {})
        body_facing_score = persons.get("body_facing_score", 1.0)

        body_not_facing = body_facing_score < self.config.body_facing_threshold
        penalty = self.config.body_not_facing if body_not_facing else 0.0
        return penalty

    def _compute_eyes_penalty(
        self, image_path: str, context: PipelineContext
    ) -> float:
        """Compute penalty for eyes closed - uses WORST score from all faces."""
        path_normalized = image_path.replace('\\', '/')
        face_count = self._get_face_count(image_path, context)

        # Find the worst (lowest) eyes score across all faces
        worst_eyes_score = 1.0
        for i in range(max(face_count, 10)):  # Check up to 10 faces
            face_key = f"{path_normalized}:face_{i}"
            if face_key in context.face_eyes_scores:
                score = context.face_eyes_scores[face_key]
                worst_eyes_score = min(worst_eyes_score, score)

        eyes_closed = worst_eyes_score < self.config.eyes_threshold
        penalty = self.config.eyes_closed if eyes_closed else 0.0
        return penalty

    def _compute_smile_penalty(
        self, image_path: str, context: PipelineContext
    ) -> float:
        """Compute penalty for not smiling - uses WORST score from all faces."""
        path_normalized = image_path.replace('\\', '/')
        face_count = self._get_face_count(image_path, context)

        # Find the worst (lowest) smile score across all faces
        worst_smile_score = 1.0
        for i in range(max(face_count, 10)):  # Check up to 10 faces
            face_key = f"{path_normalized}:face_{i}"
            if face_key in context.face_smile_scores:
                score = context.face_smile_scores[face_key]
                worst_smile_score = min(worst_smile_score, score)

        not_smiling = worst_smile_score < self.config.smile_threshold
        penalty = self.config.not_smiling if not_smiling else 0.0
        return penalty

    def _compute_pose_penalty(
        self, image_path: str, context: PipelineContext
    ) -> float:
        """Compute penalty for face turned away - uses WORST score from all faces."""
        path_normalized = image_path.replace('\\', '/')
        face_count = self._get_face_count(image_path, context)

        # Find the worst (lowest) pose score across all faces
        worst_pose_score = 1.0
        for i in range(max(face_count, 10)):  # Check up to 10 faces
            face_key = f"{path_normalized}:face_{i}"
            if face_key in context.face_pose_scores:
                score = context.face_pose_scores[face_key]
                worst_pose_score = min(worst_pose_score, score)

        face_turned = worst_pose_score < self.config.pose_threshold
        penalty = self.config.face_turned if face_turned else 0.0
        return penalty


class PersonPenaltyFactory:
    """Factory for creating person penalty computer."""

    @staticmethod
    def create(config: Dict) -> PersonPenaltyComputer:
        """
        Create person penalty computer from config.

        Args:
            config: Configuration dictionary

        Returns:
            PersonPenaltyComputer instance
        """
        penalty_config = PersonPenaltyConfig(
            face_occlusion=config.get("face_occlusion", -0.3),
            body_not_facing=config.get("body_not_facing", -0.1),
            eyes_closed=config.get("eyes_closed", -0.15),
            not_smiling=config.get("not_smiling", -0.05),
            face_turned=config.get("face_turned", -0.1),
            max_penalty=config.get("max_penalty", -0.7),
            body_facing_threshold=config.get("body_facing_threshold", 0.5),
            eyes_threshold=config.get("eyes_threshold", 0.5),
            smile_threshold=config.get("smile_threshold", 0.5),
            pose_threshold=config.get("pose_threshold", 0.5),
        )

        computer = PersonPenaltyComputer(penalty_config)
        return computer
