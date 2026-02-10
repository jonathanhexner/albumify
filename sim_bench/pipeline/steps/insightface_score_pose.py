"""InsightFace Score Pose step - face pose scoring."""

import logging
from typing import Dict, List, Any, Optional

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


class FacePoseScorer:
    """Strategy for computing face pose scores from 5-point landmarks."""

    def compute_score(self, face_data: Dict[str, Any]) -> float:
        """
        Compute frontal face score from InsightFace 5-point landmarks.

        The landmarks array has shape (5, 2) with points:
        [0] left_eye, [1] right_eye, [2] nose, [3] mouth_left, [4] mouth_right

        Algorithm:
        1. Compute eye center and eye vector
        2. Compute perpendicular distance of nose from eye line
        3. Normalize by eye distance to get yaw estimate
        4. Frontal score = 1 - abs(normalized_yaw), clamped to [0, 1]

        Returns:
            Float score 0-1 (1 = frontal, 0 = profile)
        """
        landmarks = face_data.get('landmarks')
        if landmarks is None:
            return 0.5

        # Convert to numpy array if needed
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)

        if landmarks.shape != (5, 2):
            logger.warning(f"Unexpected landmarks shape: {landmarks.shape}, expected (5, 2)")
            return 0.5

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        # Compute eye center
        eye_center = (left_eye + right_eye) / 2.0

        # Compute eye vector
        eye_vector = right_eye - left_eye
        eye_distance = np.linalg.norm(eye_vector)

        if eye_distance < 1e-6:
            return 0.5

        # Normalize eye vector
        eye_unit = eye_vector / eye_distance

        # Vector from eye center to nose
        nose_vector = nose - eye_center

        # Perpendicular component (cross product in 2D gives signed area)
        # Positive = nose is below eye line, which is expected for frontal
        nose_to_line_perpendicular = nose_vector[1] - (nose_vector[0] * eye_unit[1] / eye_unit[0] if abs(eye_unit[0]) > 1e-6 else 0)

        # Horizontal deviation from center (yaw indicator)
        # Project nose vector onto eye vector direction
        nose_along_eye = np.dot(nose_vector, eye_unit)

        # Normalize by eye distance - if nose is centered, this should be ~0
        normalized_yaw = nose_along_eye / eye_distance

        # Frontal score: 1 when centered, decreases as face turns
        # Typically |normalized_yaw| < 0.3 for frontal faces
        frontal_score = 1.0 - min(abs(normalized_yaw) * 2.0, 1.0)

        # Clamp to valid range
        return max(0.0, min(1.0, frontal_score))


@register_step
class InsightFaceScorePoseStep(BaseStep):
    """Score face pose (yaw/pitch/roll) using InsightFace attributes."""
    
    def __init__(self):
        self._metadata = StepMetadata(
            name="insightface_score_pose",
            display_name="InsightFace Score Pose",
            description="Score face pose (frontal vs turned away) using InsightFace.",
            category="people",
            requires=set(),
            produces={"face_pose_scores"},  # Common interface
            depends_on=["insightface_detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "use_sixdrepnet": {
                        "type": "boolean",
                        "default": True
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "mps"],
                        "default": "cpu"
                    }
                }
            }
        )
        self._scorer = FacePoseScorer()
    
    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        """Get cache configuration for pose scoring."""
        all_faces = context.insightface_faces if hasattr(context, 'insightface_faces') else {}
        
        face_keys = []
        for image_path, face_data in all_faces.items():
            for face in face_data.get('faces', []):
                face_key = f"{image_path}:face_{face['face_index']}"
                face_keys.append(face_key)
        
        return {
            "items": face_keys if face_keys else ["dummy"],
            "feature_type": "insightface_pose",
            "model_name": "insightface",
            "metadata": {"device": config.get("device", "cpu")}
        } if face_keys else None
    
    def _process_uncached(self, items: List[str], context: PipelineContext, config: dict) -> Dict[str, Dict[str, Any]]:
        """Process uncached items - score face poses."""
        all_faces = context.insightface_faces if hasattr(context, 'insightface_faces') else {}
        results = {}
        
        for i, face_key in enumerate(items):
            face_data = self._find_face(face_key, all_faces)
            score = self._compute_pose_score(face_data, config) if face_data else 0.5
            results[face_key] = {'pose_score': score}
            
            progress = (i + 1) / len(items)
            context.report_progress("insightface_score_pose", progress, f"Scoring {i + 1}/{len(items)}")
        
        return results
    
    def _find_face(self, face_key: str, all_faces: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find face data by key."""
        image_path = face_key.split(':face_')[0]
        face_index = int(face_key.split(':face_')[1])
        
        faces_data = all_faces.get(image_path, {}).get('faces', [])
        for face in faces_data:
            if face.get('face_index') == face_index:
                return face
        return None
    
    def _compute_pose_score(self, face_data: Dict[str, Any], config: dict) -> float:
        """Compute pose score for face."""
        confidence = face_data.get('confidence', 0.0)
        min_confidence = 0.5
        
        # Strategy: delegate to scorer if face is reliable
        reliable = confidence >= min_confidence
        scorer = self._scorer if reliable else NeutralScorer()
        return scorer.compute_score(face_data)
    
    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize pose score to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to pose score."""
        return Serializers.json_deserialize(data)
    
    def _store_results(self, context: PipelineContext, results: Dict[str, Dict[str, Any]], config: dict) -> None:
        """Store pose scores in context (using standard attribute name)."""
        context.face_pose_scores = {
            key: result['pose_score']
            for key, result in results.items()
        }
        
        logger.info(f"Scored pose for {len(context.face_pose_scores)} faces")


class NeutralScorer:
    """Neutral scorer for unreliable faces."""
    
    def compute_score(self, face_data: Dict[str, Any]) -> float:
        """Return neutral score."""
        return 0.5
