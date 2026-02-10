"""InsightFace Score Eyes step - eye state scoring."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers
from sim_bench.pipeline.insightface_pipeline.face_cropper import InsightFaceCropper
from sim_bench.portrait_analysis.eye_state import detect_eye_state

logger = logging.getLogger(__name__)


class EyeStateScorer:
    """Strategy for computing eye state scores using MediaPipe on cropped faces."""

    def __init__(self, crop_margin: float = 0.3, target_size: int = 256, ear_threshold: float = 0.2):
        """
        Initialize the eye state scorer.

        Args:
            crop_margin: Margin around face bbox for cropping
            target_size: Size to resize cropped face
            ear_threshold: Eye Aspect Ratio threshold for open/closed
        """
        self.cropper = InsightFaceCropper(margin=crop_margin, target_size=target_size)
        self.ear_threshold = ear_threshold
        self._face_mesh = None

    def _get_face_mesh(self):
        """Lazy load MediaPipe Face Mesh."""
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        return self._face_mesh

    def compute_score(self, face_data: Dict[str, Any], image_path: Path) -> float:
        """
        Compute eye openness score from face data using MediaPipe.

        Args:
            face_data: InsightFace detection data with bbox
            image_path: Path to the original image

        Returns:
            Eye openness score 0-1 (1 = both eyes open, 0 = both closed)
        """
        # Crop face from image
        face_crop = self.cropper.crop_face_from_detection(image_path, face_data)
        if face_crop is None:
            logger.warning(f"Failed to crop face from {image_path}")
            return 0.5

        # Run MediaPipe Face Mesh on crop
        face_mesh = self._get_face_mesh()
        results = face_mesh.process(face_crop)

        if not results.multi_face_landmarks:
            logger.debug(f"No face landmarks found in crop from {image_path}")
            return 0.5

        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Detect eye state using existing utility
        eye_data = detect_eye_state(
            face_landmarks,
            face_crop.shape[:2],  # (height, width)
            self.ear_threshold
        )

        # Compute score: average of left and right EAR, normalized
        left_ear = eye_data['left_ear']
        right_ear = eye_data['right_ear']
        avg_ear = (left_ear + right_ear) / 2.0

        # Normalize EAR to 0-1 score
        # Typical EAR: 0.15-0.25 for open eyes, <0.15 for closed
        # Map 0.1 -> 0.0, 0.3 -> 1.0
        score = (avg_ear - 0.1) / 0.2
        return max(0.0, min(1.0, score))


@register_step
class InsightFaceScoreEyesStep(BaseStep):
    """Score eye state (open/closed) using MediaPipe on InsightFace face crops."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="insightface_score_eyes",
            display_name="InsightFace Score Eyes",
            description="Score eye state (open/closed) using MediaPipe Face Mesh on cropped faces.",
            category="people",
            requires=set(),
            produces={"face_eyes_scores"},  # Common interface
            depends_on=["insightface_detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_face_size": {
                        "type": "number",
                        "default": 50
                    },
                    "min_confidence": {
                        "type": "number",
                        "default": 0.5
                    },
                    "crop_margin": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Margin around face bbox for cropping (0.2-0.4)"
                    },
                    "target_size": {
                        "type": "integer",
                        "default": 256,
                        "description": "Size to resize cropped face"
                    },
                    "ear_threshold": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Eye Aspect Ratio threshold for open/closed"
                    }
                }
            }
        )
        self._scorer = None
        self._scorer_config = None
    
    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        """Get cache configuration for eye scoring."""
        all_faces = context.insightface_faces if hasattr(context, 'insightface_faces') else {}
        
        face_keys = []
        for image_path, face_data in all_faces.items():
            for face in face_data.get('faces', []):
                face_key = f"{image_path}:face_{face['face_index']}"
                face_keys.append(face_key)
        
        return {
            "items": face_keys if face_keys else ["dummy"],
            "feature_type": "insightface_eyes",
            "model_name": "insightface",
            "metadata": {}
        } if face_keys else None
    
    def _process_uncached(self, items: List[str], context: PipelineContext, config: dict) -> Dict[str, Dict[str, Any]]:
        """Process uncached items - score eye states."""
        all_faces = context.insightface_faces if hasattr(context, 'insightface_faces') else {}
        results = {}
        
        for i, face_key in enumerate(items):
            face_data = self._find_face(face_key, all_faces)
            score = self._compute_eyes_score(face_data, config) if face_data else 0.5
            results[face_key] = {'eyes_score': score}
            
            progress = (i + 1) / len(items)
            context.report_progress("insightface_score_eyes", progress, f"Scoring {i + 1}/{len(items)}")
        
        return results
    
    def _find_face(self, face_key: str, all_faces: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find face data by key, enriching with image path."""
        image_path = face_key.split(':face_')[0]
        face_index = int(face_key.split(':face_')[1])

        faces_data = all_faces.get(image_path, {}).get('faces', [])
        for face in faces_data:
            if face.get('face_index') == face_index:
                # Add original_path if not present (needed for cropping)
                if 'original_path' not in face:
                    face = dict(face)  # Copy to avoid modifying original
                    face['original_path'] = image_path
                return face
        return None
    
    def _get_scorer(self, config: dict) -> EyeStateScorer:
        """Get or create scorer with config."""
        crop_margin = config.get('crop_margin', 0.3)
        target_size = config.get('target_size', 256)
        ear_threshold = config.get('ear_threshold', 0.2)

        config_key = (crop_margin, target_size, ear_threshold)
        if self._scorer is None or self._scorer_config != config_key:
            self._scorer = EyeStateScorer(
                crop_margin=crop_margin,
                target_size=target_size,
                ear_threshold=ear_threshold
            )
            self._scorer_config = config_key
        return self._scorer

    def _compute_eyes_score(self, face_data: Dict[str, Any], config: dict) -> float:
        """Compute eyes score for face."""
        min_face_size = config.get('min_face_size', 50)
        min_confidence = config.get('min_confidence', 0.5)

        # Check face size and confidence
        bbox = face_data.get('bbox', {})
        face_size = min(bbox.get('w_px', 0), bbox.get('h_px', 0))
        confidence = face_data.get('confidence', 0.0)

        # Strategy: only score if face is reliable
        if face_size < min_face_size or confidence < min_confidence:
            return 0.5

        # Get image path from face data
        image_path = face_data.get('original_path')
        if image_path is None:
            return 0.5

        image_path = Path(image_path) if isinstance(image_path, str) else image_path

        # Use MediaPipe scorer
        scorer = self._get_scorer(config)
        return scorer.compute_score(face_data, image_path)
    
    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize eyes score to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to eyes score."""
        return Serializers.json_deserialize(data)
    
    def _store_results(self, context: PipelineContext, results: Dict[str, Dict[str, Any]], config: dict) -> None:
        """Store eyes scores in context (using standard attribute name)."""
        context.face_eyes_scores = {
            key: result['eyes_score']
            for key, result in results.items()
        }
        
        logger.info(f"Scored eyes for {len(context.face_eyes_scores)} faces")
