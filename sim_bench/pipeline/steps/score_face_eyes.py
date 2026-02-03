"""Score Face Eyes step - eye openness detection using MediaPipe."""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class ScoreFaceEyesStep(BaseStep):
    """Score eye openness using MediaPipe face mesh landmarks."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_face_eyes",
            display_name="Score Face Eyes",
            description="Detect eye openness using MediaPipe face mesh. Useful for filtering blinks.",
            category="people",
            requires=set(),  # faces is optional - step skips if none
            produces={"face_eyes_scores"},
            depends_on=["detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "ear_threshold": {
                        "type": "number",
                        "default": 0.2,
                        "minimum": 0.1,
                        "maximum": 0.5,
                        "description": "Eye Aspect Ratio threshold for open eyes (higher = stricter)"
                    }
                }
            }
        )
        self._face_mesh = None

    def _load_face_mesh(self):
        """Lazy load MediaPipe face mesh."""
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe face mesh loaded for eye detection")

    def _generate_cache_key(self, face) -> str:
        """Generate unique cache key for a face."""
        return f"{face.original_path}:face_{face.face_index}"

    def _compute_eye_scores(self, image: Image.Image, config: dict) -> Dict[str, Any]:
        """Compute eye openness scores from face image."""
        from sim_bench.portrait_analysis.eye_state import detect_eye_state

        self._load_face_mesh()
        ear_threshold = config.get('ear_threshold', 0.2)

        image_np = np.array(image)
        results = self._face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return {
                'eyes_open_score': 0.0,
                'left_eye_open': False,
                'right_eye_open': False,
                'both_eyes_open': False,
                'left_ear': 0.0,
                'right_ear': 0.0
            }

        landmarks = results.multi_face_landmarks[0]
        eye_data = detect_eye_state(landmarks, image_np.shape, ear_threshold)

        # Normalize EAR to 0-1 score
        left_score = min(1.0, eye_data['left_ear'] / ear_threshold)
        right_score = min(1.0, eye_data['right_ear'] / ear_threshold)
        eyes_open_score = (left_score + right_score) / 2.0

        return {
            'eyes_open_score': float(eyes_open_score),
            'left_eye_open': bool(eye_data['left_eye_open']),
            'right_eye_open': bool(eye_data['right_eye_open']),
            'both_eyes_open': bool(eye_data['left_eye_open'] and eye_data['right_eye_open']),
            'left_ear': float(eye_data['left_ear']),
            'right_ear': float(eye_data['right_ear'])
        }

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for eye scoring."""
        if not context.faces:
            return None

        all_faces = []
        for faces_list in context.faces.values():
            all_faces.extend(faces_list)

        if not all_faces:
            return None

        cache_keys = [self._generate_cache_key(f) for f in all_faces]

        return {
            "items": cache_keys,
            "feature_type": "face_eyes",
            "model_name": "mediapipe",
            "metadata": {"ear_threshold": config.get("ear_threshold", 0.2)}
        }

    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, Dict[str, Any]]:
        """Process uncached faces - detect eye state."""
        all_faces = []
        for faces_list in context.faces.values():
            all_faces.extend(faces_list)

        key_to_face = {
            self._generate_cache_key(f): f
            for f in all_faces
        }

        uncached_faces = [key_to_face[key] for key in items if key in key_to_face]

        if not uncached_faces:
            return {}

        results = {}

        for i, face in enumerate(uncached_faces):
            eye_data = self._compute_eye_scores(face.image, config)
            face.eyes_open_score = eye_data['eyes_open_score']
            face.both_eyes_open = eye_data['both_eyes_open']

            key = self._generate_cache_key(face)
            results[key] = eye_data

            progress = (i + 1) / len(uncached_faces)
            context.report_progress(
                "score_face_eyes", progress,
                f"Scoring eyes {i + 1}/{len(uncached_faces)}"
            )

        return results

    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize eye data to JSON bytes."""
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to eye data."""
        return Serializers.json_deserialize(data)

    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, Dict[str, Any]],
        config: dict
    ) -> None:
        """Store eye scores in context and update face objects."""
        all_faces = []
        for faces_list in context.faces.values():
            all_faces.extend(faces_list)

        # Apply cached data to faces
        for face in all_faces:
            key = self._generate_cache_key(face)
            if key in results:
                eye_data = results[key]
                face.eyes_open_score = eye_data['eyes_open_score']
                face.both_eyes_open = eye_data['both_eyes_open']

        # Store scores in context
        context.face_eyes_scores = {
            self._generate_cache_key(f): f.eyes_open_score
            for f in all_faces if f.eyes_open_score is not None
        }

        logger.info(f"Scored eyes for {len(context.face_eyes_scores)} faces")
