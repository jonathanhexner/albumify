"""Score Face Smile step - smile detection using MediaPipe."""

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
class ScoreFaceSmileStep(BaseStep):
    """Score smile using MediaPipe face mesh landmarks."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_face_smile",
            display_name="Score Face Smile",
            description="Detect smiles using MediaPipe face mesh. Useful for selecting happy expressions.",
            category="people",
            requires=set(),  # faces is optional - step skips if none
            produces={"face_smile_scores"},
            depends_on=["detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "width_threshold": {
                        "type": "number",
                        "default": 0.15,
                        "minimum": 0.05,
                        "maximum": 0.5,
                        "description": "Mouth width ratio threshold for smile detection"
                    },
                    "elevation_threshold": {
                        "type": "number",
                        "default": 0.005,
                        "minimum": 0.0,
                        "maximum": 0.05,
                        "description": "Mouth corner elevation threshold for smile detection"
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
            logger.info("MediaPipe face mesh loaded for smile detection")

    def _generate_cache_key(self, face) -> str:
        """Generate unique cache key for a face."""
        return f"{face.original_path}:face_{face.face_index}"

    def _compute_smile_scores(self, image: Image.Image, config: dict) -> Dict[str, Any]:
        """Compute smile scores from face image."""
        from sim_bench.portrait_analysis.smile_detection import detect_smile

        self._load_face_mesh()
        width_threshold = config.get('width_threshold', 0.15)
        elevation_threshold = config.get('elevation_threshold', 0.005)

        image_np = np.array(image)
        results = self._face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return {
                'smile_score': 0.0,
                'is_smiling': False,
                'mouth_width_ratio': 0.0,
                'corner_elevation': 0.0
            }

        landmarks = results.multi_face_landmarks[0]
        smile_data = detect_smile(
            landmarks,
            image_np.shape,
            width_threshold,
            elevation_threshold
        )

        return {
            'smile_score': float(smile_data['smile_score']),
            'is_smiling': bool(smile_data['is_smiling']),
            'mouth_width_ratio': float(smile_data['mouth_width_ratio']),
            'corner_elevation': float(smile_data['corner_elevation'])
        }

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for smile scoring."""
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
            "feature_type": "face_smile",
            "model_name": "mediapipe",
            "metadata": {
                "width_threshold": config.get("width_threshold", 0.15),
                "elevation_threshold": config.get("elevation_threshold", 0.005)
            }
        }

    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, Dict[str, Any]]:
        """Process uncached faces - detect smile state."""
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
            smile_data = self._compute_smile_scores(face.image, config)
            face.smile_score = smile_data['smile_score']
            face.is_smiling = smile_data['is_smiling']

            key = self._generate_cache_key(face)
            results[key] = smile_data

            progress = (i + 1) / len(uncached_faces)
            context.report_progress(
                "score_face_smile", progress,
                f"Scoring smiles {i + 1}/{len(uncached_faces)}"
            )

        return results

    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize smile data to JSON bytes."""
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to smile data."""
        return Serializers.json_deserialize(data)

    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, Dict[str, Any]],
        config: dict
    ) -> None:
        """Store smile scores in context and update face objects."""
        all_faces = []
        for faces_list in context.faces.values():
            all_faces.extend(faces_list)

        # Apply cached data to faces
        for face in all_faces:
            key = self._generate_cache_key(face)
            if key in results:
                smile_data = results[key]
                face.smile_score = smile_data['smile_score']
                face.is_smiling = smile_data['is_smiling']

        # Store scores in context
        context.face_smile_scores = {
            self._generate_cache_key(f): f.smile_score
            for f in all_faces if f.smile_score is not None
        }

        logger.info(f"Scored smiles for {len(context.face_smile_scores)} faces")
