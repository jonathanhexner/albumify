"""Score Face Quality step - pose, eyes, smile, sharpness assessment."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class ScoreFaceQualityStep(BaseStep):
    """Score face quality (pose, eyes open, smile, sharpness) using MediaPipe and SixDRepNet."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_face_quality",
            display_name="Score Face Quality",
            description="Assess face quality: frontal pose, eyes open, smile, and sharpness.",
            category="people",
            requires={"faces"},
            produces={"face_pose_scores", "face_eyes_scores", "face_smile_scores"},
            depends_on=["detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "eye_open_ear_threshold": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Eye Aspect Ratio threshold for open eyes"
                    },
                    "smile_width_threshold": {
                        "type": "number",
                        "default": 0.15,
                        "description": "Mouth width threshold for smile detection"
                    },
                    "smile_elevation_threshold": {
                        "type": "number",
                        "default": 0.005,
                        "description": "Mouth corner elevation threshold"
                    },
                    "sharpness_norm": {
                        "type": "number",
                        "default": 1000.0,
                        "description": "Normalization factor for sharpness"
                    }
                }
            }
        )
        self._quality_scorer = None

    def _get_scorer(self, config: dict):
        """Lazy load quality scorer."""
        if self._quality_scorer is None:
            from sim_bench.face_pipeline.quality_scorer import FaceQualityScorer
            logger.info("Loading face quality scorer (MediaPipe + SixDRepNet)")
            full_config = {
                'face_pipeline': config,
                'portrait_analysis': config
            }
            self._quality_scorer = FaceQualityScorer(full_config)
        return self._quality_scorer

    def _generate_cache_key(self, face) -> str:
        """Generate unique cache key for a face."""
        return f"{face.original_path}:face_{face.face_index}"

    def _serialize_quality(self, face) -> Dict[str, Any]:
        """Convert face quality data to JSON-serializable dict."""
        return {
            'pose': {
                'yaw': face.pose.yaw,
                'pitch': face.pose.pitch,
                'roll': face.pose.roll,
                'frontal_score': face.pose.frontal_score
            } if face.pose else None,
            'eyes_open_score': face.eyes_open_score,
            'both_eyes_open': face.both_eyes_open,
            'smile_score': face.smile_score,
            'is_smiling': face.is_smiling,
            'quality': {
                'pose_score': face.quality.pose_score,
                'eyes_open_score': face.quality.eyes_open_score,
                'smile_score': face.quality.smile_score,
                'sharpness_score': face.quality.sharpness_score,
                'detection_confidence': face.quality.detection_confidence,
                'overall': face.quality.overall
            } if face.quality else None
        }

    def _apply_quality_from_cache(self, face, cached_data: Dict[str, Any]):
        """Apply cached quality data to a face."""
        from sim_bench.face_pipeline.types import PoseEstimate, FaceQualityScore
        
        if cached_data.get('pose'):
            pose_data = cached_data['pose']
            face.pose = PoseEstimate(
                yaw=pose_data['yaw'],
                pitch=pose_data['pitch'],
                roll=pose_data['roll']
            )
        
        face.eyes_open_score = cached_data.get('eyes_open_score')
        face.both_eyes_open = cached_data.get('both_eyes_open')
        face.smile_score = cached_data.get('smile_score')
        face.is_smiling = cached_data.get('is_smiling')
        
        if cached_data.get('quality'):
            quality_data = cached_data['quality']
            face.quality = FaceQualityScore(
                pose_score=quality_data['pose_score'],
                eyes_open_score=quality_data['eyes_open_score'],
                smile_score=quality_data['smile_score'],
                sharpness_score=quality_data['sharpness_score'],
                detection_confidence=quality_data['detection_confidence']
            )

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for face quality scoring."""
        if not context.faces:
            return None
        
        # Flatten all faces from all images
        all_faces = []
        for faces_list in context.faces.values():
            all_faces.extend(faces_list)
        
        if not all_faces:
            return None
        
        # Use composite cache keys as items
        cache_keys = [self._generate_cache_key(f) for f in all_faces]
        
        return {
            "items": cache_keys,
            "feature_type": "face_quality",
            "model_name": "mediapipe_sixdrepnet",
            "metadata": {}
        }
    
    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, Dict[str, Any]]:
        """Process uncached items - score face quality."""
        # Map cache keys back to faces
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
        
        scorer = self._get_scorer(config)
        results = {}
        
        for i, face in enumerate(uncached_faces):
            scorer.score_face(face)
            key = self._generate_cache_key(face)
            results[key] = self._serialize_quality(face)
            
            progress = (i + 1) / len(uncached_faces)
            context.report_progress(
                "score_face_quality", progress,
                f"Scoring {i + 1}/{len(uncached_faces)}"
            )
        
        return results
    
    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize quality dict to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to quality dict."""
        return Serializers.json_deserialize(data)
    
    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, Dict[str, Any]],
        config: dict
    ) -> None:
        """Store face quality scores in context and update face objects."""
        # Get all faces
        all_faces = []
        for faces_list in context.faces.values():
            all_faces.extend(faces_list)
        
        # Apply cached data to faces
        for face in all_faces:
            key = self._generate_cache_key(face)
            if key in results:
                self._apply_quality_from_cache(face, results[key])
        
        # Store aggregate scores in context
        context.face_pose_scores = {
            self._generate_cache_key(f): f.pose.frontal_score
            for f in all_faces if f.pose
        }
        context.face_eyes_scores = {
            self._generate_cache_key(f): f.eyes_open_score
            for f in all_faces if f.eyes_open_score is not None
        }
        context.face_smile_scores = {
            self._generate_cache_key(f): f.smile_score
            for f in all_faces if f.smile_score is not None
        }
