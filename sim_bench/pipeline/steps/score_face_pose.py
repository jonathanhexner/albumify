"""Score Face Pose step - head pose estimation using SixDRepNet."""

import logging
from typing import Dict, Any, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class ScoreFacePoseStep(BaseStep):
    """Score face pose (yaw/pitch/roll) using SixDRepNet."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_face_pose",
            display_name="Score Face Pose",
            description="Estimate head pose (yaw/pitch/roll) and compute frontal score using SixDRepNet.",
            category="people",
            requires=set(),  # faces is optional - step skips if none
            produces={"face_pose_scores"},
            depends_on=["detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "mps"],
                        "default": "cpu",
                        "description": "Device to run SixDRepNet on"
                    }
                }
            }
        )
        self._pose_estimator = None

    def _get_estimator(self, config: dict):
        """Lazy load pose estimator."""
        if self._pose_estimator is None:
            from sim_bench.face_pipeline.pose_estimator import SixDRepNetEstimator
            logger.info("Loading SixDRepNet pose estimator")
            estimator_config = {
                'device': config.get('device', 'cpu'),
                'face_pipeline': config
            }
            self._pose_estimator = SixDRepNetEstimator(estimator_config)
        return self._pose_estimator

    def _generate_cache_key(self, face) -> str:
        """Generate unique cache key for a face."""
        return f"{face.original_path}:face_{face.face_index}"

    def _serialize_pose(self, pose) -> Dict[str, Any]:
        """Convert pose to JSON-serializable dict."""
        if pose is None:
            return None
        return {
            'yaw': float(pose.yaw),
            'pitch': float(pose.pitch),
            'roll': float(pose.roll),
            'frontal_score': float(pose.frontal_score)
        }

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for pose scoring."""
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
            "feature_type": "face_pose",
            "model_name": "sixdrepnet",
            "metadata": {"device": config.get("device", "cpu")}
        }

    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, Dict[str, Any]]:
        """Process uncached faces - estimate pose."""
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

        estimator = self._get_estimator(config)
        results = {}

        for i, face in enumerate(uncached_faces):
            pose = estimator.estimate_pose(face.image)
            face.pose = pose
            key = self._generate_cache_key(face)
            results[key] = self._serialize_pose(pose)

            progress = (i + 1) / len(uncached_faces)
            context.report_progress(
                "score_face_pose", progress,
                f"Estimating pose {i + 1}/{len(uncached_faces)}"
            )

        return results

    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize pose dict to JSON bytes."""
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to pose dict."""
        return Serializers.json_deserialize(data)

    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, Dict[str, Any]],
        config: dict
    ) -> None:
        """Store pose scores in context and update face objects."""
        from sim_bench.face_pipeline.types import PoseEstimate

        all_faces = []
        for faces_list in context.faces.values():
            all_faces.extend(faces_list)

        # Apply cached data to faces
        for face in all_faces:
            key = self._generate_cache_key(face)
            if key in results and results[key]:
                pose_data = results[key]
                face.pose = PoseEstimate(
                    yaw=pose_data['yaw'],
                    pitch=pose_data['pitch'],
                    roll=pose_data['roll']
                )

        # Store scores in context
        context.face_pose_scores = {
            self._generate_cache_key(f): f.pose.frontal_score
            for f in all_faces if f.pose
        }

        logger.info(f"Scored pose for {len(context.face_pose_scores)} faces")
