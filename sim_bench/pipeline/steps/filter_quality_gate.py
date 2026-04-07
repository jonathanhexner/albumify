"""Filter Quality Gate step - Filter low-quality faces for clustering.

Applies quality filters (pose, blur, area) to select high-quality faces for clustering.
Low-quality faces are marked as holdout and can be attached later.
"""

import logging
import time
from typing import Dict, Any, List

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from face_cluster.types import FaceRecord
from face_cluster.quality import QualityGater
from face_cluster.config import PipelineConfig as FaceClusterConfig

logger = logging.getLogger(__name__)


@register_step
class FilterQualityGateStep(BaseStep):
    """Filter faces using quality gate (pose, blur, area criteria).

    Requires face_embeddings step to have populated embeddings.
    Produces core_indices and holdout_indices in context.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="filter_quality_gate",
            display_name="Filter Quality Gate",
            description="Filter low-quality faces (pose, blur, area) for clustering",
            category="people",
            requires={"insightface_faces", "aligned_faces", "face_embeddings"},
            produces={"core_indices", "holdout_indices", "face_records"},
            depends_on=["extract_face_embeddings"],
            config_schema={
                "type": "object",
                "properties": {
                    "yaw_max": {
                        "type": "number",
                        "default": 45.0,
                        "description": "Maximum absolute yaw angle (degrees)"
                    },
                    "pitch_max": {
                        "type": "number",
                        "default": 30.0,
                        "description": "Maximum absolute pitch angle (degrees)"
                    },
                    "roll_max": {
                        "type": "number",
                        "default": 30.0,
                        "description": "Maximum absolute roll angle (degrees)"
                    },
                    "blur_min": {
                        "type": "number",
                        "default": 100.0,
                        "description": "Minimum blur score (Laplacian variance)"
                    },
                    "min_face_area": {
                        "type": "number",
                        "default": None,
                        "description": "Minimum face area in pixels (optional)"
                    },
                    "max_faces_per_image_core": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max faces per image to include in core set"
                    },
                    "use_pose_estimation": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to use SixDRepNet pose estimation"
                    }
                }
            }
        )

    def _create_face_records(self, context: PipelineContext) -> List[FaceRecord]:
        """Create FaceRecord objects from context data.

        Args:
            context: Pipeline context with insightface_faces, aligned_faces, embeddings

        Returns:
            List of FaceRecord objects
        """
        face_records = []
        face_id = 0

        for img_path, face_data in context.insightface_faces.items():
            for face_info in face_data.get('faces', []):
                face_idx = face_info.get('face_index', 0)
                face_key = f"{str(img_path).replace(chr(92), '/')}:face_{face_idx}"

                # Skip if no embedding
                embedding = context.face_embeddings.get(face_key)
                if embedding is None:
                    continue

                # Get aligned crop
                aligned_crop = context.aligned_faces.get(face_key)

                # Compute area from bbox
                bbox = face_info.get('bbox', {})
                w_px = bbox.get('w_px', 0)
                h_px = bbox.get('h_px', 0)
                area = w_px * h_px

                # Get pose from face_info if available
                pose = face_info.get('pose')  # (yaw, pitch, roll) or None

                # Create FaceRecord
                record = FaceRecord(
                    face_id=face_id,
                    image_id=str(img_path),
                    bbox=(bbox.get('x_px', 0), bbox.get('y_px', 0),
                          bbox.get('x_px', 0) + w_px, bbox.get('y_px', 0) + h_px),
                    landmarks=None,  # Not needed for quality gate
                    aligned_face=aligned_crop,
                    embedding=embedding,
                    embedding_normalized=embedding / (np.linalg.norm(embedding) + 1e-8),
                    pose=pose,
                    blur_score=0.0,  # Will be computed by QualityGater
                    area=area,
                    is_core=False,
                    image_path=str(img_path),
                    face_index=face_idx
                )
                face_records.append(record)
                face_id += 1

        logger.info(f"Created {len(face_records)} FaceRecord objects")
        return face_records

    def process(self, context: PipelineContext, config: dict) -> None:
        """Apply quality gate filtering.

        Args:
            context: Pipeline context
            config: Step configuration
        """
        start_time = time.time()

        # Create FaceRecord objects from context
        face_records = self._create_face_records(context)

        if not face_records:
            logger.warning("No faces with embeddings to filter")
            context.core_indices = []
            context.holdout_indices = []
            context.face_records = []
            context.report_progress("filter_quality_gate", 1.0, "No faces to filter")
            return

        # Create face_cluster config from step config
        fc_config = FaceClusterConfig(
            yaw_max=config.get('yaw_max', 45.0),
            pitch_max=config.get('pitch_max', 30.0),
            roll_max=config.get('roll_max', 30.0),
            blur_min=config.get('blur_min', 100.0),
            min_face_area=config.get('min_face_area'),
            max_faces_per_image_core=config.get('max_faces_per_image_core', 10),
        )

        # Create quality gater
        use_pose_estimation = config.get('use_pose_estimation', False)
        gater = QualityGater(fc_config, use_pose_estimation=use_pose_estimation)

        # Compute blur scores
        context.report_progress("filter_quality_gate", 0.3, "Computing blur scores")
        face_records = gater.compute_blur_scores(face_records)

        # Optionally compute pose scores (if use_pose_estimation=True)
        if use_pose_estimation:
            context.report_progress("filter_quality_gate", 0.5, "Computing pose scores")
            face_records = gater.compute_pose_scores(face_records)

        # Apply quality gate
        context.report_progress("filter_quality_gate", 0.7, "Applying quality filters")
        core_indices, holdout_indices = gater.select_core_set(face_records)

        # Validation check: Ensure we have at least one core face
        assert len(core_indices) > 0, \
            "No faces passed quality gate - check thresholds (all faces filtered out)"

        # Store results in context
        context.core_indices = core_indices
        context.holdout_indices = holdout_indices
        context.face_records = face_records

        duration = time.time() - start_time

        # Log with timing
        logger.info("=" * 60)
        logger.info("FILTER_QUALITY_GATE: Stage completed")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Total faces: {len(face_records)}")
        logger.info(f"Core faces: {len(core_indices)} ({len(core_indices)/len(face_records)*100:.1f}%)")
        logger.info(f"Holdout faces: {len(holdout_indices)} ({len(holdout_indices)/len(face_records)*100:.1f}%)")
        logger.info("=" * 60)

        context.report_progress(
            "filter_quality_gate", 1.0,
            f"Quality gate: {len(core_indices)} core, {len(holdout_indices)} holdout"
        )
