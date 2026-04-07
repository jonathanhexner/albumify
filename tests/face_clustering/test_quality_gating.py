"""Unit tests for face_cluster.quality — blur and pose quality gating."""
import numpy as np
import pytest

from face_cluster.types import FaceRecord
from face_cluster.config import PipelineConfig
from face_cluster.quality import QualityGater


def make_face(
    face_id: int = 0,
    image_id: str = "test.jpg",
    blur_score: float = 100.0,
    pose=None,
    area: float = 10000.0,
) -> FaceRecord:
    return FaceRecord(
        face_id=face_id,
        image_id=image_id,
        bbox=(0, 0, 100, 100),
        area=area,
        blur_score=blur_score,
        pose=pose,
        aligned_face=None,
    )


class ut_QualityGater:
    """Unit tests for QualityGater.select_core_set()."""

    def test_blur_passes_above_threshold(self):
        config = PipelineConfig(blur_min=50.0)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0)
        core, holdout = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout

    def test_blur_rejects_below_threshold(self):
        config = PipelineConfig(blur_min=50.0)
        gater = QualityGater(config)
        face = make_face(blur_score=0.0)
        core, holdout = gater.select_core_set([face])
        assert 0 in holdout
        assert 0 not in core

    def test_heuristic_pose_never_gates(self):
        """With use_pose_estimation=False (default), pose angles are not used for gating.

        Heuristic landmark-based pose from InsightFace produces inflated pitch values
        (50-85 deg for frontal faces) and must not be used for quality gating.
        """
        config = PipelineConfig(blur_min=50.0, yaw_max=30.0, pitch_max=25.0)
        gater = QualityGater(config)  # use_pose_estimation=False by default
        face = make_face(blur_score=100.0, pose=(90.0, 85.0, 0.0))
        core, holdout = gater.select_core_set([face])
        assert 0 in core, "Heuristic pose must not gate faces"
        assert 0 not in holdout

    def test_sixdrepnet_bad_yaw_rejected(self):
        """With SixDRepNet active, yaw exceeding yaw_max goes to holdout."""
        config = PipelineConfig(blur_min=50.0, yaw_max=30.0)
        gater = QualityGater(config, use_pose_estimation=True)
        face = make_face(blur_score=100.0, pose=(90.0, 0.0, 0.0))
        core, holdout = gater.select_core_set([face])
        assert 0 in holdout
        assert 0 not in core

    def test_sixdrepnet_no_pose_rejected_when_required(self):
        """With SixDRepNet active and require_pose=True, face with pose=None goes to holdout."""
        config = PipelineConfig(require_pose=True, blur_min=50.0)
        gater = QualityGater(config, use_pose_estimation=True)
        face = make_face(blur_score=100.0, pose=None)
        core, holdout = gater.select_core_set([face])
        assert 0 in holdout
        assert 0 not in core

    def test_no_pose_passes_when_not_required(self):
        """Face with pose=None and require_pose=False goes to core if blur passes."""
        config = PipelineConfig(require_pose=False, blur_min=50.0)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, pose=None)
        core, holdout = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout
