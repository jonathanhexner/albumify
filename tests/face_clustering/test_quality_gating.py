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
    det_score: float | None = None,
) -> FaceRecord:
    return FaceRecord(
        face_id=face_id,
        image_id=image_id,
        bbox=(0, 0, 100, 100),
        area=area,
        blur_score=blur_score,
        pose=pose,
        aligned_face=None,
        det_score=det_score,
    )


class ut_QualityGater:
    """Unit tests for QualityGater.select_core_set()."""

    def test_blur_passes_above_threshold(self):
        config = PipelineConfig(blur_min=50.0)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout

    def test_blur_rejects_below_threshold(self):
        config = PipelineConfig(blur_min=50.0)
        gater = QualityGater(config)
        face = make_face(blur_score=0.0)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in holdout
        assert 0 not in core

    def test_heuristic_pose_never_gates(self):
        """InsightFace 1k3d68 pose IS used for gating (reliable model — SIGHTING-020 resolved).

        pose angles from InsightFace buffalo_l 1k3d68 are reliable and used for quality
        gating. A face with extreme pose (yaw=90) should be rejected.
        """
        config = PipelineConfig(blur_min=50.0, yaw_max=30.0, pitch_max=25.0)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, pose=(90.0, 85.0, 0.0))
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in holdout, "Extreme pose (yaw=90) must be rejected by quality gate"
        assert 0 not in core

    def test_sixdrepnet_bad_yaw_rejected(self):
        """With SixDRepNet active, yaw exceeding yaw_max goes to holdout."""
        config = PipelineConfig(blur_min=50.0, yaw_max=30.0)
        gater = QualityGater(config, use_pose_estimation=True)
        face = make_face(blur_score=100.0, pose=(90.0, 0.0, 0.0))
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in holdout
        assert 0 not in core

    def test_sixdrepnet_no_pose_rejected_when_required(self):
        """With SixDRepNet active and require_pose=True, face with pose=None goes to holdout."""
        config = PipelineConfig(require_pose=True, blur_min=50.0)
        gater = QualityGater(config, use_pose_estimation=True)
        face = make_face(blur_score=100.0, pose=None)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in holdout
        assert 0 not in core

    def test_no_pose_passes_when_not_required(self):
        """Face with pose=None and require_pose=False goes to core if blur passes."""
        config = PipelineConfig(require_pose=False, blur_min=50.0)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, pose=None)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout

    # --- det_score gate ---

    def test_det_score_gate_disabled_by_default(self):
        """det_score_min=None means gate is off; any det_score passes."""
        config = PipelineConfig()  # default: det_score_min=None
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, det_score=0.1)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout

    def test_det_score_rejects_low_score(self):
        """Face with det_score below threshold goes to holdout."""
        config = PipelineConfig(det_score_min=0.7)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, det_score=0.5)
        core, holdout, verdicts = gater.select_core_set([face])
        assert 0 in holdout
        assert 0 not in core
        assert verdicts[0].rejection_reason == "det_score"

    def test_det_score_passes_high_score(self):
        """Face with det_score at or above threshold goes to core."""
        config = PipelineConfig(det_score_min=0.7)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, det_score=0.85)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout

    def test_det_score_passes_at_exact_threshold(self):
        """Face with det_score exactly equal to threshold is accepted."""
        config = PipelineConfig(det_score_min=0.7)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, det_score=0.7)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout

    def test_det_score_none_passes_permissively(self):
        """Face with det_score=None passes even when gate is enabled (legacy data)."""
        config = PipelineConfig(det_score_min=0.7)
        gater = QualityGater(config)
        face = make_face(blur_score=100.0, det_score=None)
        core, holdout, _ = gater.select_core_set([face])
        assert 0 in core
        assert 0 not in holdout

    def test_det_score_rejection_takes_priority_over_blur(self):
        """det_score rejection_reason appears before blur in priority order."""
        config = PipelineConfig(det_score_min=0.7, blur_min=50.0)
        gater = QualityGater(config)
        # Both det_score and blur fail
        face = make_face(blur_score=10.0, det_score=0.3)
        core, holdout, verdicts = gater.select_core_set([face])
        assert 0 in holdout
        assert verdicts[0].rejection_reason == "det_score"
