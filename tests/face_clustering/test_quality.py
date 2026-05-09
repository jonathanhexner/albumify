"""Unit tests for QualityGater verdict emission (spec 012 Phase 2)."""
import json
from pathlib import Path
import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.quality import QualityGater
from face_cluster.types import FaceRecord


def _make_face(face_id, blur=120.0, yaw=5.0, pitch=3.0, area=5000.0, image_id="img0"):
    face = FaceRecord(
        face_id=face_id,
        image_id=image_id,
        bbox=(0, 0, 100, 100),
    )
    face.blur_score = blur
    face.pose = (yaw, pitch, 0.0)
    face.area = area
    return face


def _default_config(**overrides):
    cfg = PipelineConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class ut_QualityGater_Verdicts:
    def test_passing_face_has_verdict_all_passed(self):
        cfg = _default_config(blur_min=50.0, yaw_max=30.0, pitch_max=25.0, require_pose=False)
        gater = QualityGater(cfg)
        faces = [_make_face(0, blur=120.0, yaw=5.0, pitch=3.0, area=5000.0)]
        core, holdout, verdicts = gater.select_core_set(faces)

        assert len(verdicts) == 1
        assert verdicts[0].all_passed() is True
        assert verdicts[0].rejection_reason is None
        assert 0 in core
        assert 0 not in holdout

    def test_blur_fail_sets_rejection_reason(self):
        cfg = _default_config(blur_min=50.0, yaw_max=30.0, pitch_max=25.0, require_pose=False)
        gater = QualityGater(cfg)
        faces = [_make_face(0, blur=20.0)]  # below 50
        core, holdout, verdicts = gater.select_core_set(faces)

        assert 0 in holdout
        assert verdicts[0].rejection_reason == "blur"
        assert verdicts[0].gates["blur"].passed is False
        assert verdicts[0].gates["blur"].value == pytest.approx(20.0)

    def test_pose_yaw_fail_rejection_priority_over_pitch(self):
        cfg = _default_config(blur_min=10.0, yaw_max=30.0, pitch_max=25.0, require_pose=False)
        gater = QualityGater(cfg)
        faces = [_make_face(0, blur=200.0, yaw=60.0, pitch=40.0)]  # both fail
        _, _, verdicts = gater.select_core_set(faces)

        assert verdicts[0].rejection_reason == "pose_yaw"

    def test_pitch_fail_when_yaw_passes(self):
        cfg = _default_config(blur_min=10.0, yaw_max=30.0, pitch_max=25.0, require_pose=False)
        gater = QualityGater(cfg)
        faces = [_make_face(0, blur=200.0, yaw=10.0, pitch=40.0)]
        _, _, verdicts = gater.select_core_set(faces)

        assert verdicts[0].rejection_reason == "pose_pitch"

    def test_verdict_length_equals_face_count(self):
        cfg = _default_config(blur_min=50.0)
        gater = QualityGater(cfg)
        faces = [_make_face(i) for i in range(10)]
        _, _, verdicts = gater.select_core_set(faces)

        assert len(verdicts) == 10

    def test_top_k_holdout_reason(self):
        cfg = _default_config(blur_min=10.0, max_faces_per_image_core=1)
        gater = QualityGater(cfg)
        # Two faces from same image — only top-1 by area is candidate
        f0 = _make_face(0, area=5000.0, image_id="img0")
        f1 = _make_face(1, area=1000.0, image_id="img0")
        core, holdout, verdicts = gater.select_core_set([f0, f1])

        # f1 dropped by top_k gate
        assert 1 in holdout
        assert verdicts[1].rejection_reason == "top_k_per_image"

    def test_quality_config_json_written(self, tmp_path):
        from face_cluster.pipeline import FaceClusteringPipeline, _RunContext
        from pathlib import Path
        import logging

        # We test _write_quality_config directly
        cfg = _default_config(
            blur_min=55.0, yaw_max=28.0, pitch_max=22.0,
            min_face_area=800.0, require_pose=True, max_faces_per_image_core=3
        )
        # Minimal _RunContext-like object for the write helper
        class _FakeCtx:
            config = cfg
            output_dir = tmp_path
            run_record = {}
            def write_run_record(self): pass

        FaceClusteringPipeline._write_quality_config(FaceClusteringPipeline(), _FakeCtx())

        qc = json.loads((tmp_path / "quality_config.json").read_text())
        assert qc["blur_min"] == pytest.approx(55.0)
        assert qc["yaw_max"] == pytest.approx(28.0)
        assert qc["pitch_max"] == pytest.approx(22.0)
        assert qc["area_min"] == pytest.approx(800.0)
        assert qc["require_pose"] is True
        assert qc["top_k_per_image"] == 3

    def test_verdict_attached_to_face(self):
        cfg = _default_config(blur_min=50.0, yaw_max=30.0, pitch_max=25.0, require_pose=False)
        gater = QualityGater(cfg)
        face = _make_face(0, blur=20.0)
        gater.select_core_set([face])

        assert face.quality_verdict is not None
        assert face.rejection_reason == "blur"
