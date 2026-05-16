"""Tests for spec-032 P1: filter_quality + QualityGater + _save_crops migrations.

Each test builds a minimal context, runs the step, then asserts the expected
filter decisions landed in ctx.filters with the canonical filter_name.

These are unit tests; the heavyweight E2E pipeline test is left to the
existing test_merge_stage.py suite (which exercises the full chain).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.filter_context import FilterContext
from face_cluster.pipeline import FaceClusteringPipeline
from face_cluster.types import FaceRecord


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_face(face_id: int, image_path: str, blur: float = 100.0,
               pose=None, area: float = 1.0,
               aligned: bool = True, noisy: bool = True) -> FaceRecord:
    # Note: compute_blur_scores re-computes blur from aligned_face's Laplacian
    # variance, overwriting whatever we pass in.
    #   aligned=False → blur_score forced to 0.0 by compute_blur_scores
    #   aligned=True + noisy=True  → variance > blur_min, gate passes
    #   aligned=True + noisy=False → variance ≈ 0, blur gate REJECTS
    if not aligned:
        aligned_arr = None
    elif noisy:
        rng = np.random.default_rng(face_id)
        aligned_arr = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        aligned_arr = np.zeros((112, 112, 3), dtype=np.uint8)
    return FaceRecord(
        face_id=face_id,
        image_id=Path(image_path).stem,
        bbox=(0.0, 0.0, 100.0, 100.0),
        landmarks=None,
        aligned_face=aligned_arr,
        embedding=np.zeros(512, dtype=np.float32),
        embedding_normalized=np.zeros(512, dtype=np.float32),
        pose=pose,
        blur_score=blur,
        area=area,
        is_core=False,
        image_path=image_path,
        face_index=0,
    )


def _make_run_ctx(faces: List[FaceRecord], cfg: PipelineConfig, tmp_path: Path):
    """Build a minimal _RunContext sufficient for the quality / crops stages."""
    from face_cluster.pipeline import _RunContext, _setup_run_logging
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True)
    handler, log_path = _setup_run_logging(output_dir)
    ctx = _RunContext(
        config=cfg,
        image_dir=tmp_path,
        output_dir=output_dir,
        on_progress=None,
        run_record={"stages": {}, "status": "running"},
        file_handler=handler,
        log_path=log_path,
        faces=faces,
    )
    return ctx


# -----------------------------------------------------------------------------
# Quality gate translation
# -----------------------------------------------------------------------------
class ut_QualityGateEmitsFilters:
    def test_quality_passing_face_records_no_rejections(self, tmp_path):
        cfg = PipelineConfig(blur_min=50.0, yaw_max=30.0, pitch_max=25.0,
                             max_faces_per_image_core=3)
        # Single high-blur face, neutral pose — should pass everything.
        faces = [_make_face(0, str(tmp_path / "img.jpg"),
                            blur=200.0, pose=(0.0, 0.0, 0.0), area=1.0)]
        ctx = _make_run_ctx(faces, cfg, tmp_path)

        pipeline = FaceClusteringPipeline()
        pipeline._quality_gate(ctx)

        face_state = ctx.filters.get("face_0000")
        assert face_state is not None
        # No decision should be rejected.
        rejected = [d for d in face_state.decisions if d.rejected]
        assert not rejected, f"unexpected rejections: {[d.filter_name for d in rejected]}"

    def test_blur_too_low_records_face_blur_rejection(self, tmp_path):
        cfg = PipelineConfig(blur_min=50.0, max_faces_per_image_core=3)
        # Two faces — one noisy (passes blur), one all-zero (fails blur).
        faces = [
            _make_face(0, str(tmp_path / "a.jpg"), pose=(0.0, 0.0, 0.0),
                       noisy=True),
            _make_face(1, str(tmp_path / "b.jpg"), pose=(0.0, 0.0, 0.0),
                       noisy=False),
        ]
        ctx = _make_run_ctx(faces, cfg, tmp_path)

        pipeline = FaceClusteringPipeline()
        pipeline._quality_gate(ctx)

        # face_0001 must have a rejected face_blur decision.
        f1 = ctx.filters.get("face_0001")
        assert f1 is not None
        blur_decisions = [d for d in f1.decisions if d.filter_name == "face_blur"]
        assert len(blur_decisions) == 1
        assert blur_decisions[0].rejected is True

    def test_top_k_culled_face_records_top_k_rejection(self, tmp_path):
        # 4 faces in same image, top_k=1 → 3 culled.
        cfg = PipelineConfig(blur_min=50.0, max_faces_per_image_core=1)
        faces = [_make_face(i, str(tmp_path / "img.jpg"),
                            blur=200.0, pose=(0.0, 0.0, 0.0),
                            area=float(4 - i))  # area decreases by id
                 for i in range(4)]
        ctx = _make_run_ctx(faces, cfg, tmp_path)

        pipeline = FaceClusteringPipeline()
        pipeline._quality_gate(ctx)

        # Face 0 (largest area) survives; 1/2/3 are top-K-culled.
        survived = [d.rejected
                    for d in ctx.filters.get("face_0000").decisions
                    if d.filter_name == "face_top_k_per_image"]
        culled1 = [d.rejected
                   for d in ctx.filters.get("face_0001").decisions
                   if d.filter_name == "face_top_k_per_image"]
        assert survived == [False]
        assert culled1 == [True]

    def test_face_parent_id_is_image_path(self, tmp_path):
        cfg = PipelineConfig(blur_min=50.0, max_faces_per_image_core=3)
        img_path = str(tmp_path / "img.jpg")
        faces = [_make_face(7, img_path, blur=200.0, pose=(0.0, 0.0, 0.0))]
        ctx = _make_run_ctx(faces, cfg, tmp_path)

        FaceClusteringPipeline()._quality_gate(ctx)

        assert ctx.filters.get("face_0007").parent_id == img_path


# -----------------------------------------------------------------------------
# Crop stage records face_crop decisions (closes SIGHTING-059 #1)
# -----------------------------------------------------------------------------
class ut_CropStageEmitsFilters:
    def test_saved_face_records_face_crop_passed(self, tmp_path):
        cfg = PipelineConfig(blur_min=50.0, max_faces_per_image_core=3)
        faces = [_make_face(0, str(tmp_path / "img.jpg"),
                            blur=200.0, pose=(0.0, 0.0, 0.0), aligned=True)]
        ctx = _make_run_ctx(faces, cfg, tmp_path)

        FaceClusteringPipeline()._save_crops(ctx)

        decisions = [d for d in ctx.filters.get("face_0000").decisions
                     if d.filter_name == "face_crop"]
        assert len(decisions) == 1
        assert decisions[0].rejected is False

    def test_no_aligned_face_records_face_crop_rejected_landmarks_missing(self, tmp_path):
        # This is the SIGHTING-059 #1 reproduction: face survives every other
        # stage but the crop step skips it because aligned_face is None.
        # Today: silent. With spec-032 P1: a recorded rejection with reason.
        cfg = PipelineConfig(blur_min=50.0, max_faces_per_image_core=3)
        faces = [_make_face(46, str(tmp_path / "20250822_123354.jpg"),
                            blur=200.0, pose=(0.0, 0.0, 0.0), aligned=False)]
        ctx = _make_run_ctx(faces, cfg, tmp_path)

        FaceClusteringPipeline()._save_crops(ctx)

        decisions = [d for d in ctx.filters.get("face_0046").decisions
                     if d.filter_name == "face_crop"]
        assert len(decisions) == 1
        assert decisions[0].rejected is True
        assert "landmarks_missing" in decisions[0].reason

    def test_summary_counts_skipped_crops(self, tmp_path):
        cfg = PipelineConfig(blur_min=50.0, max_faces_per_image_core=3)
        faces = [
            _make_face(0, str(tmp_path / "a.jpg"), aligned=True,
                       blur=200.0, pose=(0.0, 0.0, 0.0)),
            _make_face(1, str(tmp_path / "b.jpg"), aligned=False,
                       blur=200.0, pose=(0.0, 0.0, 0.0)),
            _make_face(2, str(tmp_path / "c.jpg"), aligned=False,
                       blur=200.0, pose=(0.0, 0.0, 0.0)),
        ]
        ctx = _make_run_ctx(faces, cfg, tmp_path)

        FaceClusteringPipeline()._save_crops(ctx)

        summary = ctx.filters.summary()
        assert summary.get("face_crop", {}).get("face", 0) == 2


# -----------------------------------------------------------------------------
# Quality gate writes filter decisions BEFORE save_crops (ordering invariant)
# -----------------------------------------------------------------------------
class ut_FilterOrderingInvariants:
    def test_face_top_k_is_recorded_before_face_crop_per_KNOWN_FILTERS(self):
        """The list order in KNOWN_FILTERS must match the pipeline order so
        that `active(after="face_top_k_per_image")` answers correctly when
        querying state mid-pipeline."""
        from face_cluster.filter_context import filter_position
        assert filter_position("face_top_k_per_image") \
            < filter_position("face_crop"), (
            "face_crop must come after face_top_k_per_image in KNOWN_FILTERS — "
            "the pipeline runs quality before crops."
        )

    def test_image_quality_comes_first_among_filters(self):
        from face_cluster.filter_context import KNOWN_FILTERS
        assert KNOWN_FILTERS[0][0] == "image_quality"
