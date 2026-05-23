"""spec-041 Phase 2 — guard the (params=, step_configs=) kwarg contract.

These are unit-level checks on the public signature of
``run_v2_pipeline`` — they do NOT run the producer chain. Pass an empty
source directory so the pipeline returns early before invoking
InsightFace. That keeps the test fast and CPU/GPU-free.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from app.face_clustering_v2 import pipeline as v2_pipeline
from app.face_clustering_v2.pipeline import run_v2_pipeline
from face_cluster.fc_params import FCParams


def test_both_params_and_step_configs_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="not both"):
        run_v2_pipeline(
            src_dir=tmp_path,
            output_dir=tmp_path / "out",
            params=FCParams(),
            step_configs={"foo": {}},
        )


def test_step_configs_emits_deprecation_warning(tmp_path: Path):
    # Reset the once-per-process latch so the test is order-independent.
    v2_pipeline._DEPRECATION_WARNED = False
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_v2_pipeline(src_dir=src, output_dir=out, step_configs={"foo": {}})
    assert any(
        issubclass(w.category, DeprecationWarning) and "FCParams" in str(w.message)
        for w in caught
    ), "expected DeprecationWarning pointing at FCParams"


def test_no_kwargs_runs_without_crashing(tmp_path: Path):
    """Neither params nor step_configs → defaults flow through."""
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"
    result = run_v2_pipeline(src_dir=src, output_dir=out)
    # Empty src → pipeline returns success=False with the no-images message,
    # but it must not raise.
    assert result.success is False
    assert "No images" in (result.error_message or "")


def test_params_path_does_not_emit_deprecation(tmp_path: Path):
    v2_pipeline._DEPRECATION_WARNED = False
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_v2_pipeline(src_dir=src, output_dir=out, params=FCParams())
    assert not any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "params= path must not emit DeprecationWarning"
