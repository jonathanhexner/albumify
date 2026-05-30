"""spec-063 — FCAppRunner.recluster() unit tests.

Verifies the runner can re-cluster a prior run dir without re-running the
producer chain (detect / align / embed). The prior run dir is built by the
spec-045 synthetic fixture helper, which writes a schema-valid v5 layout
with embeddings and 32 face rows across 3 real clusters + a noise cluster.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from face_cluster.fc_app_runner import FCAppRunner, FCAppRunResult
from face_cluster.fc_params import FCParams
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


@pytest.fixture
def prior_run_dir(tmp_path: Path) -> Path:
    """A schema-valid v5 run dir suitable as input to recluster()."""
    return _build_synthetic_run_dir(tmp_path)


def test_recluster_skips_producer_chain(prior_run_dir: Path) -> None:
    """recluster() returns a successful FCAppRunResult without invoking
    the producer chain. The synthetic fixture has no source images, so a
    successful run proves the producer steps were not invoked."""
    params = FCParams(
        K=3,
        distance_threshold=0.5,
        min_cluster_size=2,
        blur_min=0.0,
        yaw_max=999.0,
        pitch_max=999.0,
        roll_max=999.0,
        max_faces_per_image_core=50,
        merge_enabled=False,
        attach_enabled=False,
    )
    result = FCAppRunner().recluster(prior_run_dir, params.to_step_configs())
    assert isinstance(result, FCAppRunResult)
    assert result.success, f"recluster() failed: {result.error_message}"
    # Synthetic fixture has 32 faces (30 real + 2 noise). At minimum some
    # of them should be assigned to clusters.
    assert result.n_faces_assigned > 0


def test_recluster_uses_prior_face_records(prior_run_dir: Path) -> None:
    """The face_records loaded for clustering match what RunStore reads."""
    from sim_bench.run_db.store import RunStore

    expected_faces = RunStore(prior_run_dir).faces()
    assert expected_faces, "synthetic fixture must produce >= 1 face record"

    # Build a context-capturing variant: spy on the underlying .run() call
    # by overriding it on a one-off FCAppRunner instance.
    runner = FCAppRunner()
    captured: dict = {}
    original_run = runner.run

    def spy_run(context, step_configs=None):  # type: ignore[no-untyped-def]
        captured["face_records"] = list(context.face_records)
        captured["parent_run_id"] = getattr(context, "parent_run_id", None)
        return original_run(context, step_configs=step_configs)

    runner.run = spy_run  # type: ignore[method-assign]

    params = FCParams(
        K=3, distance_threshold=0.5, min_cluster_size=2,
        blur_min=0.0, yaw_max=999.0, pitch_max=999.0, roll_max=999.0,
        max_faces_per_image_core=50, merge_enabled=False, attach_enabled=False,
    )
    runner.recluster(prior_run_dir, params.to_step_configs())

    assert len(captured["face_records"]) == len(expected_faces)
    assert captured["parent_run_id"] == prior_run_dir.name
