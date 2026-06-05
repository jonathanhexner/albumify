"""spec-069 — synthetic-data tests for FaceMetricsService.

Uses the spec-045 synthetic v5 run dir. Verifies one row per face, the
assigned/unassigned partition (derived, since the pipeline doesn't persist
noise rows — SIGHTING-093), and that per-face metrics are surfaced.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from face_cluster.views.face_metrics import FaceMetricRow, FaceMetricsService
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


@pytest.fixture
def service(tmp_path: Path) -> FaceMetricsService:
    run_dir = _build_synthetic_run_dir(tmp_path)
    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    return FaceMetricsService(repo)


def test_rejects_none_repo():
    with pytest.raises(ValueError):
        FaceMetricsService(None)  # type: ignore[arg-type]


def test_one_row_per_face_and_typed(service: FaceMetricsService):
    rows = service.list_faces()
    assert rows, "expected at least one face row"
    assert all(isinstance(r, FaceMetricRow) for r in rows)
    # face_ids unique.
    ids = [r.face_id for r in rows]
    assert len(ids) == len(set(ids))


def test_status_partition_covers_every_face(service: FaceMetricsService):
    """assigned + unassigned must equal the total — no face falls through."""
    rows = service.list_faces()
    assigned = [r for r in rows if r.status == "assigned"]
    unassigned = [r for r in rows if r.status == "unassigned"]
    assert len(assigned) + len(unassigned) == len(rows)
    # assigned rows carry a cluster_id; unassigned do not.
    assert all(r.cluster_id is not None for r in assigned)
    assert all(r.cluster_id is None for r in unassigned)


def test_metrics_surfaced(service: FaceMetricsService):
    """blur / area / det_score come straight from the faces table."""
    rows = service.list_faces()
    assert all(isinstance(r.blur, float) for r in rows)
    assert all(isinstance(r.area, float) for r in rows)
    # det_score is Optional but, when present, numeric.
    assert all(r.det_score is None or isinstance(r.det_score, float) for r in rows)
