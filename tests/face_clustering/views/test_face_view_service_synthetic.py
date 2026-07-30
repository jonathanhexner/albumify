"""spec-064 — synthetic-data tests for FaceAnalysisService.

Reuses the spec-045 synthetic run-dir builder so the service sees a
schema-valid v5 run with 3 real clusters + noise (~32 faces). Covers the
6 cases listed in spec §"Tests" plus 1 opt-in slow real-fixture smoke.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from face_cluster.views.face_view import FaceAnalysisService, FaceView
from face_cluster.views._base import CloseFace
from face_cluster.types import FaceRecord
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


@pytest.fixture
def synthetic_run_dir(tmp_path: Path) -> Path:
    return _build_synthetic_run_dir(tmp_path)


@pytest.fixture
def repo(synthetic_run_dir: Path) -> ClusterAnalysisRepository:
    return ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=synthetic_run_dir))


@pytest.fixture
def service(repo: ClusterAnalysisRepository) -> FaceAnalysisService:
    return FaceAnalysisService(repo)


def test_compute_face_detail_returns_face_view(service: FaceAnalysisService) -> None:
    """compute_face_detail(known id) returns a typed FaceView."""
    ids = service.list_face_ids()
    assert ids, "synthetic fixture should yield face ids"
    view = service.compute_face_detail(ids[0])
    assert isinstance(view, FaceView)
    assert view.face_id == ids[0]


def test_compute_face_detail_unknown_id_raises(service: FaceAnalysisService) -> None:
    """Unknown face_id raises a clear ValueError."""
    with pytest.raises(ValueError, match="not found"):
        service.compute_face_detail(face_id=-1)


def test_gate_result_populated(service: FaceAnalysisService) -> None:
    """gate_result is one of the documented strings (no None / empty)."""
    ids = service.list_face_ids()
    view = service.compute_face_detail(ids[0])
    assert view.gate_result in {"core", "holdout"}, view.gate_result


def test_closest_lists_have_typed_entries(service: FaceAnalysisService) -> None:
    """closest_same_cluster + closest_other_clusters contain CloseFace rows
    (one of the two may be empty for a noise-only face — total must be > 0
    in this fixture because the seeded clusters share embedding space)."""
    ids = service.list_face_ids()
    view = service.compute_face_detail(ids[0])
    total = view.closest_same_cluster + view.closest_other_clusters
    assert total, "expected at least one nearest face on a 30+-face fixture"
    assert all(isinstance(cf, CloseFace) for cf in total)


def test_get_face_record_returns_face_record(service: FaceAnalysisService) -> None:
    """get_face_record returns a FaceRecord with bbox + landmarks fields."""
    ids = service.list_face_ids()
    rec = service.get_face_record(ids[0])
    assert isinstance(rec, FaceRecord)
    assert rec.face_id == ids[0]
    # bbox is always a 4-tuple in the schema-valid synthetic fixture (may be
    # all zeros); landmarks may be None (the fixture doesn't seed them).
    assert rec.bbox is not None and len(rec.bbox) == 4


def test_score_fields_populated(service: FaceAnalysisService) -> None:
    """blur_score, area, pose fields exist and are correctly typed."""
    ids = service.list_face_ids()
    view = service.compute_face_detail(ids[0])
    assert isinstance(view.blur_score, float)
    assert isinstance(view.area, float)
    # pose may be None for fixtures that don't seed it.
    assert view.pose is None or len(view.pose) == 3


# ---------------------------------------------------------------------------
# Opt-in real-fixture smoke (spec §"Tests" — slow marker)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_fixture_compute_face_detail(v2_budapest_run_dir: Path) -> None:
    """Smoke: real Budapest run → first face id → FaceView returned."""
    repo = ClusterAnalysisRepository(
        ClusterAnalysisRepoConfig(run_dir=v2_budapest_run_dir, read_only=True)
    )
    svc = FaceAnalysisService(repo)
    ids = svc.list_face_ids()
    if not ids:
        pytest.skip("Real run has no faces.")
    view = svc.compute_face_detail(ids[0])
    assert isinstance(view, FaceView)
    assert view.face_id == ids[0]
