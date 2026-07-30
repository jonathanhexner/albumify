"""spec-065 — synthetic-data tests for QualityService.

Reuses the spec-045 synthetic run-dir builder and seeds a deterministic
set of filter_decisions covering both passes and rejections across two
gates ("blur", "pose_yaw") plus one image-level gate. Covers spec
§"Tests" (5 synthetic + 1 opt-in real-fixture smoke).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from face_cluster.views.quality import (
    GateCounts,
    QualityService,
    QualitySummary,
)
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
    FilterDecisionCriteria,
)
from sim_bench.run_db.store import FilterDecisionRow
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


def _seed_filter_decisions(run_dir: Path) -> None:
    """Seed:
      - face_001: blur rejected, area passed
      - face_002: blur passed, pose_yaw rejected
      - face_003: blur rejected, pose_yaw rejected
      - image_999: scene_quality rejected
    => 7 rows, 5 rejections, blur is top rejection gate (2 rejections).
    """
    conn = sqlite3.connect(str(run_dir / "face_clustering.db"))
    try:
        rows = [
            ("face_001", "face",  None, "blur",          1, "blur low",  '{"v": 12.3}'),
            ("face_001", "face",  None, "area",          0, "ok",        '{"v": 5000}'),
            ("face_002", "face",  None, "blur",          0, "ok",        '{"v": 80.1}'),
            ("face_002", "face",  None, "pose_yaw",      1, "yaw high",  '{"v": 45.0}'),
            ("face_003", "face",  None, "blur",          1, "blur low",  '{"v": 9.0}'),
            ("face_003", "face",  None, "pose_yaw",      1, "yaw high",  '{"v": 50.0}'),
            ("image_999", "image", None, "scene_quality", 1, "low iqa",   '{"v": 0.2}'),
        ]
        conn.executemany(
            "INSERT INTO filter_decisions "
            "(item_id, item_type, parent_id, filter_name, rejected, reason, measured_json) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = _build_synthetic_run_dir(tmp_path)
    _seed_filter_decisions(d)
    return d


@pytest.fixture
def service(run_dir: Path) -> QualityService:
    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    return QualityService(repo)


def test_list_filter_decisions_typed(service):
    rows = service.list_filter_decisions()
    assert isinstance(rows, list)
    assert all(isinstance(r, FilterDecisionRow) for r in rows)
    assert len(rows) == 7


def test_list_filter_decisions_passes_criteria_through(service):
    rejected = service.list_filter_decisions(FilterDecisionCriteria(rejected=True))
    assert len(rejected) == 5
    assert all(r.rejected for r in rejected)


def test_summary_counts_match_seed(service):
    s = service.summary()
    assert isinstance(s, QualitySummary)
    assert s.n_decisions == 7
    assert s.n_rejected == 5
    assert s.n_items == 4  # face_001/002/003 + image_999


def test_summary_per_gate_breakdown_correct(service):
    s = service.summary()
    by_name = {g.gate_name: g for g in s.gates}
    assert isinstance(by_name["blur"], GateCounts)
    # 3 blur rows: face_001 rejected, face_002 passed, face_003 rejected
    assert by_name["blur"].n_passed == 1
    assert by_name["blur"].n_rejected == 2
    # 2 pose_yaw rows: both rejected
    assert by_name["pose_yaw"].n_passed == 0
    assert by_name["pose_yaw"].n_rejected == 2
    # 1 area row: passed
    assert by_name["area"].n_passed == 1
    assert by_name["area"].n_rejected == 0


def test_summary_top_rejection_gate_and_pass_rate(service):
    s = service.summary()
    # blur and pose_yaw tie at 2 rejections; "blur" wins as max() is stable on
    # dict iteration order (insertion order = scan order). What matters: a gate
    # IS picked and it's one of the two top gates.
    assert s.top_rejection_gate in {"blur", "pose_yaw"}
    # pass_rate = 1 - 5/7
    assert s.pass_rate == pytest.approx(1.0 - 5 / 7)


# ---------------------------------------------------------------------------
# Opt-in real-fixture smoke
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_fixture_summary(v2_budapest_run_dir: Path) -> None:
    """Smoke: real Budapest run yields a sensible QualitySummary."""
    repo = ClusterAnalysisRepository(
        ClusterAnalysisRepoConfig(run_dir=v2_budapest_run_dir, read_only=True)
    )
    svc = QualityService(repo)
    s = svc.summary()
    assert isinstance(s, QualitySummary)
    assert s.n_decisions >= 0
    assert 0.0 <= s.pass_rate <= 1.0
