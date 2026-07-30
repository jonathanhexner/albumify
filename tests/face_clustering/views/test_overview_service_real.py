"""spec-066 — OverviewService smoke test against the real action_log.

Correctness is covered by test_overview_service_synthetic.py; this file
guards against real-data-only failure modes (schema drift, the actual
payload shape the v2 app writes). Marked ``slow`` and skips cleanly when
the user's DB / fc_app_v2 runs aren't present.

Bypasses the session-scoped action_log monkeypatch by passing an explicit
db_path — same pattern + rationale as test_history_service_real.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from face_cluster.repositories import RunHistoryRepoConfig, RunHistoryRepository
from face_cluster.views.overview import DashboardMetrics, OverviewService

pytestmark = pytest.mark.slow


def _real_db_path() -> Path:
    return Path.home() / ".sim_bench" / "sim_bench.db"


@pytest.fixture(scope="module")
def real_service() -> OverviewService:
    db = _real_db_path()
    if not db.exists():
        pytest.skip(f"Real action_log DB not present at {db}")
    return OverviewService(
        repo=RunHistoryRepository(RunHistoryRepoConfig(db_path=db)),
    )


def test_real_dashboard_computes(real_service):
    m = real_service.compute_dashboard(limit=50)
    assert isinstance(m, DashboardMetrics)
    if m.total_runs == 0:
        pytest.skip("No fc_app_v2 runs in the real action_log.")
    # Real data invariants — no exact numbers (they drift as the user runs).
    assert m.total_runs >= 1
    assert m.total_faces_ever >= 0
    assert m.last_run_at is not None
    assert 0.0 <= (m.gate_pass_rate or 0.0) <= 1.0
    assert len(m.per_album) >= 1
    assert sum(s.n_runs for s in m.per_status) == m.total_runs
    assert len(m.timeseries) == m.total_runs
