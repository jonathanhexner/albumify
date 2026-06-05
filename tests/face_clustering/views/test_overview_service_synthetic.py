"""spec-066 — OverviewService unit tests against synthetic action_log data.

Uses the function-scoped ``synthetic_action_log_db`` fixture (an empty
in-memory-ish SQLite with the action_log schema) seeded via the
``_seed.py`` row factories. No real-data dependency — runs in ms on CI.

Per the spec-066 testing strategy this is the load-bearing layer; the
real-fixture test (test_overview_service_real.py) is a smoke confirmation.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from face_cluster.repositories import RunHistoryRepoConfig, RunHistoryRepository
from face_cluster.views.overview import (
    DashboardMetrics,
    OverviewService,
    NO_PROFILE,
)
from tests.face_clustering.views._seed import insert_action


def _service(db_path) -> OverviewService:
    return OverviewService(
        repo=RunHistoryRepository(RunHistoryRepoConfig(db_path=db_path)),
    )


# ===========================================================================
# Case 1 — empty history yields zeros / None, never raises
# ===========================================================================

def test_empty_history_returns_zeroed_metrics(synthetic_action_log_db):
    m = _service(synthetic_action_log_db).compute_dashboard()
    assert isinstance(m, DashboardMetrics)
    assert m.total_runs == 0
    assert m.total_faces_ever == 0
    assert m.avg_n_clusters is None
    assert m.median_n_clusters is None
    assert m.last_run_at is None
    assert m.gate_pass_rate is None
    assert m.per_album == [] and m.per_status == [] and m.timeseries == []


# ===========================================================================
# Case 2 — only fc_app_v2 runs are counted (producer filter)
# ===========================================================================

def test_producer_filter_excludes_other_producers(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, producer="fc_app_v2", n_faces=10, n_clusters=3)
    insert_action(synthetic_action_log_db, producer="fc_app_v2", n_faces=20, n_clusters=4)
    insert_action(synthetic_action_log_db, producer="albumify", n_faces=999, n_clusters=99)
    insert_action(synthetic_action_log_db, producer=None, n_faces=999, n_clusters=99)

    m = _service(synthetic_action_log_db).compute_dashboard()
    assert m.total_runs == 2
    assert m.total_faces_ever == 30  # 10 + 20; the albumify/None rows excluded


# ===========================================================================
# Case 3 — avg / median / total faces math, tolerant of None fields
# ===========================================================================

def test_cluster_stats_skip_none_and_compute_correctly(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, n_faces=100, n_clusters=10)
    insert_action(synthetic_action_log_db, n_faces=200, n_clusters=20)
    insert_action(synthetic_action_log_db, n_faces=None, n_clusters=None)  # incomplete run

    m = _service(synthetic_action_log_db).compute_dashboard()
    assert m.total_runs == 3
    assert m.total_faces_ever == 300            # None counts as 0
    assert m.avg_n_clusters == 15.0             # mean(10, 20); None skipped
    assert m.median_n_clusters == 15.0


# ===========================================================================
# Case 4 — gate pass rate + per-status breakdown
# ===========================================================================

def test_gate_pass_rate_and_per_status(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, status="complete", n_clusters=5)
    insert_action(synthetic_action_log_db, status="complete", n_clusters=6)
    insert_action(synthetic_action_log_db, status="complete", n_clusters=7)
    insert_action(synthetic_action_log_db, status="failed")

    m = _service(synthetic_action_log_db).compute_dashboard()
    assert m.gate_pass_rate == 0.75             # 3 of 4 complete
    statuses = {s.status: s.n_runs for s in m.per_status}
    assert statuses == {"complete": 3, "failed": 1}
    # desc-by-count ordering: complete (3) before failed (1)
    assert m.per_status[0].status == "complete"


# ===========================================================================
# Case 5 — per-album ranking + chronological time-series
# ===========================================================================

def test_per_album_ranking_and_timeseries_order(synthetic_action_log_db):
    base = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    # Budapest x3, Rome x1 — inserted newest-first to prove the service re-sorts.
    insert_action(synthetic_action_log_db, source_album="Rome", started_at=base + timedelta(hours=3), n_clusters=9)
    insert_action(synthetic_action_log_db, source_album="Budapest", started_at=base + timedelta(hours=2), n_clusters=8)
    insert_action(synthetic_action_log_db, source_album="Budapest", started_at=base + timedelta(hours=1), n_clusters=7)
    insert_action(synthetic_action_log_db, source_album="Budapest", started_at=base, n_clusters=6)

    m = _service(synthetic_action_log_db).compute_dashboard()
    assert m.per_album[0].album == "Budapest" and m.per_album[0].n_runs == 3
    assert m.per_album[1].album == "Rome" and m.per_album[1].n_runs == 1
    # last_run_at is the max started_at (Rome @ +3h)
    assert m.last_run_at == (base + timedelta(hours=3)).astimezone(timezone.utc).isoformat()
    # timeseries is oldest -> newest
    ts_clusters = [p.n_clusters for p in m.timeseries]
    assert ts_clusters == [6, 7, 8, 9]


# ===========================================================================
# Case 6 — profile read from payload; (none) for runs that predate the writer
# ===========================================================================

def test_per_profile_reads_payload_and_falls_back(synthetic_action_log_db):
    insert_action(
        synthetic_action_log_db, n_clusters=5,
        payload_json=json.dumps({"profile": "profile_4.json"}),
    )
    insert_action(
        synthetic_action_log_db, n_clusters=6,
        payload_json=json.dumps({"profile": "profile_4.json"}),
    )
    insert_action(synthetic_action_log_db, n_clusters=7, payload_json=None)  # pre-writer run

    svc = _service(synthetic_action_log_db)
    m = svc.compute_dashboard()
    profiles = {p.profile: p.n_runs for p in m.per_profile}
    assert profiles == {"profile_4.json": 2, NO_PROFILE: 1}
    assert svc.has_real_profiles(m) is True


def test_has_real_profiles_false_when_all_none(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, n_clusters=5, payload_json=None)
    insert_action(synthetic_action_log_db, n_clusters=6, payload_json=json.dumps({"profile": None}))
    svc = _service(synthetic_action_log_db)
    m = svc.compute_dashboard()
    assert svc.has_real_profiles(m) is False
    assert [p.profile for p in m.per_profile] == [NO_PROFILE]
