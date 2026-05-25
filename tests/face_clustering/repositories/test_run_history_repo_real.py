"""spec-043 Phase 3 — RunHistoryRepository real-fixture smoke.

3 cases against the user's actual ``~/.sim_bench/sim_bench.db``.
Smoke-level only — correctness is covered by the synthetic-data
test suite (33 cases) in ``test_run_history_repo_synthetic.py``.

These tests construct the Repository with an explicit ``db_path``
pointing at the real DB. They sidestep the session monkeypatch in
``tests/face_clustering/conftest.py`` the same way the existing
``test_history_service_real.py`` tests do.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from face_cluster.repositories import (
    RunHistoryCriteria,
    RunHistoryRepoConfig,
    RunHistoryRepository,
)


def _real_db_path() -> Path:
    """The user's actual action_log DB, bypassing the session monkeypatch."""
    return Path.home() / ".sim_bench" / "sim_bench.db"


@pytest.fixture(scope="module")
def real_repo() -> RunHistoryRepository:
    """Repository pointing at the real action_log DB."""
    db = _real_db_path()
    if not db.exists():
        pytest.skip(f"Real action_log DB not present at {db}")
    return RunHistoryRepository(RunHistoryRepoConfig(db_path=db))


def test_find_returns_at_least_one_row(real_repo, v2_budapest_run_dir):
    """The real action_log has been populated by the v2 app at least once."""
    rows = real_repo.find(RunHistoryCriteria())
    assert len(rows) >= 1


def test_find_by_producer_v2(real_repo, v2_budapest_run_dir):
    """At least one row carries producer='fc_app_v2' (spec-040 T4 wiring)."""
    rows = real_repo.find(RunHistoryCriteria(producer="fc_app_v2"))
    assert len(rows) >= 1, (
        "Expected fc_app_v2 rows in real action_log. Run scripts/run_v2.py first."
    )


def test_distinct_albums_contains_budapest(real_repo, v2_budapest_run_dir):
    """The Budapest album appears in distinct_albums."""
    albums = real_repo.distinct_albums()
    matching = [a for a in albums if "Budapest" in (a or "")]
    assert matching, f"No Budapest-named album. Found: {albums[:10]}"
