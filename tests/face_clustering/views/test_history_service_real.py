"""spec-042 H1 — HistoryService integration tests against the real Budapest fixture.

These are smoke-level: they confirm the service works against the
actually-shaped action_log that the v2 app populates. Correctness is
covered by the synthetic-data unit tests in
``test_history_service_synthetic.py``; this file guards against
real-data-only failure modes (schema drift, encoding edge cases, etc.).

All tests skip cleanly via the ``v2_budapest_run_dir`` fixture when the
album / run dir isn't present on the dev machine.

Implementation note: the session-scoped autouse fixture in
``tests/face_clustering/conftest.py`` monkeypatches
``run_history_db.get_db_path`` to a temp DB. We bypass it by passing an
explicit ``db_path`` so these tests can see the user's actual action_log.
This is the ONLY place in the suite that does so, and it does it for a
clear reason — real-fixture smoke against real data.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from face_cluster.views.history import HistoryQuery, HistoryService


def _real_db_path() -> Path:
    """The user's actual action_log DB, bypassing the session monkeypatch."""
    return Path.home() / ".sim_bench" / "sim_bench.db"


@pytest.fixture(scope="module")
def real_service() -> HistoryService:
    """Service against the user's actual action_log DB.

    Module-scoped so all read-only tests share one connection — this is
    a smoke check, not a unit test.
    """
    db = _real_db_path()
    if not db.exists():
        pytest.skip(f"Real action_log DB not present at {db}")
    return HistoryService(db_path=db)


def test_list_runs_returns_at_least_one_row(real_service, v2_budapest_run_dir):
    """Verifies the real action_log has been populated."""
    rows = real_service.list_runs(HistoryQuery())
    assert len(rows) >= 1, (
        "Expected at least one row in the real action_log. "
        "Have you run scripts/run_v2.py yet?"
    )


def test_at_least_one_run_has_v2_producer(real_service, v2_budapest_run_dir):
    """At least one row should carry producer='fc_app_v2' from the spec-040 T4 wire-up."""
    # producer isn't on the typed RunRow yet; query through list_actions raw
    # to inspect the column. Use the same real DB path the service uses.
    from face_cluster import run_history_db
    actions = run_history_db.list_actions(limit=200, db_path=_real_db_path())
    producers = {a.get("producer") for a in actions}
    assert "fc_app_v2" in producers, (
        f"No fc_app_v2 producer rows in real action_log. "
        f"Producers seen: {producers}"
    )


def test_get_run_detail_on_a_real_run(real_service, v2_budapest_run_dir):
    """The most recent v2 run's detail must come back well-formed."""
    rows = real_service.list_runs(HistoryQuery())
    v2_rows = [r for r in rows if r.action_type and "v2" in r.action_type]
    if not v2_rows:
        pytest.skip("No v2 rows present in action_log to inspect")
    detail = real_service.get_run_detail(v2_rows[0].id)
    assert detail.row.id == v2_rows[0].id
    # config either populated from config_json or from pipeline_run.json
    # — assert the join works (returns a dict, even if empty).
    assert isinstance(detail.config, dict)


def test_list_albums_contains_budapest(real_service, v2_budapest_run_dir):
    """The Budapest album should appear in the distinct-albums list."""
    albums = real_service.list_albums()
    matching = [a for a in albums if "Budapest" in (a or "")]
    assert matching, (
        f"No Budapest-named album in real action_log. Found: {albums[:10]}"
    )
