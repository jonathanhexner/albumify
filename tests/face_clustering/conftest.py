"""Conftest for face_clustering tests.

Autouse fixture: redirect run_history_db to a per-session temp DB so that
pipeline E2E tests never pollute the real ~/.sim_bench/sim_bench.db.

test_pipeline_history_hook.py tests the DB hook explicitly and uses its own
per-test monkeypatch — that override takes priority over this session-level one.
"""
import pytest
import face_cluster.run_history_db as _rh


@pytest.fixture(autouse=True, scope="session")
def _isolate_run_history_db(tmp_path_factory):
    """Redirect all face_cluster DB writes to a temp file for this test session."""
    db_path = tmp_path_factory.mktemp("face_cluster_db") / "test_hist.db"
    _rh.init_table(db_path=db_path)

    original = _rh.get_db_path
    _rh.get_db_path = lambda: db_path
    yield db_path
    _rh.get_db_path = original
