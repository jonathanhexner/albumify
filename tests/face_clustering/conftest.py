"""Conftest for face_clustering tests.

Autouse fixture: redirect ``run_history_db.get_db_path()`` to a per-session
temp DB so that pipeline E2E tests don't pollute the real
``~/.sim_bench/sim_bench.db``.

spec-043 note: the Repository pattern (B0 of the architecture standards)
makes this fixture *less* critical than it used to be — tests that
construct a ``RunHistoryRepository(RunHistoryRepoConfig(db_path=...))``
explicitly are already isolated by construction. The fixture remains as
a safety net for tests that use the default-path Repository
(``RunHistoryRepository()`` with no Config) — they go through
``run_history_db.get_db_path()`` which this fixture monkeypatches.

Real-fixture tests that intentionally want the production DB construct
the Repository with an explicit ``db_path=Path.home() / ".sim_bench" / "sim_bench.db"``
to bypass this fixture.

``test_pipeline_history_hook.py`` uses its own per-test monkeypatch which
takes priority over this session-level one.
"""
import pytest
import face_cluster.run_history_db as _rh


@pytest.fixture(autouse=True, scope="session")
def _isolate_run_history_db(tmp_path_factory):
    """Redirect default-Repository DB access to a temp file for the session.

    Repository instances constructed with an explicit ``db_path`` are
    unaffected (they don't go through ``get_db_path``).
    """
    db_path = tmp_path_factory.mktemp("face_cluster_db") / "test_hist.db"
    _rh.init_table(db_path=db_path)

    original = _rh.get_db_path
    _rh.get_db_path = lambda: db_path
    yield db_path
    _rh.get_db_path = original
