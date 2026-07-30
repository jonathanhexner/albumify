"""Shared test utilities and fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


def get_test_data_dir() -> Path:
    """Return the absolute path to the test_data/ directory at the project root."""
    return Path(__file__).parent.parent / "test_data"


# ---------------------------------------------------------------------------
# spec-051 — production DB isolation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def isolate_action_log_db(tmp_path_factory):
    """Redirect ``face_cluster._paths.default_db_path`` to a per-session
    tmp file so NO test can accidentally write to the user's real
    ``~/.sim_bench/sim_bench.db``.

    Spec-051 ships this fixture because the test suite had been writing
    orphan rows into the production action_log for an unknown number of
    days (surfaced when spec-050's v2 picker auto-selected one of them).

    Opt-out: tests that genuinely need the real DB construct
    ``RunHistoryRepository`` (or any consumer) with an **explicit**
    ``db_path`` argument — they bypass ``default_db_path()`` entirely.
    See ``tests/face_clustering/repositories/test_run_history_repo_real.py``
    for the canonical opt-out pattern.

    Per-test fixtures may still monkeypatch ``_paths.default_db_path``
    on top of this autouse setup; pytest's ``monkeypatch.setattr`` runs
    later in the fixture chain and wins.
    """
    import face_cluster._paths as _paths
    fake_db = tmp_path_factory.mktemp("isolated_action_log") / "sim_bench.db"
    orig = _paths.default_db_path
    _paths.default_db_path = lambda: fake_db
    try:
        yield fake_db
    finally:
        _paths.default_db_path = orig


# ---------------------------------------------------------------------------
# spec-042 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def v2_budapest_run_dir() -> Path:
    """Resolves to the most recent v2 Budapest run dir.

    Skips the test cleanly when no such dir exists (e.g., CI without
    the user's album). All view-service "real fixture" integration tests
    depend on this; they're smoke-level, not the primary correctness
    signal — the synthetic-data unit tests carry that.
    """
    base = Path.home() / ".sim_bench" / "runs"
    if not base.exists():
        pytest.skip(f"No ~/.sim_bench/runs/ dir; run scripts/run_v2.py first.")
    candidates = sorted(
        base.glob("v2_budapest_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        pytest.skip(
            "No v2_budapest_* run dir found under ~/.sim_bench/runs/. "
            "Run scripts/run_v2.py against D:/Budapest2025_Google first."
        )
    return candidates[0]


@pytest.fixture
def synthetic_action_log_db(tmp_path: Path) -> Path:
    """Empty action_log SQLite DB at ``tmp_path/sim_bench.db`` with the
    current schema applied.

    Tests insert rows via ``tests/face_clustering/views/_seed.py`` helpers.
    Each test gets a fresh DB (function-scoped); isolation is automatic.
    """
    from face_cluster.run_history_db import init_table
    db_path = tmp_path / "sim_bench.db"
    init_table(db_path=db_path)
    return db_path
