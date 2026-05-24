"""Shared test utilities and fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


def get_test_data_dir() -> Path:
    """Return the absolute path to the test_data/ directory at the project root."""
    return Path(__file__).parent.parent / "test_data"


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
