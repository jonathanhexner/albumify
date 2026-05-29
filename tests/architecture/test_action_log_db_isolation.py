"""spec-051 — meta-guard: the session-wide ``isolate_action_log_db``
autouse fixture must remain in ``tests/conftest.py``.

If someone removes the fixture (or downgrades it from session-scoped
or autouse), this test fails before any other test even runs.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tests.conftest as project_conftest


def test_isolate_action_log_db_fixture_is_session_autouse() -> None:
    fixture = getattr(project_conftest, "isolate_action_log_db", None)
    assert fixture is not None, (
        "tests/conftest.py must define `isolate_action_log_db` (spec-051). "
        "Without it, any test constructing RunHistoryRepository() with no "
        "db_path will write to the user's real ~/.sim_bench/sim_bench.db."
    )
    marker = getattr(fixture, "_fixture_function_marker", None)
    assert marker is not None, "isolate_action_log_db is not a pytest fixture"
    assert marker.scope == "session", (
        f"isolate_action_log_db must be scope='session' (got {marker.scope!r}). "
        "Function-scope would re-create the redirect every test, leaving a "
        "gap during teardown."
    )
    assert marker.autouse is True, (
        "isolate_action_log_db must be autouse=True. Without autouse, tests "
        "must opt in — defeating the purpose of the guardrail."
    )


def test_isolate_action_log_db_targets_paths_module() -> None:
    """The fixture body must monkeypatch ``face_cluster._paths.default_db_path``
    specifically. Targeting any other attribute (e.g., the deprecated
    ``run_history_db.get_db_path``) leaves the production DB exposed."""
    src = inspect.getsource(project_conftest.isolate_action_log_db)
    assert "_paths.default_db_path" in src or "default_db_path" in src, (
        "Fixture must target face_cluster._paths.default_db_path. "
        "Source:\n" + src
    )
