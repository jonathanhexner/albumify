"""spec-048 Phase 4 — `RunRow` dataclass fields == `ActionLog` columns.

The two definitions live in different modules but represent the same
shape. Drift between them is silent at runtime (NULL UI field, KeyError
in serialisation), so we lock the match at PR time.

If you add a column to ActionLog, add the field to RunRow (and vice
versa). If you intentionally want one to diverge from the other,
you need a deliberate mapping layer — and this test should be deleted
and replaced with an explicit assertion of the new mapping.
"""
from __future__ import annotations

from dataclasses import fields

from face_cluster.repositories.models.action_log import ActionLog
from face_cluster.run_history import RunRow


def test_runrow_fields_match_action_log_columns() -> None:
    runrow_names = {f.name for f in fields(RunRow)}
    action_log_names = set(ActionLog.__table__.columns.keys())

    missing_on_runrow = action_log_names - runrow_names
    extra_on_runrow = runrow_names - action_log_names

    assert not missing_on_runrow, (
        f"RunRow is missing fields present on ActionLog: {sorted(missing_on_runrow)}"
    )
    assert not extra_on_runrow, (
        f"RunRow has fields not present on ActionLog: {sorted(extra_on_runrow)}"
    )
