"""spec-044 — drift guards for the RunHistoryRepository column registry.

Permanent invariants (four ``test_*`` functions) catch column-registry
drift at PR time. Adding a column to ``_COLUMNS`` without also extending
``RunRow`` / ``RunHistoryCriteria`` will fail one of these tests.
"""
from __future__ import annotations

import dataclasses

from face_cluster.repositories.run_history_repo import (
    _COLUMNS,
    _FILTERABLE_ALIASES,
    _FILTERABLE_FIELDS,
    RunHistoryCriteria,
)
from face_cluster.run_history import RunRow


# ===========================================================================
# PERMANENT — survive past Phase 3 / Phase 5
# ===========================================================================

def test_runrow_fields_match_columns_registry():
    """Every column in ``_COLUMNS`` has a matching ``RunRow`` field.

    Direction: registry ⊆ RunRow. RunRow may have extra fields
    (computed properties, future un-persisted fields) but every
    persisted column MUST be accessible via RunRow.
    """
    column_names = {c.name for c in _COLUMNS}
    runrow_fields = {f.name for f in dataclasses.fields(RunRow)}
    missing_in_runrow = column_names - runrow_fields
    assert not missing_in_runrow, (
        f"_COLUMNS declares {sorted(missing_in_runrow)!r} but RunRow doesn't "
        f"expose them. Add fields to face_cluster.run_history.RunRow + the "
        f"_row_to_run_row mapper."
    )


def test_filterable_columns_have_matching_criteria_fields():
    """Every column with ``filterable=True`` has a matching
    ``RunHistoryCriteria`` field, or is documented as a special-case
    alias in ``_FILTERABLE_ALIASES``.
    """
    criteria_fields = {f.name for f in dataclasses.fields(RunHistoryCriteria)}
    for col_name in _FILTERABLE_FIELDS:
        target = _FILTERABLE_ALIASES.get(col_name, col_name)
        assert target in criteria_fields, (
            f"Column {col_name!r} is marked filterable but the matching "
            f"RunHistoryCriteria field {target!r} doesn't exist. Either add "
            f"the field, or document the alias in _FILTERABLE_ALIASES."
        )


def test_initial_and_migration_partition_is_complete():
    """No column appears in both the initial CREATE list AND the migration
    list. Every column has an explicit ``initial`` flag (no None / missing)."""
    initial = [c.name for c in _COLUMNS if c.initial]
    migration = [c.name for c in _COLUMNS if not c.initial]
    overlap = set(initial) & set(migration)
    assert not overlap, (
        f"Columns {sorted(overlap)!r} appear in both initial and migration. "
        f"A column is either created in the original schema or added later, "
        f"not both."
    )
    # Sanity: at least one of each kind exists (otherwise the partitioning
    # is meaningless on this schema).
    assert initial, "No initial columns — _COLUMNS is empty?"
    assert migration, (
        "No migration columns — the spec-013 / spec-040 columns should "
        "remain ALTER-added rather than baked into the initial schema, "
        "so older DBs continue to migrate forward correctly."
    )


def test_only_nullable_hot_fields():
    """Hot columns must be nullable.

    ``start_action`` writes ``payload.get(col_name)`` (which is None when
    the key is missing) for every hot column. A NOT NULL hot field would
    crash on insert when the payload doesn't provide it.
    """
    bad = [c.name for c in _COLUMNS if c.hot and not c.nullable]
    assert not bad, (
        f"Hot columns {bad!r} are NOT NULL. start_action and complete_action "
        f"need to be able to leave them NULL when no payload value is supplied."
    )


