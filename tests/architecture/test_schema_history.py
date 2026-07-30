"""spec-054 — every SCHEMA_VERSION bump must document itself.

If someone bumps ``SCHEMA_VERSION`` (e.g. to 6) but forgets to add a
``SCHEMA_HISTORY[6] = "what changed"`` entry, this test fails. The test
that breaks on the next bump is the prompt to update the history.
"""
from __future__ import annotations

from sim_bench.run_db._schema import SCHEMA_HISTORY, SCHEMA_VERSION


def test_schema_version_is_in_history() -> None:
    assert SCHEMA_VERSION in SCHEMA_HISTORY, (
        f"SCHEMA_VERSION={SCHEMA_VERSION} but SCHEMA_HISTORY has no "
        f"entry for it. Add an entry describing what changed."
    )


def test_schema_version_equals_max_history_key() -> None:
    assert SCHEMA_VERSION == max(SCHEMA_HISTORY), (
        f"SCHEMA_VERSION={SCHEMA_VERSION} does not match the highest "
        f"SCHEMA_HISTORY key ({max(SCHEMA_HISTORY)}). SCHEMA_VERSION is "
        f"meant to be derived from SCHEMA_HISTORY."
    )


def test_schema_history_keys_are_contiguous() -> None:
    """No gaps in the version sequence — every version we ever shipped
    has a documentation entry. If we ever need a gap (skipped version),
    delete this test and document why."""
    keys = sorted(SCHEMA_HISTORY)
    expected = list(range(keys[0], keys[-1] + 1))
    assert keys == expected, (
        f"SCHEMA_HISTORY keys have a gap: {keys}. Expected contiguous: {expected}."
    )


def test_every_history_entry_has_a_non_empty_description() -> None:
    empty = [v for v, desc in SCHEMA_HISTORY.items() if not (desc or "").strip()]
    assert not empty, f"SCHEMA_HISTORY versions with empty description: {empty}"
