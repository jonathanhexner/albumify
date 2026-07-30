"""spec-059 T040 / AC1 — no raw SQL in the per-run-DB read layer.

Locks the SQLAlchemy migration. The two read layers ``sim_bench/run_db/store.py``
and ``sim_bench/db/face_clustering/cluster_analysis_repo.py`` must not contain
any raw SQL strings — every read goes through ORM models from
``sim_bench.run_db.models``.

The single allowed raw-SQL site is ``RunStore._load_and_validate``'s
``PRAGMA user_version`` check, which must run before any ORM machinery
touches the DB (locked decision #3). It uses ``PRAGMA``, not ``SELECT|INSERT|
UPDATE|DELETE``, so the grep below already excludes it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Match raw SQL verbs at word boundaries. Uppercase with a trailing space
# matches the ANSI style used throughout the repo's hand-DDL; lowercase
# matches potential leftovers.
_RAW_SQL_PATTERN = re.compile(
    r"\b(?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "rel_path",
    [
        "sim_bench/run_db/store.py",
        "sim_bench/db/face_clustering/cluster_analysis_repo.py",
    ],
)
def test_no_raw_sql_strings(rel_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / rel_path).read_text(encoding="utf-8")
    offenders: list[tuple[int, str]] = []
    for i, line in enumerate(src.splitlines(), start=1):
        # Skip comments and docstrings is too lossy; we only flag string
        # literals that contain SQL verbs. Detect by requiring at least one
        # quote on the line *and* a verb match.
        if ('"' in line or "'" in line) and _RAW_SQL_PATTERN.search(line):
            offenders.append((i, line.strip()))
    assert not offenders, (
        f"{rel_path} contains raw SQL strings (spec-059 AC1 forbids these — "
        f"use ORM models from sim_bench.run_db.models instead):\n"
        + "\n".join(f"  L{n}: {text}" for n, text in offenders)
    )
