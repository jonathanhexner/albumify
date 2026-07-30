"""spec-057 T032 — arch guard: extracted writers stay slim.

After spec-057, each per-table writer lives in its own module under
``sim_bench/run_db/writers/`` or ``sim_bench/run_db/artifact_writers/``.
A writer doing more than one table's worth of work is a refactoring
regression; this test caps each writer file at 200 lines so the next
person to grow one is forced to think about whether a second writer
should be carved off.

Exempt from this check:
* ``sim_bench/run_db/exporter.py``    — the facade itself; carries the
  19-field ``RunExportInputs`` dataclass + 8 thin delegations. spec-059
  will trim further when RunStore moves to ORM.
* ``sim_bench/run_db/store.py``       — covered by spec-059 (RunStore on
  SQLAlchemy will shrink it to <450 LOC).
* ``sim_bench/run_db/_schema.py``     — pure DDL string constants
  (will be regenerated from ORM models in spec-058).
* ``__init__.py`` files                — empty package markers.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRITERS_DIRS = (
    REPO_ROOT / "sim_bench" / "run_db" / "writers",
    REPO_ROOT / "sim_bench" / "run_db" / "artifact_writers",
)
LOC_LIMIT = 200


def _iter_writer_modules():
    for d in WRITERS_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            yield p


@pytest.mark.parametrize(
    "writer_path",
    list(_iter_writer_modules()),
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_writer_module_loc_under_limit(writer_path: Path) -> None:
    """Each writer module under sim_bench/run_db/{writers,artifact_writers}/
    stays ≤ 200 LOC. If you need more, that's a signal to split."""
    loc = sum(1 for _ in writer_path.read_text(encoding="utf-8").splitlines())
    assert loc <= LOC_LIMIT, (
        f"{writer_path.relative_to(REPO_ROOT)} is {loc} LOC (cap: {LOC_LIMIT}). "
        f"Refactor: split into multiple writer modules or move helpers out."
    )
