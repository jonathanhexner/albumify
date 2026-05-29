"""Shared helpers used by per-table writers (spec-057)."""
from __future__ import annotations

from typing import Optional

from sim_bench.run_db._errors import RunExporterError


def maybe_float(v) -> Optional[float]:
    """Best-effort float coercion. Returns None if conversion fails.

    Used by writers that translate model fields (which may be None / numpy
    scalars / strings) into SQLite REAL columns.
    """
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_sql(value, field_name: str):
    """Convert a Python value into a SQLite-storable form.

    SQLite has no boolean or infinity type; bools become ints, and `inf`
    is preserved via Python's REAL handling. None passes through. Raises
    RunExporterError for types the writer cannot represent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str)):
        return value
    if hasattr(value, "item"):
        return value.item()
    raise RunExporterError(
        f"unsupported type {type(value).__name__} for field {field_name!r}"
    )
