"""Shared helpers used by per-table writers (spec-057)."""
from __future__ import annotations

from typing import Optional


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
