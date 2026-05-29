"""Per-table writer for `filter_decisions` (spec-032 / spec-057).

Extracted from RunExporter._write_filter_decisions.
"""
from __future__ import annotations

import json
import sqlite3
from typing import List, Tuple


def write_filter_decisions(conn: sqlite3.Connection, filters) -> None:
    """Persist FilterContext to the filter_decisions table.

    No-op when filters is None or empty — keeps the schema consistent
    for runs that don't use the new contract yet (P1 dual-write window).
    """
    if filters is None or len(filters) == 0:
        return
    rows: List[Tuple] = []
    for item, decision in filters.all_decisions():
        rows.append((
            item.item_id,
            item.item_type,
            item.parent_id,
            decision.filter_name,
            1 if decision.rejected else 0,
            decision.reason,
            json.dumps(decision.measured, default=str),
        ))
    conn.executemany(
        "INSERT INTO filter_decisions "
        "(item_id, item_type, parent_id, filter_name, rejected, reason, measured_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
