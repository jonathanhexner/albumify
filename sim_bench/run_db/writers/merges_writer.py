"""Per-table writer for `merge_decisions` (spec-057).

Extracted from RunExporter._write_merges.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Tuple

from face_cluster.types import MergeDecisionRow
from sim_bench.run_db.writers._common import to_sql


def write_merges(conn: sqlite3.Connection, merge_log: List[Dict]) -> None:
    if not merge_log:
        return
    field_order = MergeDecisionRow.field_names()
    rows: List[Tuple] = []
    for entry in merge_log:
        rows.append(tuple(to_sql(entry[name], name) for name in field_order))
    placeholders = ",".join(["?"] * len(field_order))
    conn.executemany(
        f"INSERT INTO merge_decisions VALUES ({placeholders})",
        rows,
    )
