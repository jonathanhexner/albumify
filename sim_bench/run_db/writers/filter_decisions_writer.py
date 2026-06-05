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


def write_filter_decisions_from_verdicts(conn: sqlite3.Connection, verdicts, faces) -> None:
    """Persist a list of ``QualityVerdict`` to filter_decisions (SIGHTING-093 G1).

    The v2 quality gate produces one ``QualityVerdict`` per face (``gates`` =
    per-gate ``GateResult`` + ``rejection_reason``) but the FC App v2 export
    path never wrote them, so the Quality / Excluded-Faces tabs had no data.
    This writer emits one row per (face, gate); ``verdicts[i]`` corresponds to
    ``faces[i]``. Complements :func:`write_filter_decisions` (the Albumify
    ``FilterContext`` path) — same table, verdict-shaped input.

    No-op when ``verdicts`` is falsy.
    """
    if not verdicts:
        return
    rows: List[Tuple] = []
    for face, verdict in zip(faces, verdicts):
        item_id = str(face.face_id)
        parent_id = getattr(face, "image_id", None) or getattr(face, "image_path", None)
        for gate_name, gate in verdict.gates.items():
            rejected = 0 if gate.passed else 1
            reason = "" if gate.passed else (verdict.rejection_reason or gate_name)
            rows.append((
                item_id, "face", parent_id, gate_name, rejected, reason,
                json.dumps({"value": gate.value, "threshold": gate.threshold}, default=str),
            ))
        # top_k_per_image is a disposition, not a per-gate GateResult — record
        # it as its own filter so "why isn't this face clustered?" is complete.
        if verdict.rejection_reason == "top_k_per_image":
            rows.append((
                item_id, "face", parent_id, "top_k_per_image", 1,
                "top_k_per_image", json.dumps({}),
            ))
    conn.executemany(
        "INSERT OR REPLACE INTO filter_decisions "
        "(item_id, item_type, parent_id, filter_name, rejected, reason, measured_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
