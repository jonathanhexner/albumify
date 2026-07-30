"""Artifact writer for pipeline_run.json (spec-057).

Extracted from RunExporter._write_pipeline_run_json. This is the small
human-readable run pointer file (FR-007); carries no config blob — just
the pointer fields a consumer needs to locate the per-run DB.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from sim_bench.run_db._schema import SCHEMA_VERSION


def write_pipeline_run(
    output_dir: Path,
    *,
    run_id: str,
    source_album: str,
    producer: str,
    parent_run_id: Optional[str],
    started_at: str,
    finished_at: str,
) -> None:
    payload = {
        "run_id": run_id,
        "source_album": source_album,
        "producer": producer,
        "parent_run_id": parent_run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "db_path": "face_clustering.db",
    }
    path = output_dir / "pipeline_run.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
