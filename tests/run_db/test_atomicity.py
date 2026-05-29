"""spec-057 Phase 4 (T040) — single-transaction atomicity test.

Proves the writer split preserves the all-or-nothing contract: when a
mid-export writer raises, the run dir is left in a consistent state.
The DB file is removed before the transaction is reopened, so the
post-failure run dir should have no usable face_clustering.db.

This is the test that would catch a future refactor that accidentally
commits partial state (e.g., calling conn.commit() per writer instead
of once at the end).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sim_bench.run_db._errors import RunExporterError
from sim_bench.run_db.exporter import RunExporter

# Reuse spec-057 Phase 0 fixture and helpers.
from tests.run_db.test_split_equivalence import (
    golden_exporter_input,  # noqa: F401 — pytest fixture
)


def _has_data(db_path: Path) -> bool:
    """True if face_clustering.db exists AND has any row in any data table."""
    if not db_path.is_file():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            for table in ("faces", "clusters", "cluster_assignments", "merge_decisions"):
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if n > 0:
                    return True
            return False
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        # If the file exists but isn't valid SQLite, treat as no usable data.
        return False


def test_writer_failure_leaves_no_partial_db(
    golden_exporter_input, monkeypatch
):
    """Monkeypatch merges_writer.write_merges to raise; assert the DB has
    no committed rows. This is the single-transaction contract: every
    writer either commits as a whole or none of them do.
    """
    # Force the failure deep enough that earlier writers have already run
    # in-transaction.
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated mid-export failure")

    monkeypatch.setattr(
        "sim_bench.run_db.writers.merges_writer.write_merges",
        _boom,
    )

    out_dir = golden_exporter_input["output_dir"]
    with pytest.raises(RuntimeError, match="simulated mid-export failure"):
        RunExporter(out_dir).export(
            faces=golden_exporter_input["faces"],
            base_cluster_result=golden_exporter_input["base"],
            merged_cluster_result=golden_exporter_input["merged"],
            core_indices=golden_exporter_input["core_indices"],
            merge_log=golden_exporter_input["merge_log"],
            merge_metadata={"n_iterations": 1},
            config=golden_exporter_input["config"],
            source_album="atomicity_test",
            producer="fc_app",
            run_id="atomic_run",
            started_at="2026-05-29T00:00:00",
            finished_at="2026-05-29T00:00:01",
        )

    db_path = out_dir / "face_clustering.db"
    assert not _has_data(db_path), (
        "Mid-export failure left committed rows in face_clustering.db — "
        "single-transaction contract broken."
    )
