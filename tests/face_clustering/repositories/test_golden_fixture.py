"""Phase 4 — golden fixture has the shape the equivalence test depends on.

Catches accidental drift if someone regenerates the fixture differently.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

GOLDEN_DB = Path(__file__).resolve().parents[1] / "fixtures" / "golden_run_history.db"


def test_golden_fixture_shape():
    assert GOLDEN_DB.exists(), f"Missing fixture: {GOLDEN_DB}. Run tests/face_clustering/fixtures/rebuild_golden.py"

    with sqlite3.connect(str(GOLDEN_DB)) as conn:
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]
        assert total >= 20, f"expected >=20 rows, got {total}"

        statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM action_log")}
        assert statuses == {"running", "complete", "failed"}

        n_null_album = conn.execute(
            "SELECT COUNT(*) FROM action_log WHERE source_album IS NULL"
        ).fetchone()[0]
        assert n_null_album >= 1, "need >=1 row with NULL source_album"

        n_unicode = conn.execute(
            "SELECT COUNT(*) FROM action_log WHERE comment LIKE '%ünicode%'"
        ).fetchone()[0]
        assert n_unicode >= 1, "need >=1 row with unicode comment"

        n_parent = conn.execute(
            "SELECT COUNT(*) FROM action_log WHERE parent_run_id IS NOT NULL"
        ).fetchone()[0]
        assert n_parent >= 3, "need >=3 rows with parent_run_id chain"

        action_types = {r[0] for r in conn.execute("SELECT DISTINCT action_type FROM action_log")}
        for required in ("face_cluster_run", "recluster", "merge_apply", "profile_save", "ml_training"):
            assert required in action_types, f"missing action_type {required}"

        producers = {r[0] for r in conn.execute(
            "SELECT DISTINCT producer FROM action_log WHERE producer IS NOT NULL"
        )}
        assert {"albumify", "fc_app_v2", "fc_app"}.issubset(producers)
