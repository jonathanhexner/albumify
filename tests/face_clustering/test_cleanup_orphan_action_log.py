"""spec-051 Phase 3 — cleanup CLI tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from face_cluster.repositories import RunHistoryRepoConfig, RunHistoryRepository
from scripts.cleanup_orphan_action_log import main as cleanup_main


def _seed(db: Path) -> tuple[int, int]:
    """Seed a mix of orphan + real rows. Returns (n_orphan, n_real)."""
    repo = RunHistoryRepository(RunHistoryRepoConfig(db_path=db))

    orphan_dirs = [
        r"C:\Users\u\AppData\Local\Temp\pytest-of-u\pytest-1\test_a0\out",
        r"C:\Users\u\AppData\Local\Temp\pytest-of-u\pytest-2\test_b0\out",
        "/tmp/some-other-tmp/run",
        "/var/folders/xx/tmpfs/r",
    ]
    real_dirs = [
        r"D:\Photos\Budapest\run_001",
        "/home/user/.sim_bench/runs/real_run",
    ]

    for i, d in enumerate(orphan_dirs):
        aid = repo.start_action("fc_app_v2_run", payload={
            "run_id": f"orphan{i}" + "0" * 24,
            "output_dir": d,
            "source_album": f"orphan_album_{i}",
            "producer": "fc_app_v2",
        })
        repo.complete_action(aid, result_fields={"n_faces": 1, "n_clusters": 1})

    for i, d in enumerate(real_dirs):
        aid = repo.start_action("fc_app_v2_run", payload={
            "run_id": f"real{i}" + "0" * 26,
            "output_dir": d,
            "source_album": f"real_album_{i}",
            "producer": "fc_app_v2",
        })
        repo.complete_action(aid, result_fields={"n_faces": 1, "n_clusters": 1})

    return len(orphan_dirs), len(real_dirs)


def _row_count(db: Path) -> int:
    with sqlite3.connect(str(db)) as c:
        return c.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]


def test_dry_run_makes_no_changes(tmp_path: Path, capsys) -> None:
    db = tmp_path / "iso.db"
    n_orphan, n_real = _seed(db)
    before = _row_count(db)

    rc = cleanup_main(["--db", str(db)])
    assert rc == 0
    assert _row_count(db) == before, "dry-run must not alter the DB"
    out = capsys.readouterr().out
    assert f"Orphan rows found: {n_orphan}" in out
    assert "Dry-run" in out


def test_apply_deletes_only_orphans(tmp_path: Path) -> None:
    db = tmp_path / "iso.db"
    n_orphan, n_real = _seed(db)
    before = _row_count(db)
    assert before == n_orphan + n_real

    rc = cleanup_main(["--db", str(db), "--apply", "--yes-i-counted", str(n_orphan)])
    assert rc == 0
    assert _row_count(db) == n_real, (
        "should delete exactly the orphan rows and preserve the rest"
    )


def test_apply_with_wrong_count_aborts(tmp_path: Path, capsys) -> None:
    db = tmp_path / "iso.db"
    n_orphan, n_real = _seed(db)
    before = _row_count(db)

    rc = cleanup_main(["--db", str(db), "--apply", "--yes-i-counted", str(n_orphan + 1)])
    assert rc == 3, "mismatched count must abort"
    assert _row_count(db) == before, "no rows should be touched on abort"


def test_apply_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "iso.db"
    n_orphan, n_real = _seed(db)

    rc1 = cleanup_main(["--db", str(db), "--apply", "--yes-i-counted", str(n_orphan)])
    assert rc1 == 0

    # Second apply: zero orphans, --yes-i-counted 0 required.
    rc2 = cleanup_main(["--db", str(db), "--apply", "--yes-i-counted", "0"])
    assert rc2 == 0
    assert _row_count(db) == n_real


def test_filter_covers_cross_platform_tmp_patterns(tmp_path: Path, capsys) -> None:
    """The orphan filter must match Windows AppData\\Local\\Temp, POSIX /tmp/,
    and pytest-of-*/pytest-N paths."""
    db = tmp_path / "iso.db"
    _seed(db)  # seeds 4 distinct tmp-path styles
    cleanup_main(["--db", str(db)])  # dry-run
    out = capsys.readouterr().out
    # Each of the 4 orphan styles must appear in the output.
    assert "AppData" in out
    assert "/tmp/" in out
    assert "tmpfs" in out
    assert "pytest" in out
