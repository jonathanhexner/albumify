"""spec-050 Phase 3 — `_entries_from_repo` produces the right slice.

Tests the data-fetch + mapping layer of the picker directly. Streamlit
rendering itself is covered by the AppTest E2E in Phase 5.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.face_clustering_v2.components.run_picker import _entries_from_repo
from face_cluster.repositories import (
    RunHistoryRepoConfig,
    RunHistoryRepository,
)


def _seed_runs(db_path: Path) -> None:
    repo = RunHistoryRepository(RunHistoryRepoConfig(db_path=db_path))

    # 3 v2 runs (newest last) — picker should return them newest-first.
    for i in range(1, 4):
        aid = repo.start_action("fc_app_v2_run", payload={
            "run_id": f"v2run{i:028d}",
            "output_dir": f"/runs/v2run{i:028d}",
            "source_album": f"album_{i}",
            "producer": "fc_app_v2",
        })
        repo.complete_action(aid, result_fields={"n_faces": 10 * i, "n_clusters": i})

    # 1 non-v2 run (legacy producer) — must be filtered out.
    aid = repo.start_action("face_cluster_run", payload={
        "run_id": "legacy_run_x",
        "output_dir": "/runs/legacy_run_x",
        "source_album": "legacy_album",
        "producer": "fc_app",
    })
    repo.complete_action(aid, result_fields={"n_faces": 99, "n_clusters": 1})

    # 1 v2 row with NULL output_dir — must be filtered out (not loadable).
    repo.start_action("fc_app_v2_run", payload={
        "source_album": "no_output_dir",
        "producer": "fc_app_v2",
    })


def test_entries_filters_to_v2_producer(tmp_path: Path) -> None:
    db = tmp_path / "isolated.db"
    _seed_runs(db)
    entries = _entries_from_repo(limit=20, db_path=db)
    assert all(e.album.startswith("album_") for e in entries), (
        f"non-v2 row leaked through: {[e.album for e in entries]}"
    )


def test_entries_ordered_newest_first(tmp_path: Path) -> None:
    db = tmp_path / "isolated.db"
    _seed_runs(db)
    entries = _entries_from_repo(limit=20, db_path=db)
    assert len(entries) == 3
    # Newest started_at first → albums seeded 1..3 should appear 3,2,1.
    assert [e.album for e in entries] == ["album_3", "album_2", "album_1"]


def test_entries_skip_rows_with_null_output_dir(tmp_path: Path) -> None:
    db = tmp_path / "isolated.db"
    _seed_runs(db)
    entries = _entries_from_repo(limit=20, db_path=db)
    # 4 v2 rows seeded total (3 complete + 1 with NULL output_dir), only 3 returned.
    assert len(entries) == 3
    assert all(str(e.output_dir) for e in entries)


def test_entries_carry_face_and_cluster_counts(tmp_path: Path) -> None:
    db = tmp_path / "isolated.db"
    _seed_runs(db)
    entries = _entries_from_repo(limit=20, db_path=db)
    by_album = {e.album: e for e in entries}
    assert by_album["album_3"].n_faces == 30
    assert by_album["album_3"].n_clusters == 3
    assert by_album["album_1"].n_faces == 10
    assert by_album["album_1"].n_clusters == 1


# ---------------------------------------------------------------------------
# spec-051 — orphan partitioning + label formatting
# ---------------------------------------------------------------------------

def test_partition_separates_orphan_from_loadable(tmp_path: Path) -> None:
    """Mix one real on-disk run + one orphan; verify _partition_entries
    splits them and that is_orphan is set correctly."""
    from app.face_clustering_v2.components.run_picker import (
        _all_entries_from_repo,
        _partition_entries,
    )

    db = tmp_path / "iso.db"
    repo = RunHistoryRepository(RunHistoryRepoConfig(db_path=db))

    # Loadable: create a real run_dir + face_clustering.db
    real_dir = tmp_path / "real_run"
    real_dir.mkdir()
    (real_dir / "face_clustering.db").touch()
    aid = repo.start_action("fc_app_v2_run", payload={
        "run_id": "realrun" + "0" * 25,
        "output_dir": str(real_dir),
        "source_album": "loadable_album",
        "producer": "fc_app_v2",
    })
    repo.complete_action(aid, result_fields={"n_faces": 7, "n_clusters": 2})

    # Orphan: output_dir points at a path that doesn't exist
    aid = repo.start_action("fc_app_v2_run", payload={
        "run_id": "orphan" + "0" * 26,
        "output_dir": str(tmp_path / "deleted_run"),
        "source_album": "orphan_album",
        "producer": "fc_app_v2",
    })
    repo.complete_action(aid, result_fields={"n_faces": 99, "n_clusters": 5})

    entries = _all_entries_from_repo(limit=20, db_path=db)
    loadable, orphan = _partition_entries(entries)

    assert len(loadable) == 1 and loadable[0].album == "loadable_album"
    assert loadable[0].is_orphan is False
    assert len(orphan) == 1 and orphan[0].album == "orphan_album"
    assert orphan[0].is_orphan is True


def test_format_label_marks_orphans(tmp_path: Path) -> None:
    """Orphan entries get a [missing] prefix; loadable entries do not."""
    from app.face_clustering_v2.components.run_picker import (
        RunPickerEntry,
        _format_label,
    )

    loadable = RunPickerEntry(
        run_id="abc12345" + "0" * 24, output_dir=Path("/x"), album="A",
        started_at="2026-05-28T10:00:00", n_faces=10, n_clusters=2,
        status="complete", is_orphan=False,
    )
    orphan = RunPickerEntry(
        run_id="def67890" + "0" * 24, output_dir=Path("/y"), album="B",
        started_at="2026-05-28T09:00:00", n_faces=5, n_clusters=1,
        status="complete", is_orphan=True,
    )

    assert not _format_label(loadable).startswith("[missing]")
    assert _format_label(orphan).startswith("[missing] ")
    assert "A" in _format_label(loadable)
    assert "B" in _format_label(orphan)
