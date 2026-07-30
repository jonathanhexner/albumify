"""spec-050 Phase 0 — `allocate_run_dir` unit tests."""
from __future__ import annotations

import re
from pathlib import Path

from face_cluster.run_layout import allocate_run_dir


_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def test_returns_unique_paths(tmp_path: Path) -> None:
    seen: set[Path] = set()
    for _ in range(10):
        run_dir, _ = allocate_run_dir(tmp_path, album_slug="album_a")
        assert run_dir not in seen
        seen.add(run_dir)


def test_dir_exists_after_allocation(tmp_path: Path) -> None:
    run_dir, _ = allocate_run_dir(tmp_path, album_slug="album_a")
    assert run_dir.is_dir()


def test_run_id_is_lowercase_hex32_and_matches_dirname(tmp_path: Path) -> None:
    run_dir, run_id = allocate_run_dir(tmp_path, album_slug="album_a")
    assert _HEX32.fullmatch(run_id), f"run_id not 32 hex chars: {run_id!r}"
    assert run_dir.name == run_id


def test_album_slug_is_not_in_path(tmp_path: Path) -> None:
    """Documentation test — flat-uuid layout. If you change this, you're
    re-introducing per-album subdirs; the picker reads from action_log,
    not from disk, so adding an album dir layer only adds slug-sanitation
    complexity. See spec-050 Locked decision §1."""
    run_dir, _ = allocate_run_dir(tmp_path, album_slug="My Album / With Spaces")
    assert "My Album" not in str(run_dir)
    assert run_dir.parent == tmp_path
