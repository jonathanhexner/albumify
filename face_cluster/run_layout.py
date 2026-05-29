"""Per-run output directory allocation for the v2 FC App.

One job: take a base directory and an album name, return a fresh
``<base>/<uuid4-hex>/`` directory plus the ``run_id`` (= the UUID).
Album is not used in the path — it lives in ``action_log.source_album``
and is the user-facing label in the run picker.

Why flat: the action_log is the index. Disk layout doesn't need to be
human-browsable; the History tab and the v2 Clusters tab's picker both
list runs by album anyway. Flat layout keeps the allocator trivial and
avoids album-slug sanitation rules.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def allocate_run_dir(base_dir: Path, album_slug: str) -> tuple[Path, str]:
    """Allocate a fresh per-run output directory.

    Args:
        base_dir: parent directory for all per-run dirs (e.g.
            ``~/.sim_bench/runs``). Created if missing.
        album_slug: free-form album label. Recorded in
            ``action_log.source_album`` by the caller. NOT used in the
            returned path — see module docstring.

    Returns:
        ``(run_dir, run_id)`` where ``run_dir == base_dir / run_id``
        and ``run_id`` is a 32-char lowercase hex UUID4. The directory
        is created with ``exist_ok=False`` — a collision is a hard
        crash, not silent reuse.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


__all__ = ["allocate_run_dir"]
