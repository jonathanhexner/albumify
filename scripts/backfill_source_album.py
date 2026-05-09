"""One-shot migration: backfill source_album for action_log rows where it is missing.

Two inference strategies:

1. New-style paths  results/<album>/<run_dir>  — album = parent.name
2. Old-style paths  results/<run_dir>          — album = strip_run_suffixes(run_dir)
   Suffixes stripped: _recluster_N, _recluster_HHMMSS, _merge_recluster_*, _merge

Usage:
    .venv/Scripts/python scripts/backfill_source_album.py [--apply]

Without --apply the script runs in dry-run mode and only prints what it would do.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster.run_history_db import get_db_path, _connect, init_table

# Folder names that look like a results root, not an album name
_SKIP_PARENT_NAMES = {"results", ".", "", ".."}

# Run-name suffixes to strip (applied iteratively until stable)
_SUFFIX_RE = re.compile(
    r'(_recluster_[\w]+|_merge_recluster_[\w]+|_merge)$',
    re.IGNORECASE,
)

# Names that are clearly merge artifacts with no base album encoded
_SKIP_NAMES_PREFIX = {"merge_snap", "merge_remerge", "merge_"}


def _strip_run_suffixes(name: str) -> str:
    """Iteratively strip run suffixes until name is stable."""
    prev = None
    while prev != name:
        prev = name
        name = _SUFFIX_RE.sub('', name)
    return name


def _looks_like_tmp(name: str) -> bool:
    """Return True for temp-dir names like tmpnb6q9nny."""
    return bool(re.match(r'^tmp[a-z0-9]{6,}$', name, re.IGNORECASE))


def _infer_album(output_dir: str) -> str | None:
    """Infer album name from output_dir.

    Strategy 1 (new-style): .../results/<album>/<run_dir>  → album = parent.name
    Strategy 2 (old-style): .../results/<run_dir>          → album = strip suffixes from dir.name
    """
    if not output_dir:
        return None
    try:
        p = Path(output_dir)
        parent_name = p.parent.name
        dir_name    = p.name

        # Strategy 1: well-structured session path
        if parent_name and parent_name not in _SKIP_PARENT_NAMES:
            if _looks_like_tmp(parent_name):
                return None          # temp dir from test run
            return parent_name

        # Strategy 2: old flat structure — parent is "results" or similar root
        if not dir_name:
            return None

        # Skip obvious merge artifacts
        for prefix in _SKIP_NAMES_PREFIX:
            if dir_name.lower().startswith(prefix):
                return None

        album = _strip_run_suffixes(dir_name)
        if not album or album.lower().startswith('merge_'):
            return None
        return album

    except Exception:
        return None


def backfill(apply: bool = False, db_path: Path | None = None) -> None:
    init_table(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, output_dir, source_album FROM action_log "
            "WHERE source_album IS NULL OR source_album = '' OR source_album = '(unknown)'"
        ).fetchall()

    if not rows:
        print("No rows need backfilling.")
        return

    updates: list[tuple[str, int]] = []
    skipped: list[tuple[int, str | None]] = []

    for row in rows:
        row_id   = row["id"]
        out_dir  = row["output_dir"] or ""
        inferred = _infer_album(out_dir)
        if inferred:
            updates.append((inferred, row_id))
            print(f"  Row {row_id:5d}: '{out_dir}' -> album='{inferred}'")
        else:
            skipped.append((row_id, out_dir or None))
            print(f"  Row {row_id:5d}: SKIP  output_dir='{out_dir}' (cannot infer)")

    print()
    print(f"Will update {len(updates)} rows, skip {len(skipped)} rows.")

    if not apply:
        print("Dry-run mode. Pass --apply to write changes.")
        return

    if not updates:
        print("Nothing to apply.")
        return

    with _connect(db_path) as conn:
        conn.executemany(
            "UPDATE action_log SET source_album = ? WHERE id = ?",
            updates,
        )
        conn.commit()
    print(f"Done. Updated {len(updates)} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill source_album in action_log")
    parser.add_argument("--apply", action="store_true",
                        help="Write changes to DB (default is dry-run)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to sim_bench.db (default: ~/.sim_bench/sim_bench.db)")
    args = parser.parse_args()

    db = Path(args.db) if args.db else None
    backfill(apply=args.apply, db_path=db)
