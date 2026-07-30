"""spec-051 — One-shot cleanup of orphan rows in the action_log table.

Orphans are rows whose ``output_dir`` points at a tmp / non-existent
directory. The common cause is test fixtures that bypassed the
spec-051 session-wide isolation guardrail before it existed.

Usage
-----

Dry-run (default — never touches the DB):

    .venv/Scripts/python scripts/cleanup_orphan_action_log.py

Apply (must include the exact orphan count to confirm intent):

    .venv/Scripts/python scripts/cleanup_orphan_action_log.py \
        --apply --yes-i-counted 18

The script targets ``face_cluster._paths.default_db_path()`` — the
real ``~/.sim_bench/sim_bench.db`` in normal use. Tests redirect this
via the autouse ``isolate_action_log_db`` fixture in
``tests/conftest.py``.

Exit codes: 0 on success (dry-run or apply); 2 on argument error or
missing DB; 3 when ``--apply`` is requested but ``--yes-i-counted``
does not match the actual orphan count.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

# Make face_cluster.* importable when invoked from the repo root.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from face_cluster._paths import default_db_path  # noqa: E402


# Path fragments that mark a row as an orphan candidate. Cross-platform.
_ORPHAN_PATTERNS: tuple[str, ...] = (
    "pytest",                # pytest-of-... tmp dirs
    "AppData\\Local\\Temp",  # Windows user tmp
    "AppData/Local/Temp",    # Windows user tmp, forward-slash form
    "\\Temp\\",              # Generic Windows Temp
    "/Temp/",                # Generic, forward-slash
    "/tmp/",                 # POSIX
    "tmpfs",                 # POSIX tmpfs
)


def _select_orphans(conn: sqlite3.Connection) -> List[Tuple[int, str, str, str, str]]:
    """Return ``[(id, run_id, source_album, output_dir, status), ...]``
    for every action_log row whose output_dir matches an orphan pattern."""
    where = " OR ".join(["output_dir LIKE ?"] * len(_ORPHAN_PATTERNS))
    params = [f"%{p}%" for p in _ORPHAN_PATTERNS]
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT id, run_id, source_album, output_dir, status FROM action_log "
        f"WHERE output_dir IS NOT NULL AND ({where}) "
        f"ORDER BY id DESC",
        params,
    ).fetchall()
    return [(r["id"], r["run_id"] or "", r["source_album"] or "",
             r["output_dir"], r["status"] or "") for r in rows]


def _print_rows(rows: List[Tuple[int, str, str, str, str]], *, limit: int = 20) -> None:
    if not rows:
        print("  (none)")
        return
    for rid, run_id, album, out_dir, status in rows[:limit]:
        short_run = (run_id[:8] + "…") if len(run_id) > 8 else run_id
        print(f"  id={rid:>4}  status={status:<9}  album={album!r:<30}  "
              f"run_id={short_run}  dir={out_dir}")
    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete the matching rows (default is dry-run).",
    )
    parser.add_argument(
        "--yes-i-counted", type=int, default=None, metavar="N",
        help="Required with --apply: pass the EXACT orphan count you "
             "saw in the prior dry-run. Mismatched count aborts.",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Override DB path. Default: face_cluster._paths.default_db_path().",
    )
    args = parser.parse_args(argv)

    db_path = args.db or default_db_path()
    if not db_path.exists():
        print(f"DB not found at {db_path}", file=sys.stderr)
        return 2

    with sqlite3.connect(str(db_path)) as conn:
        orphans = _select_orphans(conn)
        print(f"\nDB: {db_path}")
        print(f"Orphan rows found: {len(orphans)}\n")
        _print_rows(orphans)

        if not args.apply:
            print(f"\nDry-run. To delete, re-run with:")
            print(f"  --apply --yes-i-counted {len(orphans)}")
            return 0

        if args.yes_i_counted is None:
            print("\n--apply requires --yes-i-counted N (the count you saw above).",
                  file=sys.stderr)
            return 3
        if args.yes_i_counted != len(orphans):
            print(f"\n--yes-i-counted {args.yes_i_counted} does not match actual "
                  f"orphan count {len(orphans)}. Aborting.", file=sys.stderr)
            return 3

        ids = [r[0] for r in orphans]
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM action_log WHERE id IN ({placeholders})", ids)
        conn.commit()
        print(f"\nDeleted {len(ids)} row(s).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
