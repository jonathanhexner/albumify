"""One-shot migration: import existing results/ pipeline runs into action_log.

Usage:
    .venv/Scripts/python scripts/migrate_runs_to_db.py [results_root]

    results_root defaults to "results" relative to the project root.

The script is idempotent — rows already present (matched by run_id + output_dir)
are skipped.  Run it once after upgrading to the DB-backed History tab.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running without package install
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster.run_history_db import upsert_run, get_db_path


def _duration(started: str, finished: str) -> float | None:
    if not started or not finished:
        return None
    from datetime import datetime, timezone
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in started else "%Y-%m-%dT%H:%M:%S"
    try:
        s = datetime.fromisoformat(started)
        f = datetime.fromisoformat(finished)
        return (f - s).total_seconds()
    except Exception:
        return None


def migrate(results_root: str = "results") -> None:
    root = Path(results_root)
    if not root.exists():
        print(f"Results root not found: {root.resolve()}")
        return

    db_path = get_db_path()
    print(f"DB: {db_path}")

    imported = skipped = failed = 0

    for run_dir in sorted(root.iterdir()):
        summary_path = run_dir / "pipeline_run.json"
        if not summary_path.exists():
            continue

        rec = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id    = rec.get("run_id", run_dir.name)
        output_dir = str(run_dir.resolve())
        mode      = rec.get("mode", "pipeline_run")

        summary  = rec.get("summary", {})
        started  = rec.get("started_at", "")
        finished = rec.get("finished_at", "")

        # Derive log_file from summary or scan logs/
        log_file = summary.get("log_file")
        if not log_file:
            logs_dir = run_dir / "logs"
            if logs_dir.exists():
                candidates = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                log_file = str(candidates[0]) if candidates else None

        fields = {
            "status":     rec.get("status", "complete"),
            "started_at": started,
            "ended_at":   finished or None,
            "duration_s": _duration(started, finished),
            "error":      rec.get("error"),
            "source_dir": rec.get("source_album") or rec.get("source_run"),
            "album":      Path(rec.get("source_album", "")).name or None,
            "n_faces":    summary.get("n_faces"),
            "n_clusters": summary.get("n_clusters"),
            "n_noise":    summary.get("n_noise"),
            "log_file":   log_file,
            "payload_json_extra": {
                "config":          rec.get("config", {}),
                "stages_timing":   {k: v.get("elapsed_s") for k, v in rec.get("stages", {}).items()},
                "migrated_from_fs": True,
            },
        }

        try:
            upsert_run(run_id, output_dir, mode, fields)
            imported += 1
            print(f"  imported  {run_id}  ({mode})  status={fields['status']}")
        except Exception as exc:
            print(f"  FAILED    {run_id}: {exc}")
            failed += 1

    print(f"\nDone: {imported} imported, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    migrate(root)
