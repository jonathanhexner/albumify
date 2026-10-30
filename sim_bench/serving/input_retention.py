"""Which staged photos may be deleted from a worker's input store (spec-115 W3, R13).

Works on any input-store root (``<sha16>/<name>.jpg``, see :mod:`input_store`): a local folder in tests, the
mounted ``albumify-inputs`` Volume on Modal. Each photo folder carries ``.used.json`` = ``{job_id: last-used
epoch seconds}``. A photo is kept while any job used it within ``days``; otherwise it is deleted.

    mark_used(root, job_id, keys, now)       a job is about to use these photos
    release(root, job_id, keys, now, days)   owner's "delete my photos": drop this job's claim, delete what no
                                              other job used in the window
    purge(root, now, days)                   daily: delete photos no job used in the window, and legacy
                                              per-job input copies (``<jobs_root>/<id>/input/*.jpg``) older than it
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["RETENTION_DAYS", "RetentionResult", "mark_used", "release", "purge"]

RETENTION_DAYS = 7          # owner decision O3, 2026-10-23
_USED = ".used.json"
_DAY = 86400.0


@dataclass
class RetentionResult:
    deleted: List[str] = field(default_factory=list)       # photo folders (sha16) or legacy input dirs removed
    kept: int = 0


def _record(folder: Path) -> Dict[str, float]:
    """The photo's stored claims ({} when there is no record or it is damaged)."""
    try:
        return {str(k): float(v) for k, v in json.loads((folder / _USED).read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def _claims(folder: Path) -> Dict[str, float]:
    """For purge decisions: no record, or a damaged one, falls back to the folder's own time, so a photo is never
    deleted just because its record is missing (uploaded moments ago, or a write went wrong; review M2)."""
    return _record(folder) or {"_unrecorded": folder.stat().st_mtime}


def _write(folder: Path, record: Dict[str, float]) -> None:
    tmp = folder / (_USED + ".part")
    tmp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    os.replace(tmp, folder / _USED)       # a reader never sees half a record


def _folders(root: Path, keys: Iterable[str]) -> List[Path]:
    return sorted({root / k.split("/", 1)[0] for k in keys})


def mark_used(root: Path, job_id: str, keys: Iterable[str], now: float) -> int:
    n = 0
    for folder in _folders(Path(root), keys):
        if folder.is_dir():
            rec = _record(folder)
            rec[job_id] = now
            _write(folder, rec)
            n += 1
    return n


def _fresh(rec: Dict[str, float], now: float, days: float, ignore: Optional[str] = None) -> bool:
    return any(now - t < days * _DAY for j, t in rec.items() if j != ignore)


def release(root: Path, job_id: str, keys: Iterable[str], now: float,
            days: float = RETENTION_DAYS) -> RetentionResult:
    res = RetentionResult()
    for folder in _folders(Path(root), keys):
        if not folder.is_dir():
            continue
        rec = _record(folder)
        rec.pop(job_id, None)
        rec.pop("uploaded", None)              # the upload claim belongs to the job being released
        if _fresh(rec, now, days):
            _write(folder, rec)
            res.kept += 1
        else:
            shutil.rmtree(folder)
            res.deleted.append(folder.name)
    return res


def purge(root: Path, now: float, days: float = RETENTION_DAYS,
          legacy_jobs_root: Optional[Path] = None) -> RetentionResult:
    res = RetentionResult()
    root = Path(root)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        if _fresh(_claims(folder), now, days):
            res.kept += 1
        else:
            shutil.rmtree(folder)
            res.deleted.append(folder.name)
    if legacy_jobs_root is not None and Path(legacy_jobs_root).is_dir():
        for input_dir in sorted(Path(legacy_jobs_root).glob("*/input")):
            photos = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
            if photos and all(now - p.stat().st_mtime >= days * _DAY for p in photos):
                for p in photos:
                    p.unlink()
                res.deleted.append(f"{input_dir.parent.name}/input")
    logger.info("input retention: deleted %d, kept %d", len(res.deleted), res.kept)
    return res
