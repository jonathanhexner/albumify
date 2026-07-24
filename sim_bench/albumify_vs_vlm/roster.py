"""Spec-102 T1.2 — coverage-roster template for a trip.

The coverage precision/recall metric (A6) needs a ground-truth answer to "who and what places
should the album cover?". Persons and distinct scenes are a human judgement, but **day
segmentation is free** from the `YYYYMMDD_HHMMSS` filenames (EXIF `DateTimeOriginal` is the
fallback). So this module pre-fills the days and emits a template the owner hand-completes:
fill each person's `images` list and label each distinct scene/place.

The template records `input_set_hash` so a roster can be tied to the exact working set it was
labelled against (a re-downsample that changes the set invalidates stale labels).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

from sim_bench.albumify_vs_vlm.downsample import DownsampleResult, ImageRecord

logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"(\d{8})_(\d{6})")
_EXIF_DT_TAG = next((k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal"), 36867)


def _ts_from_stem(stem: str) -> datetime | None:
    m = _TS_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _ts_from_exif(path: str) -> datetime | None:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
        raw = exif.get(_EXIF_DT_TAG)
        if raw:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def timestamp_for(record: ImageRecord) -> datetime | None:
    """Filename timestamp first (cheap, reliable on these exports), EXIF as fallback."""
    return _ts_from_stem(record.stem) or _ts_from_exif(record.src)


def build_roster_template(result: DownsampleResult) -> dict:
    """Group images into days by timestamp; leave persons/scenes for the human to fill."""
    dated: list[tuple[datetime | None, str]] = [
        (timestamp_for(r), r.stem) for r in result.records
    ]
    by_day: dict[str, list[str]] = {}
    undated: list[str] = []
    for ts, stem in dated:
        if ts is None:
            undated.append(stem)
        else:
            by_day.setdefault(ts.strftime("%Y-%m-%d"), []).append(stem)

    days = [
        {"date": day, "label": "", "images": sorted(stems)}
        for day, stems in sorted(by_day.items())
    ]
    if undated:
        days.append({"date": "unknown", "label": "", "images": sorted(undated)})

    return {
        "trip": result.trip,
        "input_set_hash": result.input_set_hash,
        "n_images": result.n_images,
        "_instructions": (
            "Days are auto-filled from timestamps. FILL: (1) a short label per day/scene, "
            "(2) each person's `images` list (stems they appear in), (3) split days into "
            "distinct `scenes` if a day spans multiple places. Coverage P/R (A6) is scored "
            "against `persons` and `scenes`."
        ),
        "days": days,
        "persons": [{"id": "P1", "name": "", "images": []}],
        "scenes": [{"id": "S1", "label": "", "day": "", "images": []}],
    }


def save_roster(roster: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(roster, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("wrote roster template -> %s", path)


def load_roster(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_roster(roster: dict, expected_hash: str | None = None) -> list[str]:
    """Cheap sanity checks before the roster is trusted as ground truth."""
    errors: list[str] = []
    if expected_hash and roster.get("input_set_hash") != expected_hash:
        errors.append(
            "input_set_hash mismatch: roster was labelled against a different working set"
        )
    if not any(p.get("images") for p in roster.get("persons", [])):
        errors.append("no person has any images labelled (coverage recall cannot be scored)")
    if not any(s.get("label") for s in roster.get("scenes", [])):
        errors.append("no scene labelled (scene coverage cannot be scored)")
    return errors
