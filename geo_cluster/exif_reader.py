"""EXIF geo/time extraction (spec-022, Slice 1).

Reads ``DateTimeOriginal`` and GPS lat/lon per image. Missing, malformed, or
out-of-range fields degrade to ``None`` -- the reader never raises on a bad
photo (FR-011: the pipeline must not fail when EXIF is absent).

Primary entry: ``GeoMetadataExtractor.calc(ExifInputs) -> GeoMetadataResult``.
``read_one()`` and the module-level parse helpers stay public for notebooks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS
from pillow_heif import register_heif_opener

from geo_cluster.types import GeoMetadata

# HEIC/HEIF support for PIL (idempotent; image_cache also registers it).
register_heif_opener()

logger = logging.getLogger(__name__)

# EXIF tag IDs as returned by PIL's flat ``Image._getexif()`` dict.
_DATETIME_ORIGINAL = 0x9003  # 36867
_GPS_INFO = 0x8825           # 34853
_DATE_FMT = "%Y:%m:%d %H:%M:%S"

MIN_YEAR = 1990              # reject obviously-wrong capture dates (spec edge cases)
_MAX_FUTURE = timedelta(days=1)


@dataclass
class ExifInputs:
    """Per-call data for the extractor (NOT PipelineContext)."""

    image_paths: list[str]


@dataclass
class GeoMetadataResult:
    metadata: dict[str, GeoMetadata] = field(default_factory=dict)


def _dms_to_degrees(dms, ref) -> Optional[float]:
    """Convert an EXIF (deg, min, sec) triple + N/S/E/W ref to decimal degrees."""
    try:
        deg, minute, sec = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    value = deg + minute / 60.0 + sec / 3600.0
    if ref in ("S", "W"):
        value = -value
    return value


def parse_datetime(raw, *, min_year: int, now: datetime) -> Optional[datetime]:
    """Parse an EXIF datetime string, rejecting pre-``min_year`` / future dates."""
    if not raw:
        return None
    try:
        dt = datetime.strptime(str(raw).strip(), _DATE_FMT)
    except (ValueError, TypeError):
        return None
    if dt.year < min_year or dt > now + _MAX_FUTURE:
        return None
    return dt


def extract_from_exif(exif: Optional[dict], *, min_year: int, now: datetime):
    """Pure parse of a PIL ``_getexif()`` dict -> ``(timestamp, lat, lon)``.

    Separated from file IO so the parsing rules are unit-testable without
    writing image fixtures.
    """
    if not exif:
        return None, None, None

    timestamp = parse_datetime(exif.get(_DATETIME_ORIGINAL), min_year=min_year, now=now)

    lat = lon = None
    gps = exif.get(_GPS_INFO)
    if isinstance(gps, dict):
        g = {GPSTAGS.get(k, k): v for k, v in gps.items()}
        if "GPSLatitude" in g and "GPSLongitude" in g:
            lat = _dms_to_degrees(g["GPSLatitude"], g.get("GPSLatitudeRef"))
            lon = _dms_to_degrees(g["GPSLongitude"], g.get("GPSLongitudeRef"))

    return timestamp, lat, lon


class GeoMetadataExtractor:
    """Reads EXIF geo/time per image. Config in ``__init__``; data via ``calc()``."""

    def __init__(self, min_year: int = MIN_YEAR, now: Optional[datetime] = None):
        self._min_year = min_year
        self._now = now  # injectable for deterministic date-bound tests

    def read_one(self, image_path: str) -> GeoMetadata:
        now = self._now or datetime.now()
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
        except Exception as e:  # corrupt/unreadable -> empty metadata, never raise
            logger.debug("EXIF read failed for %s: %s", image_path, e)
            exif = None
        ts, lat, lon = extract_from_exif(exif, min_year=self._min_year, now=now)
        return GeoMetadata(image_path=image_path, timestamp=ts, lat=lat, lon=lon)

    def calc(self, inputs: ExifInputs) -> GeoMetadataResult:
        result = GeoMetadataResult()
        for path in inputs.image_paths:
            result.metadata[path] = self.read_one(path)
        return result
