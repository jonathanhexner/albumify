"""Public-site limits (spec-120 R5-R8), all checked before anything is stored or computed.

    Limits       values from ``configs/public.yaml``: one upload's size, the monthly quota, the spend cap
    QuotaGuard   per account per calendar month (UTC): albums and photos, over it -> ``QuotaExceeded`` (429)
    SpendGuard   all public accounts per calendar month: estimated dollars of completed uploads against the cap;
                 an upload that would pass it -> ``SpendCapReached``; a warning is logged when 80 % is passed

Quota and spend count **completed** uploads (``uploads.completed_at`` this month): an upload that was started and
never finished costs nothing. ``create`` checks with the new upload added, ``complete`` checks again, so the
answer at the start is the answer at the end unless something else finished in between.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import yaml
from sqlalchemy import func, select

from sim_bench.control.db import ControlDB
from sim_bench.control.models import Upload

logger = logging.getLogger(__name__)

__all__ = ["Limits", "load_limits", "QuotaGuard", "SpendGuard", "QuotaExceeded", "SpendCapReached",
           "CONTENT_TYPES", "month_start", "next_month"]

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "public.yaml"

# accepted types: file extension -> the one content type the signed URL is made for
CONTENT_TYPES = {
    "jpeg": {".jpg": "image/jpeg", ".jpeg": "image/jpeg"},
    "heic": {".heic": "image/heic", ".heif": "image/heif"},
    "png": {".png": "image/png"},
    "webp": {".webp": "image/webp"},
}


@dataclass(frozen=True, slots=True)
class Limits:
    max_photos: int = 1000
    max_file_mb: float = 25
    max_total_gb: float = 6
    types: Tuple[str, ...] = ("jpeg", "heic", "png", "webp")
    albums_per_month: int = 3
    photos_per_month: int = 3000
    spend_cap_dollars: float = 50.0
    alert_fraction: float = 0.8
    gpu_min_photos: int = 300
    cpu_fixed_seconds: float = 120.0
    cpu_seconds_per_photo: float = 3.0
    cpu_dollars_per_hour: float = 0.20
    assumed_upload_mbps: float = 20.0
    upload_url_minutes: float = 180
    retention_days: int = 7
    max_running: int = 2

    def __post_init__(self):
        object.__setattr__(self, "types", tuple(self.types))
        unknown = set(self.types) - set(CONTENT_TYPES)
        if unknown:
            raise ValueError(f"types: unknown {sorted(unknown)}; known {sorted(CONTENT_TYPES)}")

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)

    @property
    def max_total_bytes(self) -> int:
        return int(self.max_total_gb * 1024 ** 3)

    def content_type(self, filename: str) -> Optional[str]:
        """The content type an accepted file is uploaded as, from its extension; None = not accepted."""
        ext = Path(filename).suffix.lower()
        for t in self.types:
            if ext in CONTENT_TYPES[t]:
                return CONTENT_TYPES[t][ext]
        return None


def load_limits(path: Optional[Path] = None) -> Limits:
    path = Path(path) if path else _DEFAULT_PATH
    raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
    unknown = set(raw) - {f.name for f in fields(Limits)}
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    return Limits(**raw)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def month_start(now: datetime) -> datetime:
    now = now.astimezone(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc)


def next_month(now: datetime) -> date:
    """The day the monthly quota and spend start again."""
    now = now.astimezone(timezone.utc)
    return date(now.year + 1, 1, 1) if now.month == 12 else date(now.year, now.month + 1, 1)


class QuotaExceeded(Exception):
    """Over the account's monthly albums or photos (-> 429 with ``resets_on``)."""

    def __init__(self, message: str, resets_on: date):
        super().__init__(message)
        self.resets_on = resets_on


class SpendCapReached(Exception):
    """This upload would take the month's public GPU/CPU spend past the cap (-> 503 with ``resets_on``)."""

    def __init__(self, message: str, resets_on: date):
        super().__init__(message)
        self.resets_on = resets_on


class QuotaGuard:
    def __init__(self, db: ControlDB, limits: Limits, *, clock: Callable[[], datetime] = _now):
        self.db, self.limits, self._clock = db, limits, clock

    def used_this_month(self, account_id: str, *, exclude: Optional[str] = None) -> Tuple[int, int]:
        """(albums, photos) the account completed this calendar month; ``exclude`` = an upload id left out."""
        q = select(func.count(Upload.id), func.coalesce(func.sum(Upload.n_photos), 0)).where(
            Upload.account_id == account_id, Upload.completed_at >= month_start(self._clock()))
        if exclude is not None:
            q = q.where(Upload.id != exclude)
        with self.db.read_session() as s:
            albums, photos = s.execute(q).one()
        return int(albums), int(photos)

    def check(self, account_id: str, n_photos: int, *, exclude: Optional[str] = None) -> None:
        """Raise ``QuotaExceeded`` if one more album of ``n_photos`` passes this month's quota."""
        albums, photos = self.used_this_month(account_id, exclude=exclude)
        resets = next_month(self._clock())
        if albums + 1 > self.limits.albums_per_month:
            raise QuotaExceeded(f"you have made {albums} albums this month, the limit is "
                                f"{self.limits.albums_per_month}; it resets on {resets.isoformat()}", resets)
        if photos + n_photos > self.limits.photos_per_month:
            left = max(0, self.limits.photos_per_month - photos)
            raise QuotaExceeded(f"{n_photos} photos is more than the {left} left this month "
                                f"(limit {self.limits.photos_per_month}); it resets on {resets.isoformat()}", resets)


class SpendGuard:
    def __init__(self, db: ControlDB, limits: Limits, *, clock: Callable[[], datetime] = _now):
        self.db, self.limits, self._clock = db, limits, clock

    def spent_this_month(self, *, exclude: Optional[str] = None) -> float:
        """Estimated dollars of every upload completed this calendar month, all accounts."""
        q = select(func.coalesce(func.sum(Upload.estimated_dollars), 0.0)).where(
            Upload.completed_at >= month_start(self._clock()))
        if exclude is not None:
            q = q.where(Upload.id != exclude)
        with self.db.read_session() as s:
            return float(s.scalar(q))

    def allows(self, dollars: float, *, exclude: Optional[str] = None, alert: bool = True) -> bool:
        """True if ``dollars`` more stays within the cap. Logs the 80 % alert when this passes it (``alert``
        False for a look ahead, e.g. when an upload is only being created)."""
        cap = self.limits.spend_cap_dollars
        spent = self.spent_this_month(exclude=exclude)
        after = spent + dollars
        if after > cap:
            logger.warning("spend cap: $%.2f spent this month, $%.2f more would pass the $%.2f cap",
                           spent, dollars, cap)
            return False
        if alert and spent < self.limits.alert_fraction * cap <= after:
            logger.warning("spend alert: this month's public spend reaches $%.2f of the $%.2f cap (%.0f %%)",
                           after, cap, 100 * after / cap)
        return True

    def check(self, dollars: float, *, exclude: Optional[str] = None, alert: bool = True) -> None:
        if not self.allows(dollars, exclude=exclude, alert=alert):
            resets = next_month(self._clock())
            raise SpendCapReached("Albumify has reached its processing limit for this month; "
                                  f"try again on {resets.isoformat()}", resets)
