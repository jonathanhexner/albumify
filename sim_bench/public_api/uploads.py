"""Uploads from the public site (spec-120 R5, R6): the browser sends photos straight to R2, never through the API.

    create(account, files)   checks consent, size, count, type, quota, spend  -> UploadTicket: one signed PUT URL
                             per file (size and type fixed in the signature), the expected wait, where it runs
    browser                  PUT <url> for each file
    complete(account, id)    every file there at its declared size -> job ``id`` queued, owned by the account

Nothing is stored and no job exists until ``complete``; an upload that is never completed costs no quota and its
files are swept with the originals (W6). Object keys are ``<id>/<nnnn><ext>``: the user's file names are kept in the
upload row, never used as keys.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import JobRepo
from sim_bench.control.models import Account, Upload
from sim_bench.public_api.accounts import AccountRepo
from sim_bench.public_api.limits import Limits, QuotaGuard, SpendGuard, load_limits
from sim_bench.serving.serving_config import ServingConfig, load_serving_config
from sim_bench.storage.object_store import ObjectStore

logger = logging.getLogger(__name__)

__all__ = ["UploadService", "FileIn", "UploadTicket", "SignedFile", "UploadEstimate",
           "UploadRefused", "UploadIncomplete"]


class UploadRefused(Exception):
    """The upload breaks a limit; ``status`` is the HTTP answer (400 count/total, 413 file size, 415 type)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class UploadIncomplete(Exception):
    """``complete`` before every file arrived (-> 409); ``missing`` names the files still to send."""

    def __init__(self, missing: List[str]):
        shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        super().__init__(f"{len(missing)} files have not arrived yet: {shown}")
        self.missing = missing


@dataclass(frozen=True, slots=True)
class FileIn:
    name: str
    size: int
    content_type: Optional[str] = None     # as the browser reports it; checked against the extension


@dataclass(frozen=True, slots=True)
class UploadEstimate:
    runner: str                            # modal_gpu for >= gpu_min_photos, else modal_cpu
    upload_minutes: float                  # at the assumed home upload speed
    processing_minutes: float
    dollars: float                         # what the run is counted at against the spend cap


@dataclass(frozen=True, slots=True)
class SignedFile:
    name: str
    url: str
    content_type: str                      # the browser must send exactly this Content-Type


@dataclass(frozen=True, slots=True)
class UploadTicket:
    upload_id: str
    n_photos: int
    total_bytes: int
    estimate: UploadEstimate
    expires_at: datetime
    files: List[SignedFile]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_upload_id() -> str:
    return f"up-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


class UploadService:
    def __init__(self, db: ControlDB, store: ObjectStore, limits: Optional[Limits] = None,
                 serving: Optional[ServingConfig] = None, *, clock: Callable[[], datetime] = _now,
                 new_id: Callable[[], str] = new_upload_id):
        """``store`` holds the uploads (R2 ``uploads/``: ``R2ObjectStore.from_env().under("uploads/")``)."""
        self.db, self.store = db, store
        self.limits = limits or load_limits()
        self.serving = serving or load_serving_config()
        self.quota = QuotaGuard(db, self.limits, clock=clock)
        self.spend = SpendGuard(db, self.limits, clock=clock)
        self.jobs = JobRepo(db)
        self._clock, self._new_id = clock, new_id

    # -- R5: limits ---------------------------------------------------------------------------------------------

    def _checked(self, files: Sequence[FileIn]) -> List[Dict]:
        """The files with their object key and signed content type, or ``UploadRefused``."""
        lim = self.limits
        if not files:
            raise UploadRefused("no photos in the upload")
        if len(files) > lim.max_photos:
            raise UploadRefused(f"{len(files)} photos is over the limit of {lim.max_photos} per upload")
        out, total = [], 0
        for i, f in enumerate(files):
            ctype = lim.content_type(f.name)
            if ctype is None:
                raise UploadRefused(f"{f.name}: only {', '.join(t.upper() for t in lim.types)} photos are accepted",
                                    415)
            if f.content_type and f.content_type.lower() not in (ctype, "application/octet-stream"):
                raise UploadRefused(f"{f.name}: the file says it is {f.content_type}, not a {ctype} photo", 415)
            if f.size <= 0:
                raise UploadRefused(f"{f.name} is empty")
            if f.size > lim.max_file_bytes:
                raise UploadRefused(f"{f.name} is {f.size / 1024 ** 2:.1f} MB, over the limit of "
                                    f"{lim.max_file_mb:g} MB per photo", 413)
            total += f.size
            out.append({"name": f.name, "key": f"{i:04d}{Path(f.name).suffix.lower()}", "size": int(f.size),
                        "content_type": ctype})
        if total > lim.max_total_bytes:
            raise UploadRefused(f"the photos add up to {total / 1024 ** 3:.2f} GB, over the limit of "
                                f"{lim.max_total_gb:g} GB per upload", 413)
        return out

    # -- R6: estimate -------------------------------------------------------------------------------------------

    def estimate(self, n_photos: int, total_bytes: int) -> UploadEstimate:
        lim, cfg = self.limits, self.serving
        upload_min = total_bytes * 8 / (lim.assumed_upload_mbps * 1e6) / 60
        if n_photos >= lim.gpu_min_photos:
            # the same figures the owner's L4 estimate uses: slowest GPU in the list, dearest price (spec-117 M6)
            minutes = (cfg.l4_fixed_seconds + n_photos * cfg.l4_seconds_per_photo) * cfg.slowest_slowdown / 60
            runner, per_hour = "modal_gpu", cfg.dearest_dollars_per_hour
        else:
            minutes = (lim.cpu_fixed_seconds + n_photos * lim.cpu_seconds_per_photo) / 60
            runner, per_hour = "modal_cpu", lim.cpu_dollars_per_hour
        return UploadEstimate(runner=runner, upload_minutes=round(upload_min, 1),
                              processing_minutes=round(minutes, 1), dollars=round(minutes / 60 * per_hour, 2))

    # -- create / complete --------------------------------------------------------------------------------------

    def create(self, account: Account, files: Sequence[FileIn]) -> UploadTicket:
        AccountRepo.require_consent(account)                       # R3
        checked = self._checked(files)                             # R5
        total = sum(f["size"] for f in checked)
        est = self.estimate(len(checked), total)                   # R6
        self.quota.check(account.id, len(checked))                 # R7
        self.spend.check(est.dollars, alert=False)                 # R8 (look ahead; counted at complete)
        upload_id, now = self._new_id(), self._clock()
        with self.db.session() as s:
            s.add(Upload(id=upload_id, account_id=account.id, files=checked, n_photos=len(checked),
                         total_bytes=total, runner=est.runner, estimated_dollars=est.dollars, created_at=now))
        expires_s = int(self.limits.upload_url_minutes * 60)
        signed = [SignedFile(name=f["name"], content_type=f["content_type"],
                             url=self.store.presigned_put(f"{upload_id}/{f['key']}", f["size"], f["content_type"],
                                                          expires_s))
                  for f in checked]
        logger.info("upload %s created: %d photos, %.1f MB, %s", upload_id, len(checked), total / 1024 ** 2,
                    est.runner)
        return UploadTicket(upload_id=upload_id, n_photos=len(checked), total_bytes=total, estimate=est,
                            expires_at=now + timedelta(seconds=expires_s), files=signed)

    def complete(self, account: Account, upload_id: str) -> str:
        """Every file arrived at its declared size -> the job ``upload_id`` is queued; returns its id.
        Calling it again after success returns the same id. Another account's upload is ``KeyError`` (404)."""
        with self.db.read_session() as s:
            upload = s.get(Upload, upload_id)
        if upload is None or upload.account_id != account.id:
            raise KeyError(upload_id)
        if upload.completed_at is not None:
            return upload.id
        arrived = self.store.sizes(f"{upload_id}/")
        missing = [f["name"] for f in upload.files if f"{upload_id}/{f['key']}" not in arrived]
        if missing:
            raise UploadIncomplete(missing)
        wrong = [f["name"] for f in upload.files if arrived[f"{upload_id}/{f['key']}"] != f["size"]]
        if wrong:          # the signed URL fixes the size, so this is never a browser mistake: drop the files
            self.store.delete_prefix(f"{upload_id}/")
            raise UploadRefused(f"{len(wrong)} files arrived at another size than declared ({wrong[0]}); "
                                "start the upload again")
        # checked again: another upload may have finished since this one was created
        self.quota.check(account.id, upload.n_photos, exclude=upload.id)
        self.spend.check(upload.estimated_dollars, exclude=upload.id)
        if self.jobs.get(upload.id) is None:      # a retry after the job was made but before completed_at
            self.jobs.create(upload.id, runner=upload.runner, source=f"upload:{upload.id}",
                             owner_id=account.id, n_inputs=upload.n_photos)
        with self.db.session() as s:
            s.get(Upload, upload.id).completed_at = self._clock()
        logger.info("upload %s complete: job queued (%d photos, %s)", upload.id, upload.n_photos, upload.runner)
        return upload.id
