"""What the website receives from the public API (spec-120). Grows step by step (W1: the account; W3: uploads)."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from sim_bench.control.models import Account

__all__ = ["AccountOut", "FileSpec", "UploadIn", "SignedFileOut", "EstimateOut", "UploadTicketOut",
           "UploadCompleteOut"]


class AccountOut(BaseModel):
    id: str
    email: str
    face_consent: bool
    face_consent_at: Optional[datetime] = None
    created_at: datetime

    @classmethod
    def of(cls, account: Account) -> "AccountOut":
        return cls(id=account.id, email=account.email, face_consent=account.face_consent_at is not None,
                   face_consent_at=account.face_consent_at, created_at=account.created_at)


class FileSpec(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    size: int = Field(ge=0)                                   # bytes, as the browser's File.size reports it
    content_type: Optional[str] = Field(default=None, max_length=100)


class UploadIn(BaseModel):
    files: List[FileSpec]


class SignedFileOut(BaseModel):
    name: str
    url: str
    content_type: str          # send exactly this Content-Type header with the PUT, or R2 refuses it


class EstimateOut(BaseModel):
    runner: str
    upload_minutes: float
    processing_minutes: float


class UploadTicketOut(BaseModel):
    upload_id: str
    n_photos: int
    total_bytes: int
    estimate: EstimateOut
    expires_at: datetime
    files: List[SignedFileOut]

    @classmethod
    def of(cls, t) -> "UploadTicketOut":
        e = t.estimate
        return cls(upload_id=t.upload_id, n_photos=t.n_photos, total_bytes=t.total_bytes,
                   estimate=EstimateOut(runner=e.runner, upload_minutes=e.upload_minutes,
                                        processing_minutes=e.processing_minutes),
                   expires_at=t.expires_at,
                   files=[SignedFileOut(name=f.name, url=f.url, content_type=f.content_type) for f in t.files])


class UploadCompleteOut(BaseModel):
    job_id: str
    status: str
