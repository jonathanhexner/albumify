"""What the website receives from the public API (spec-120). Grows step by step (W1: the account)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from sim_bench.control.models import Account

__all__ = ["AccountOut"]


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
