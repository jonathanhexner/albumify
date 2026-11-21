"""Public accounts (spec-120 R2, R3): created on first sign-in, only for invited emails; face-grouping consent is
recorded before any upload. ``delete_everything`` comes with the album endpoints (W5)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select

from sim_bench.control.db import ControlDB
from sim_bench.control.models import Account, Invite
from sim_bench.public_api.auth import Claims

logger = logging.getLogger(__name__)

__all__ = ["AccountRepo", "ConsentRequired", "NotInvited"]


class NotInvited(Exception):
    """Signup is invite-only and this email is not on the list (-> 403, "on the waiting list")."""


class ConsentRequired(Exception):
    """The account has not agreed to face grouping yet (-> 403); nothing is uploaded before it does."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AccountRepo:
    def __init__(self, db: ControlDB, *, clock: Callable[[], datetime] = _now):
        self.db, self._clock = db, clock

    def invite(self, email: str) -> None:
        """Put an email on the invite list (owner action; idempotent)."""
        email = email.strip().lower()
        with self.db.session() as s:
            if s.get(Invite, email) is None:
                s.add(Invite(email=email, invited_at=self._clock()))

    def get_or_create(self, claims: Claims) -> Account:
        """The account of a signed-in user; created on first sign-in if the email is invited."""
        with self.db.session() as s:
            account = s.get(Account, claims.subject)
            if account is not None:
                return account
            if s.get(Invite, claims.email) is None:
                raise NotInvited(claims.email)
            if s.scalar(select(Account).where(Account.email == claims.email)) is not None:
                raise NotInvited(claims.email)       # the email belongs to another sign-in: never merge silently
            account = Account(id=claims.subject, email=claims.email, created_at=self._clock())
            s.add(account)
            logger.info("account created for an invited email")
            return account

    def record_consent(self, account_id: str) -> Account:
        with self.db.session() as s:
            account = s.get(Account, account_id)
            if account is None:
                raise KeyError(account_id)
            if account.face_consent_at is None:
                account.face_consent_at = self._clock()
            return account

    @staticmethod
    def require_consent(account: Account) -> None:
        if account.face_consent_at is None:
            raise ConsentRequired(account.id)
