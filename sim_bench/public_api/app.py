"""The public API (spec-120): what the website talks to. A separate FastAPI app from the owner's API
(``sim_bench.api.main``), which is never exposed to the internet.

Every route except ``/health`` resolves the signed-in account first (R1); routes are added step by step (W1: account).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from sim_bench.control.db import ControlDB
from sim_bench.control.models import Account
from sim_bench.public_api.accounts import AccountRepo, NotInvited
from sim_bench.public_api.auth import AuthError, AuthVerifier, bearer_token
from sim_bench.public_api.schemas import AccountOut

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def create_app(verifier: Optional[AuthVerifier] = None, accounts: Optional[AccountRepo] = None) -> FastAPI:
    """Config in, app out. Without arguments: issuer from ``CLERK_ISSUER``, database from ``DATABASE_URL``."""
    app = FastAPI(title="Albumify public API", version="1")
    app.state.verifier = verifier or AuthVerifier(os.environ["CLERK_ISSUER"])
    app.state.accounts = accounts or AccountRepo(ControlDB())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/me", response_model=AccountOut)
    def me(account: Account = Depends(current_account)):
        return AccountOut.of(account)

    @app.post("/v1/me/consent", response_model=AccountOut)
    def consent(request: Request, account: Account = Depends(current_account)):
        """Record agreement to face grouping (R3): no upload is accepted before it."""
        return AccountOut.of(request.app.state.accounts.record_consent(account.id))

    return app


def current_account(request: Request, authorization: Optional[str] = Header(default=None)) -> Account:
    """The signed-in account: 401 without a valid token, 403 for an email not on the invite list."""
    try:
        claims = request.app.state.verifier.verify(bearer_token(authorization))
    except AuthError as e:
        raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"})
    try:
        return request.app.state.accounts.get_or_create(claims)
    except NotInvited:
        raise HTTPException(403, "on the waiting list: signup is invite-only for now")
