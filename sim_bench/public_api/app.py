"""The public API (spec-120): what the website talks to. A separate FastAPI app from the owner's API
(``sim_bench.api.main``), which is never exposed to the internet.

Every route except ``/health`` resolves the signed-in account first (R1); routes are added step by step (W1: account;
W3: uploads).
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import QUEUED
from sim_bench.control.models import Account
from sim_bench.public_api.accounts import AccountRepo, ConsentRequired, NotInvited
from sim_bench.public_api.auth import AuthError, AuthVerifier, bearer_token
from sim_bench.public_api.limits import QuotaExceeded, SpendCapReached
from sim_bench.public_api.schemas import AccountOut, UploadCompleteOut, UploadIn, UploadTicketOut
from sim_bench.public_api.uploads import FileIn, UploadIncomplete, UploadRefused, UploadService

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def create_app(verifier: Optional[AuthVerifier] = None, accounts: Optional[AccountRepo] = None,
               uploads: Optional[UploadService] = None) -> FastAPI:
    """Config in, app out. Without arguments: issuer from ``CLERK_ISSUER``, database from ``DATABASE_URL``,
    uploads in the R2 bucket of ``R2_*`` (made on the first upload, so the app starts without R2)."""
    app = FastAPI(title="Albumify public API", version="1")
    app.state.verifier = verifier or AuthVerifier(os.environ["CLERK_ISSUER"])
    app.state.accounts = accounts or AccountRepo(ControlDB())
    app.state.uploads = uploads

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

    @app.post("/v1/uploads", response_model=UploadTicketOut, status_code=201)
    def create_upload(body: UploadIn, request: Request, account: Account = Depends(current_account)):
        """Limits checked, then one signed PUT URL per file (R5), the expected wait and where it runs (R6)."""
        files = [FileIn(f.name, f.size, f.content_type) for f in body.files]
        with _answers():
            return UploadTicketOut.of(upload_service(request).create(account, files))

    @app.post("/v1/uploads/{upload_id}/complete", response_model=UploadCompleteOut)
    def complete_upload(upload_id: str, request: Request, account: Account = Depends(current_account)):
        """Every file arrived -> the album's job is queued."""
        with _answers():
            return UploadCompleteOut(job_id=upload_service(request).complete(account, upload_id), status=QUEUED)

    return app


def upload_service(request: Request) -> UploadService:
    if request.app.state.uploads is None:
        from sim_bench.storage.object_store import R2ObjectStore
        request.app.state.uploads = UploadService(request.app.state.accounts.db,
                                                  R2ObjectStore.from_env().under("uploads/"))
    return request.app.state.uploads


@contextmanager
def _answers() -> Iterator[None]:
    """The service's refusals as HTTP answers with plain messages."""
    try:
        yield
    except ConsentRequired:
        raise HTTPException(403, "agree to face grouping first (POST /v1/me/consent)")
    except UploadRefused as e:
        raise HTTPException(e.status, str(e))
    except UploadIncomplete as e:
        raise HTTPException(409, {"message": str(e), "missing": e.missing})
    except QuotaExceeded as e:
        raise HTTPException(429, {"message": str(e), "resets_on": e.resets_on.isoformat()})
    except SpendCapReached as e:
        raise HTTPException(503, {"message": str(e), "resets_on": e.resets_on.isoformat()})
    except KeyError:
        raise HTTPException(404, "not found")


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
