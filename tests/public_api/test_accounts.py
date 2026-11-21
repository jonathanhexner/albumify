"""spec-120 R2 (invite-only accounts) and R3 (face-grouping consent before any upload)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sim_bench.control.db import ControlDB
from sim_bench.public_api.accounts import AccountRepo, ConsentRequired, NotInvited
from sim_bench.public_api.app import create_app
from sim_bench.public_api.auth import AuthVerifier, Claims
from tests.public_api.conftest import ISSUER


@pytest.fixture
def repo(control_db_url):
    return AccountRepo(ControlDB(control_db_url))


class ut_AccountRepo:
    def test_uninvited_email_is_waitlisted(self, repo):
        with pytest.raises(NotInvited):
            repo.get_or_create(Claims("user_1", "stranger@example.com"))

    def test_an_invited_email_gets_one_account(self, repo):
        repo.invite("Ann@Example.com ")
        first = repo.get_or_create(Claims("user_1", "ann@example.com"))
        again = repo.get_or_create(Claims("user_1", "ann@example.com"))
        assert first.id == again.id == "user_1" and first.email == "ann@example.com"

    def test_an_email_taken_by_another_sign_in_is_not_merged(self, repo):
        repo.invite("ann@example.com")
        repo.get_or_create(Claims("user_1", "ann@example.com"))
        with pytest.raises(NotInvited):
            repo.get_or_create(Claims("user_2", "ann@example.com"))

    def test_upload_needs_face_consent(self, repo):
        repo.invite("ann@example.com")
        account = repo.get_or_create(Claims("user_1", "ann@example.com"))
        with pytest.raises(ConsentRequired):
            AccountRepo.require_consent(account)
        consented = repo.record_consent("user_1")
        AccountRepo.require_consent(consented)                       # no error now
        again = repo.record_consent("user_1").face_consent_at          # recorded once, not moved
        assert again.replace(tzinfo=None) == consented.face_consent_at.replace(tzinfo=None)   # SQLite drops the zone


class ut_AccountRoutes:
    @pytest.fixture
    def client(self, signer, repo):
        repo.invite("ann@example.com")
        return TestClient(create_app(verifier=AuthVerifier(ISSUER, keys=signer.keys), accounts=repo))

    def test_uninvited_sign_in_gets_403_waiting_list(self, client, signer):
        r = client.get("/v1/me", headers={"Authorization": f"Bearer {signer.token(email='bob@example.com')}"})
        assert r.status_code == 403 and "waiting list" in r.json()["detail"]

    def test_consent_is_recorded_through_the_api(self, client, signer):
        auth = {"Authorization": f"Bearer {signer.token()}"}
        assert client.get("/v1/me", headers=auth).json()["face_consent"] is False
        assert client.post("/v1/me/consent", headers=auth).json()["face_consent"] is True
        assert client.get("/v1/me", headers=auth).json()["face_consent"] is True
