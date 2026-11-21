"""spec-120 R1: every public route except /health needs a valid sign-in token from the provider."""
from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from sim_bench.control.db import ControlDB
from sim_bench.public_api.accounts import AccountRepo
from sim_bench.public_api.app import create_app
from sim_bench.public_api.auth import AuthError, AuthVerifier, bearer_token
from tests.public_api.conftest import ISSUER


@pytest.fixture
def verifier(signer):
    return AuthVerifier(ISSUER, keys=signer.keys)


class ut_AuthVerifier:
    def test_a_valid_token_gives_the_user_and_a_lower_case_email(self, verifier, signer):
        claims = verifier.verify(signer.token(sub="user_9", email="Ann@Example.COM"))
        assert (claims.subject, claims.email) == ("user_9", "ann@example.com")

    @pytest.mark.parametrize("case", ["missing", "expired", "other_issuer", "other_key", "tampered", "no_email",
                                      "hs256_with_public_key", "garbage"])
    def test_a_bad_token_is_refused(self, verifier, signer, case):
        good = signer.token()
        token = {
            "missing": None,
            "expired": signer.token(expires_in=-60),
            "other_issuer": signer.token(issuer="https://evil.example"),
            "other_key": signer.token(key=rsa.generate_private_key(public_exponent=65537, key_size=2048)),
            "tampered": good.rsplit(".", 2)[0] + "." + jwt.utils.base64url_encode(b'{"sub":"admin"}').decode()
                        + "." + good.rsplit(".", 1)[1],
            "no_email": signer.token(email=None),
            "hs256_with_public_key": jwt.encode({"sub": "x", "iss": ISSUER, "iat": int(time.time()),
                                                 "exp": int(time.time()) + 60, "email": "a@b.c"},
                                                "not-the-key", algorithm="HS256"),
            "garbage": "not.a.token",
        }[case]
        with pytest.raises(AuthError):
            verifier.verify(token)

    def test_bearer_header_parsing(self):
        assert bearer_token("Bearer abc") == "abc"
        assert bearer_token("bearer  abc ") == "abc"
        assert bearer_token("Basic abc") is None
        assert bearer_token("Bearer ") is None
        assert bearer_token(None) is None


class ut_PublicAppAuth:
    @pytest.fixture
    def client(self, verifier, control_db_url):
        accounts = AccountRepo(ControlDB(control_db_url))
        accounts.invite("ann@example.com")
        return TestClient(create_app(verifier=verifier, accounts=accounts))

    def test_health_needs_no_token(self, client):
        assert client.get("/health").status_code == 200

    @pytest.mark.parametrize("path,method", [("/v1/me", "get"), ("/v1/me/consent", "post")])
    def test_every_other_route_needs_a_valid_token(self, client, signer, path, method):
        call = getattr(client, method)
        assert call(path).status_code == 401
        assert call(path, headers={"Authorization": "Bearer junk"}).status_code == 401
        assert call(path, headers={"Authorization": f"Bearer {signer.token(expires_in=-60)}"}).status_code == 401
        assert call(path, headers={"Authorization": f"Bearer {signer.token()}"}).status_code == 200

    def test_every_route_is_covered(self, client):
        """A new route must be added to the 401 test above (or be /health)."""
        paths = {r.path for r in client.app.routes if getattr(r, "include_in_schema", False)}
        assert paths == {"/health", "/v1/me", "/v1/me/consent"}
