"""Shared helpers for the public API tests: a locally made RSA key stands in for the sign-in provider's."""
from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "https://clerk.albumify.test"


class Signer:
    """Makes the provider's session tokens; ``keys`` plays the provider's published key set (JWKS)."""

    def __init__(self):
        self.private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.keys = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=self.private.public_key()))

    def token(self, sub="user_1", email="ann@example.com", *, issuer=ISSUER, expires_in=300, key=None, **extra):
        now = int(time.time())
        claims = {"sub": sub, "iss": issuer, "iat": now, "exp": now + expires_in, **extra}
        if email is not None:
            claims["email"] = email
        return jwt.encode(claims, key or self.private, algorithm="RS256")


@pytest.fixture(scope="session")
def signer() -> Signer:
    return Signer()
