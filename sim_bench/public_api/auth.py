"""Sign-in tokens for the public API (spec-120 R1).

The website signs users in with a hosted provider (Clerk). Each request carries the provider's session token, a JWT
signed with the provider's private key; this module checks it against the provider's published public keys (JWKS):
signature, issuer, expiry. No password or session is stored here.

Clerk's default session token has no email; the session-token template must add ``"email": "{{user.primary_email_address}}"``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

__all__ = ["AuthError", "AuthVerifier", "Claims"]

ALGORITHMS = ["RS256"]


class AuthError(Exception):
    """The token is missing, malformed, expired, from another issuer or not signed by the provider (-> 401)."""


@dataclass(frozen=True, slots=True)
class Claims:
    subject: str            # the provider's user id: the account id
    email: str              # lower-case


class AuthVerifier:
    """Config in ``__init__``: the provider's issuer URL. ``keys`` is anything with ``get_signing_key_from_jwt``
    (``jwt.PyJWKClient`` by default, which fetches and caches the provider's keys); tests pass their own."""

    def __init__(self, issuer: str, *, jwks_url: Optional[str] = None, audience: Optional[str] = None,
                 keys=None, leeway_s: int = 5):
        self.issuer = issuer.rstrip("/")
        self.jwks_url = jwks_url or f"{self.issuer}/.well-known/jwks.json"
        self.audience = audience
        self.leeway_s = leeway_s
        self._keys = keys if keys is not None else jwt.PyJWKClient(self.jwks_url, cache_keys=True)

    def verify(self, token: Optional[str]) -> Claims:
        if not token:
            raise AuthError("no sign-in token")
        try:
            key = self._keys.get_signing_key_from_jwt(token).key
            data = jwt.decode(token, key, algorithms=ALGORITHMS, issuer=self.issuer, audience=self.audience,
                              leeway=self.leeway_s,
                              options={"require": ["exp", "iat", "iss", "sub"], "verify_aud": self.audience is not None})
        except jwt.PyJWTError as e:          # never log the token itself
            raise AuthError(f"invalid sign-in token ({type(e).__name__})") from e
        email = data.get("email")
        if not isinstance(email, str) or "@" not in email:
            raise AuthError("sign-in token has no email (add it to the provider's session token)")
        return Claims(subject=str(data["sub"]), email=email.strip().lower())


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    """The token from an ``Authorization: Bearer <token>`` header, else None."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None
