"""OAuth for the Google Photos Picker API (spec-092).

Installed-app (loopback) flow: a local server on 127.0.0.1 captures the consent
redirect. The refresh token is cached on disk so the user consents once;
subsequent runs refresh the access token silently.

Token storage is a JSON file under ``~/.sim_bench/`` by default. Swapping in an
OS keyring store is a P7 hardening follow-up (FR-008); the file store already
satisfies "persist + silent refresh".
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

PICKER_SCOPE = "https://www.googleapis.com/auth/photospicker.mediaitems.readonly"
# Export (Library API, post-2025 app-created-data only). Restricted scopes, but
# usable in Testing mode without CASA.
APPEND_SCOPE = "https://www.googleapis.com/auth/photoslibrary.appendonly"
EDIT_SCOPE = "https://www.googleapis.com/auth/photoslibrary.edit.appcreateddata"
DEFAULT_TOKEN_PATH = Path.home() / ".sim_bench" / "gphotos_token.json"
# Export uses a separate token file so it doesn't force re-consent of the picker.
LIBRARY_TOKEN_PATH = Path.home() / ".sim_bench" / "gphotos_library_token.json"


class GooglePhotosAuth:
    """Acquire and cache OAuth credentials for the Picker API."""

    def __init__(
        self,
        client_secret_path: str | os.PathLike,
        scopes: Sequence[str] = (PICKER_SCOPE,),
        token_path: str | os.PathLike = DEFAULT_TOKEN_PATH,
    ) -> None:
        self.client_secret_path = Path(client_secret_path)
        self.scopes = list(scopes)
        self.token_path = Path(token_path)

    def get_credentials(self, open_browser: bool = True) -> Credentials:
        """Return valid credentials, refreshing or running consent as needed."""
        creds = self._load()
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing Google Photos access token")
            creds.refresh(Request())
            self._save(creds)
            return creds
        return self._run_flow(open_browser)

    def clear(self) -> None:
        """Delete the cached token (forces re-consent next time)."""
        if self.token_path.exists():
            self.token_path.unlink()

    def _run_flow(self, open_browser: bool) -> Credentials:
        if not self.client_secret_path.exists():
            raise FileNotFoundError(
                f"OAuth client secret not found: {self.client_secret_path}. "
                "Download it from Google Cloud Console (OAuth client -> Desktop app)."
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self.client_secret_path), self.scopes
        )
        creds = flow.run_local_server(port=0, open_browser=open_browser)
        self._save(creds)
        return creds

    def _load(self) -> Optional[Credentials]:
        if not self.token_path.exists():
            return None
        try:
            return Credentials.from_authorized_user_file(
                str(self.token_path), self.scopes
            )
        except (ValueError, OSError) as exc:
            logger.warning("Could not load cached token (%s); re-authenticating", exc)
            return None

    def _save(self, creds: Credentials) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(creds.to_json(), encoding="utf-8")
        try:
            os.chmod(self.token_path, 0o600)  # best-effort; no-op on some Windows FS
        except OSError:
            pass
        logger.info("Saved Google Photos token to %s", self.token_path)
