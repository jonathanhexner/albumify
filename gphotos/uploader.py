"""Google Photos Library API upload client (spec-092, P6).

Post-2025 the Library API only touches *app-created* data. This client uploads
raw bytes, creates an app-owned album, and batch-creates media items into it --
the path for pushing a refined album back to the user's Google Photos.

Scope: ``photoslibrary.appendonly`` (restricted; works in Testing mode without
CASA). Media can only be added to albums this app created -- never the user's
existing albums (a Google constraint, not ours).
"""
from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from typing import Callable, Optional

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

API_ROOT = "https://photoslibrary.googleapis.com/v1"
UPLOAD_URL = f"{API_ROOT}/uploads"
BATCH_LIMIT = 50  # max newMediaItems per batchCreate call
_RETRY_STATUS = {429, 500, 502, 503, 504}


class LibraryClient:
    """Thin REST wrapper over the Library API upload + album endpoints.

    Pass real ``credentials`` in production; pass a ``session`` double in tests.
    Transient failures (429 / 5xx) are retried with exponential backoff.
    """

    def __init__(
        self,
        credentials: Optional[Credentials] = None,
        session=None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if session is None:
            if credentials is None:
                raise ValueError("LibraryClient needs credentials or a session")
            session = AuthorizedSession(credentials)
        self._session = session
        self._sleep = sleep

    def _post(self, url: str, *, max_retries: int = 4, **kw):
        delay = 1.0
        for attempt in range(max_retries + 1):
            resp = self._session.post(url, **kw)
            status = getattr(resp, "status_code", 200)
            if status in _RETRY_STATUS and attempt < max_retries:
                ra = (getattr(resp, "headers", {}) or {}).get("Retry-After")
                wait = float(ra) if ra and str(ra).replace(".", "", 1).isdigit() else delay
                logger.warning("HTTP %s on %s; retrying in %.1fs", status, url, wait)
                self._sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp

    def upload_bytes(self, path: str | Path) -> str:
        """Upload one file's bytes; return its (1-day) upload token."""
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        headers = {
            "Content-type": "application/octet-stream",
            "X-Goog-Upload-Content-Type": mime,
            "X-Goog-Upload-Protocol": "raw",
        }
        resp = self._post(UPLOAD_URL, data=path.read_bytes(), headers=headers)
        return resp.text

    def create_album(self, title: str) -> str:
        """Create an app-owned album; return its id."""
        resp = self._post(f"{API_ROOT}/albums", json={"album": {"title": title}})
        return resp.json()["id"]

    def batch_create(self, album_id: str, items: list[tuple[str, str]]) -> int:
        """Attach uploaded items to the album. ``items`` = [(token, filename)].

        Returns the count of successfully created media items.
        """
        body = {
            "albumId": album_id,
            "newMediaItems": [
                {"simpleMediaItem": {"uploadToken": tok, "fileName": name}}
                for tok, name in items
            ],
        }
        resp = self._post(f"{API_ROOT}/mediaItems:batchCreate", json=body)
        results = resp.json().get("newMediaItemResults", [])
        return sum(1 for r in results if "mediaItem" in r)
