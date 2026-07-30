"""Google Photos Picker API client (spec-092).

Plain REST against ``photospicker.googleapis.com`` (no official Python client
lib for the Picker API). Flow: create a session -> user picks photos in their
browser -> poll the session -> list the picked items -> download the bytes.

Auth is an ``AuthorizedSession`` that injects and auto-refreshes the bearer
token. The token is required even for ``baseUrl`` byte downloads -- a plain GET
without the header returns 403.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

from gphotos.types import PickedItem, PickerSession

logger = logging.getLogger(__name__)

API_ROOT = "https://photospicker.googleapis.com/v1"
# baseUrl download suffix: "=d" returns the full bytes with all EXIF except
# location (Google strips GPS on download by design).
DOWNLOAD_SUFFIX = "=d"


class PickerClient:
    """Thin REST wrapper over the Photos Picker API.

    Pass real ``credentials`` in production; pass a ``session`` double in tests.
    """

    def __init__(self, credentials: Optional[Credentials] = None, session=None) -> None:
        if session is None:
            if credentials is None:
                raise ValueError("PickerClient needs credentials or a session")
            session = AuthorizedSession(credentials)
        self._session = session

    def create_session(self) -> PickerSession:
        resp = self._session.post(f"{API_ROOT}/sessions", json={})
        resp.raise_for_status()
        sess = PickerSession.from_api(resp.json())
        logger.info("Created picker session %s", sess.id)
        return sess

    def get_session(self, session_id: str) -> PickerSession:
        resp = self._session.get(f"{API_ROOT}/sessions/{session_id}")
        resp.raise_for_status()
        return PickerSession.from_api(resp.json())

    def poll_until_ready(
        self,
        session: PickerSession,
        on_wait: Optional[Callable[[PickerSession], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> PickerSession:
        """Poll until the user finishes picking, honoring the server cadence.

        Interval and timeout are taken from the initial session; re-fetched
        session responses may omit ``pollingConfig``, so we do not re-read them.
        """
        interval = max(session.polling.poll_interval_s, 1.0)
        timeout = session.polling.timeout_s
        waited = 0.0
        current = session
        while not current.media_items_set:
            if waited >= timeout:
                raise TimeoutError(
                    f"Picker session {current.id} timed out after {waited:.0f}s"
                )
            if on_wait:
                on_wait(current)
            sleep(interval)
            waited += interval
            current = self.get_session(current.id)
        logger.info("Session %s ready after %.0fs", current.id, waited)
        return current

    def list_media_items(self, session_id: str, page_size: int = 100) -> list[PickedItem]:
        items: list[PickedItem] = []
        page_token: Optional[str] = None
        while True:
            params = {"sessionId": session_id, "pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token
            resp = self._session.get(f"{API_ROOT}/mediaItems", params=params)
            resp.raise_for_status()
            data = resp.json()
            for raw in data.get("mediaItems", []):
                items.append(PickedItem.from_api(raw))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        logger.info("Listed %d picked item(s)", len(items))
        return items

    def download_item(self, item: PickedItem, dest: Path) -> Path:
        url = item.base_url + DOWNLOAD_SUFFIX
        resp = self._session.get(url)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    def delete_session(self, session_id: str) -> None:
        """Best-effort cleanup; failures are logged, not raised."""
        try:
            self._session.delete(f"{API_ROOT}/sessions/{session_id}")
        except Exception as exc:  # noqa: BLE001 -- cleanup must never break a run
            logger.warning("Failed to delete session %s: %s", session_id, exc)
