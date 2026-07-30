"""Adapter: import a Google Photos selection into a local source directory (spec-092).

This is the seam between Google Photos and the rest of sim-bench. It combines
auth + picker + cache into a single call that returns a directory the existing
``discover_images`` pipeline step consumes unchanged -- the pipeline never knows
the images came from Google.

The same function backs the standalone E2E script and (later, P5) the UI button.
``auth``/``client`` are injectable so the flow is unit-testable without network.
"""
from __future__ import annotations

import logging
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from gphotos.auth import GooglePhotosAuth
from gphotos.cache import DownloadCache
from gphotos.picker import PickerClient
from gphotos.types import DownloadedItem, PickerSession

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Outcome of a Google Photos import: a dir + the items that landed in it."""

    source_directory: Path
    items: list[DownloadedItem] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)


def import_from_google_photos(
    client_secret_path: str | Path = "client_secret.json",
    out_dir: str | Path = "./_gphotos/cache",
    *,
    open_browser: bool = True,
    on_picker_url: Optional[Callable[[str], None]] = None,
    on_wait: Optional[Callable[[PickerSession], None]] = None,
    auth: Optional[GooglePhotosAuth] = None,
    client: Optional[PickerClient] = None,
) -> IngestResult:
    """Run the full Picker import and return a dir ready for ``discover_images``.

    Steps: authenticate -> create session -> surface the picker URI -> poll until
    the user finishes -> list picked items -> download into ``out_dir`` -> clean up
    the session. Returns the directory plus the downloaded items.
    """
    if client is None:
        creds = (auth or GooglePhotosAuth(client_secret_path)).get_credentials(
            open_browser=open_browser
        )
        client = PickerClient(creds)

    session = client.create_session()
    logger.info("Picker session %s", session.id)
    if on_picker_url:
        on_picker_url(session.picker_uri)
    elif open_browser and session.picker_uri:
        webbrowser.open(session.picker_uri)

    ready = client.poll_until_ready(session, on_wait=on_wait)
    items = client.list_media_items(ready.id)
    downloaded = DownloadCache(out_dir).materialize(client, items)
    client.delete_session(ready.id)

    logger.info("Imported %d item(s) -> %s", len(downloaded), out_dir)
    return IngestResult(source_directory=Path(out_dir), items=downloaded)
