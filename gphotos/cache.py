"""Local download cache for picked Google Photos media (spec-092).

Materializes picked items into a directory that the existing ``discover_images``
pipeline step can consume unchanged. Idempotent: an item already on disk (per a
small manifest) is skipped, so re-runs are cheap and resumable.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from gphotos.picker import PickerClient
from gphotos.types import DownloadedItem, PickedItem

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
MANIFEST_NAME = ".gphotos_manifest.json"


class DownloadCache:
    """A directory of downloaded media plus an id -> filename manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / MANIFEST_NAME

    def materialize(
        self, client: PickerClient, items: list[PickedItem]
    ) -> list[DownloadedItem]:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        out: list[DownloadedItem] = []
        for item in items:
            name = self._filename_for(item, manifest)
            dest = self.root / name
            if item.id in manifest and dest.exists():
                logger.debug("Cache hit %s", name)
            else:
                client.download_item(item, dest)
                manifest[item.id] = name
                logger.info("Downloaded %s", name)
            out.append(DownloadedItem(item=item, local_path=str(dest)))
        self._save_manifest(manifest)
        return out

    def _filename_for(self, item: PickedItem, manifest: dict) -> str:
        if item.id in manifest:
            return manifest[item.id]
        base = _SAFE.sub("_", item.filename or item.id)
        existing = set(manifest.values())
        stem, dot, ext = base.rpartition(".")
        candidate = base
        i = 1
        while candidate in existing or (self.root / candidate).exists():
            candidate = f"{stem}_{i}.{ext}" if dot else f"{base}_{i}"
            i += 1
        return candidate

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                logger.warning("Corrupt cache manifest; rebuilding")
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
