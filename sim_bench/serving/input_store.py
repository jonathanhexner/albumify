"""Content-addressed store of staged inputs (spec-112 Phase A2, TW1.2).

Each photo is staged once, at a path fixed by its content:

    <root>/<sha16>/<stem>.jpg      768 px JPEG of the original (see :mod:`sim_bench.serving.staging`)

``sha16`` hashes the original file's bytes plus :data:`STAGING_VERSION`, so the same photo staged by
two jobs lands on the same path, and a change to how photos are staged gives every photo a new path.
The stem keeps the original file name, which is the photo id everywhere downstream.

A stable path is what makes the feature cache work across jobs: the pipeline cache is keyed by image
path and invalidated by mtime, so a file here is written once and never rewritten.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

from sim_bench.serving.staging import ANALYSIS_EDGE, downsample

logger = logging.getLogger(__name__)

__all__ = ["InputStore", "STAGING_VERSION"]

# Bump when downsample() changes what it writes (size, quality, EXIF handling).
STAGING_VERSION = f"{ANALYSIS_EDGE}px-q85-v1"

_KEY = re.compile(r"^[0-9a-f]{16}/[^/\\]+\.jpg$")


class InputStore:
    """Staged inputs under one root (a local folder, or a Modal Volume mount)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def key_for(self, original: Path) -> str:
        digest = hashlib.sha256()
        with open(original, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(f"|{STAGING_VERSION}".encode())
        return f"{digest.hexdigest()[:16]}/{Path(original).stem}.jpg"

    def path(self, key: str) -> Path:
        if not _KEY.match(key or ""):
            raise ValueError(f"invalid input key {key!r}")
        return self.root / key

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def put(self, original: Path) -> str:
        """Stage ``original`` unless it is already here; return its key."""
        key = self.key_for(original)
        if not self.has(key):
            self.put_bytes(key, downsample(Path(original)))
        return key

    def put_bytes(self, key: str, data: bytes) -> Path:
        """Write already-staged bytes under ``key`` (atomic: a reader never sees half a file)."""
        dest = self.path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return dest
