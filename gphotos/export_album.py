"""Push a refined album back to Google Photos (spec-092, P6).

Reads a list of image paths (e.g. ``PipelineResult.selected_images``), uploads
each, and batch-creates them into an app-owned album. Idempotent via a manifest
keyed by album title: a re-run reuses the same album and skips already-uploaded
files, so interrupted exports resume instead of duplicating.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from gphotos.auth import APPEND_SCOPE, LIBRARY_TOKEN_PATH, GooglePhotosAuth
from gphotos.uploader import BATCH_LIMIT, LibraryClient

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path.home() / ".sim_bench" / "gphotos_export_manifest.json"


@dataclass
class ExportResult:
    """Outcome of an album export."""

    album_id: str
    album_title: str
    uploaded: int = 0
    created: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def export_album_to_google_photos(
    image_paths: Iterable[str | Path],
    album_title: str,
    *,
    client_secret: str | Path = "client_secret.json",
    manifest_path: str | Path | None = None,
    client: Optional[LibraryClient] = None,
    auth: Optional[GooglePhotosAuth] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> ExportResult:
    """Upload ``image_paths`` into an app-owned album named ``album_title``."""
    paths = [Path(p) for p in image_paths]
    if client is None:
        creds = (
            auth
            or GooglePhotosAuth(
                client_secret, scopes=[APPEND_SCOPE], token_path=LIBRARY_TOKEN_PATH
            )
        ).get_credentials()
        client = LibraryClient(creds)

    manifest = _load_manifest(manifest_path)
    entry = manifest.get(album_title, {})
    album_id = entry.get("album_id") or client.create_album(album_title)
    done: set[str] = set(entry.get("uploaded", []))

    pending = [p for p in paths if str(p) not in done]
    skipped = len(paths) - len(pending)

    tokens: list[tuple[str, str, str]] = []  # (upload_token, filename, src_path)
    failed: list[str] = []
    for i, p in enumerate(pending, 1):
        if not p.exists():
            logger.warning("skipping missing file %s", p)
            failed.append(str(p))
        else:
            try:
                tokens.append((client.upload_bytes(p), p.name, str(p)))
            except Exception as exc:  # one bad file must not abort the export
                logger.warning("upload failed %s: %s", p, exc)
                failed.append(str(p))
        if on_progress:
            on_progress(i, len(pending))

    created = 0
    for chunk in _chunks(tokens, BATCH_LIMIT):
        created += client.batch_create(album_id, [(t, n) for t, n, _ in chunk])
        done.update(src for _, _, src in chunk)

    manifest[album_title] = {"album_id": album_id, "uploaded": sorted(done)}
    _save_manifest(manifest_path, manifest)

    return ExportResult(
        album_id=album_id,
        album_title=album_title,
        uploaded=len(tokens),
        created=created,
        skipped=skipped,
        failed=failed,
    )


def _load_manifest(path: str | Path | None) -> dict:
    p = Path(path or DEFAULT_MANIFEST)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            logger.warning("corrupt export manifest; starting fresh")
    return {}


def _save_manifest(path: str | Path | None, manifest: dict) -> None:
    p = Path(path or DEFAULT_MANIFEST)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
