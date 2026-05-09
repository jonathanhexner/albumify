"""Embed cache for FaceClusteringPipeline.

Caches the expensive embed stage (InsightFace detection + embedding) so that
subsequent runs on the same image directory skip model inference automatically.

Cache location: {output_dir}/.embed_cache/
Cache key:      SHA-256 fingerprint of image directory + EMBED_CACHE_VERSION

Cache layout:
    .embed_cache/
        cache_meta.json    — fingerprint, version, timestamps, stats
        faces_cache.pkl    — List[FaceRecord] with embeddings + aligned_face
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from face_cluster.types import FaceRecord

logger = logging.getLogger(__name__)

EMBED_CACHE_VERSION = "1"
_CACHE_DIR_NAME = ".embed_cache"
_FACES_CACHE_FILE = "faces_cache.pkl"
_CACHE_META_FILE = "cache_meta.json"


def compute_image_fingerprint(image_dir: Path, extensions: Set[str]) -> str:
    """Deterministic SHA-256 hash of the image set in a directory.

    Catches: images added/removed, images modified (mtime or size changes),
    filename renames, files moved to/from subdirectories.

    Does NOT catch: bit-identical file replacement with a fresh mtime and
    identical size (extremely rare; causes unnecessary re-embed, not wrong results).
    """
    entries = []
    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() in extensions:
            stat = path.stat()
            entries.append(
                f"{path.relative_to(image_dir)}|{stat.st_size}|{stat.st_mtime_ns}"
            )
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return f"sha256:{digest}"


def save_embed_cache(
    output_dir: Path,
    faces: List[FaceRecord],
    image_fingerprint: str,
    image_dir: Path,
    n_images: int,
    embed_time_s: float,
) -> None:
    """Atomically write embed cache to {output_dir}/.embed_cache/.

    Uses a write-to-temp-then-rename pattern to avoid corrupt state from
    interrupted writes.
    """
    cache_dir = output_dir / _CACHE_DIR_NAME

    # Write to a sibling temp dir, then atomically replace cache_dir
    tmp_dir = Path(tempfile.mkdtemp(dir=output_dir, prefix=".embed_cache_tmp_"))
    try:
        with open(tmp_dir / _FACES_CACHE_FILE, "wb") as f:
            pickle.dump(faces, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta: Dict = {
            "embed_cache_version": EMBED_CACHE_VERSION,
            "image_fingerprint": image_fingerprint,
            "image_dir": str(image_dir),
            "n_images": n_images,
            "n_faces": len(faces),
            "created_at": datetime.now().isoformat(),
            "embed_time_seconds": round(embed_time_s, 2),
        }
        with open(tmp_dir / _CACHE_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)

        logger.info(
            f"Embed cache written: {len(faces)} faces -> {cache_dir} "
            f"(embed took {embed_time_s:.1f}s)"
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def load_embed_cache(
    output_dir: Path,
) -> Optional[Tuple[List[FaceRecord], Dict]]:
    """Load raw cache artifacts from {output_dir}/.embed_cache/.

    Returns:
        (faces, meta) if both files exist and are readable; None otherwise.
        Does NOT validate — call validate_cache() after this.
    """
    cache_dir = output_dir / _CACHE_DIR_NAME
    meta_path = cache_dir / _CACHE_META_FILE
    faces_path = cache_dir / _FACES_CACHE_FILE

    if not meta_path.exists() or not faces_path.exists():
        return None

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read embed cache meta: {e}")
        return None

    try:
        with open(faces_path, "rb") as f:
            faces = pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load embed cache faces_cache.pkl: {e}")
        return None

    return faces, meta


def validate_cache(
    meta: Dict,
    faces: List[FaceRecord],
    image_fingerprint: str,
) -> Tuple[bool, str]:
    """Validate a loaded cache against current image state and code version.

    Returns:
        (is_valid, reason) where reason is "ok" on success or an explanation
        of why validation failed.
    """
    if meta.get("embed_cache_version") != EMBED_CACHE_VERSION:
        return False, (
            f"version mismatch: cache={meta.get('embed_cache_version')!r} "
            f"!= current={EMBED_CACHE_VERSION!r}"
        )
    if meta.get("image_fingerprint") != image_fingerprint:
        return False, "image fingerprint changed"
    if not faces:
        return False, "empty face list in cache"
    first = faces[0]
    if first.embedding_normalized is None:
        return False, "first face has no embedding"
    if first.embedding_normalized.shape != (512,):
        return False, f"bad embedding shape: {first.embedding_normalized.shape}"
    return True, "ok"


def clear_embed_cache(output_dir: Path) -> bool:
    """Delete the embed cache directory.

    Returns:
        True if a cache existed and was deleted; False if no cache was found.
    """
    cache_dir = output_dir / _CACHE_DIR_NAME
    if not cache_dir.exists():
        return False
    shutil.rmtree(cache_dir)
    logger.info(f"Embed cache cleared: {cache_dir}")
    return True


def get_cache_info(output_dir: Path) -> Optional[Dict]:
    """Return cache_meta.json as a dict if cache exists, else None.

    Safe to call at any time — returns None on any read error.
    """
    meta_path = output_dir / _CACHE_DIR_NAME / _CACHE_META_FILE
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
