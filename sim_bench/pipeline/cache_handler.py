"""Universal cache handler with metadata + flexible data storage."""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from sqlalchemy.orm import Session

from sim_bench.api.database.models import UniversalCache

logger = logging.getLogger(__name__)

# spec-117 R12: photos in the shared input store are cached by content, the same key on this PC and the L4
CONTENT_PREFIX = "staged:"
_CONTENT = re.compile(r"^([0-9a-f]{16})/([^/]+\.jpg)(:face_\d+)?$")
_CHUNK = 5000         # paths per IN query: below SQLite (>= 3.32: 32766) and Postgres (65535) parameter limits


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key identifying a cached feature."""
    image_path: str
    feature_type: str
    model_name: str
    
    def __post_init__(self):
        """Validate + normalize key components.

        SIGHTING-112: the cache key IS the image path string, so the same image
        reached via ``D:/album\\a.jpg`` (studio, forward-slash folder) vs
        ``D:\\album\\a.jpg`` (batch) produced DIFFERENT keys and never shared
        cache. Normalize path separators (``os.path.normpath`` — separators only,
        case preserved so existing Windows rows still match) so one image → one
        key regardless of how the folder was typed. Frozen dataclass → set via
        ``object.__setattr__``.
        """
        if not self.image_path:
            raise ValueError("image_path cannot be empty")
        if not self.feature_type:
            raise ValueError("feature_type cannot be empty")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        object.__setattr__(self, "image_path", os.path.normpath(self.image_path))
    
    def to_string(self) -> str:
        """Convert to string for logging/debugging."""
        return f"{self.image_path}:{self.feature_type}:{self.model_name}"


class UniversalCacheHandler:
    """
    Universal cache handler with three-method interface.

    Stores opaque bytes with metadata. Steps handle serialization/deserialization.

    spec-117 R12 -- one cache for this PC and the L4 (both on ``DATABASE_URL``):
      * reads and writes are batched: one query per step (``load_many`` / ``store_many``), not one per photo,
        because on Neon every round trip costs ~0.2 s;
      * a photo in the shared input store (``inputs_root``, content-addressed ``<sha16>/<name>.jpg``) is cached
        under ``staged:<sha16>/<name>.jpg`` -- the same key on both machines whatever the folder is called -- and
        is never invalidated by mtime (its content fixes its key; an upload to Modal gives it a new mtime). A row
        an older run stored under the full staged path is still found (same query) until it is rewritten.
    """

    def __init__(self, session: Session, inputs_root: Optional[str] = None):
        """
        Initialize cache handler.

        Args:
            session: SQLAlchemy database session
            inputs_root: the shared input store's folder (``JobStore.inputs.root``), if the images come from it
        """
        self._session = session
        self._inputs_root = (os.path.normpath(str(inputs_root)) + os.sep) if inputs_root else None

    # --- keys -------------------------------------------------------------------

    def db_key(self, image_path: str) -> Tuple[str, bool]:
        """The path as stored, and whether it is content-addressed (no mtime check)."""
        root = self._inputs_root
        if root and image_path.startswith(root):
            m = _CONTENT.match(image_path[len(root):].replace("\\", "/"))
            if m:
                return f"{CONTENT_PREFIX}{m.group(1)}/{m.group(2)}{m.group(3) or ''}", True
        return image_path, False

    # --- write ------------------------------------------------------------------

    def store_to_cache(
        self,
        key: CacheKey,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store data as opaque bytes with metadata.

        Args:
            key: Cache key identifying the feature
            data: Opaque bytes to store (step handles serialization)
            metadata: Optional metadata dict (model_version, etc.)
        """
        self.store_many([(key, data)], metadata)

    def store_many(self, items: List[Tuple[CacheKey, bytes]], metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store many entries with one lookup query per (feature, model) and ONE commit. Returns how many."""
        model_version = (metadata or {}).get("model_version")
        now = datetime.utcnow()
        rows: Dict[Tuple[str, str, str], Tuple[bytes, float]] = {}
        for key, data in items:
            if not data:
                logger.warning(f"Attempted to store empty data for {key.to_string()}")
                continue
            path, content = self.db_key(key.image_path)
            mtime = 0.0 if content else self._get_image_mtime(key.image_path)
            if mtime < 0:
                logger.warning(f"Cannot cache for {key.image_path} - file not found")
                continue
            rows[(path, key.feature_type, key.model_name)] = (data, mtime)
        if not rows:
            return 0
        existing = {(e.image_path, e.feature_type, e.model_name): e for e in self._query(rows.keys())}
        for k, (data, mtime) in rows.items():
            entry = existing.get(k)
            if entry is not None:
                entry.data_blob, entry.model_version, entry.image_mtime, entry.last_accessed = \
                    data, model_version, mtime, now
            else:
                self._session.add(UniversalCache(
                    image_path=k[0], feature_type=k[1], model_name=k[2], model_version=model_version,
                    data_blob=data, image_mtime=mtime, created_at=now, last_accessed=now))
        self._session.commit()
        return len(rows)

    # --- read -------------------------------------------------------------------

    def load_from_cache(
        self,
        keys: List[CacheKey]
    ) -> Dict[str, Tuple[bytes, Dict[str, Any]]]:
        """
        Load data bytes and metadata for keys (see :meth:`load_many`).

        Returns:
            Dict mapping key string -> (data_bytes, metadata_dict)
            Only includes valid (mtime matches) entries
        """
        return self.load_many(keys)

    def load_many(self, keys: List[CacheKey]) -> Dict[str, Tuple[bytes, Dict[str, Any]]]:
        """Every valid entry for ``keys`` in one query per (feature, model) -- one per step.

        Returns ``"<image_path>:<feature_type>:<model_name>" -> (data_bytes, metadata)`` in the caller's paths.
        A path-keyed entry whose file is gone or changed (mtime) is invalidated and deleted."""
        if not keys:
            return {}
        wanted: Dict[Tuple[str, str, str], List[Tuple[str, bool]]] = {}
        for key in keys:
            key_str = f"{key.image_path}:{key.feature_type}:{key.model_name}"
            path, content = self.db_key(key.image_path)
            wanted.setdefault((path, key.feature_type, key.model_name), []).append((key_str, content))
            if content:   # an older run's row under the full staged path (mtime-checked)
                wanted.setdefault((key.image_path, key.feature_type, key.model_name), []).append((key_str, False))

        results: Dict[str, Tuple[bytes, Dict[str, Any]]] = {}
        from_content: set = set()
        to_delete = []
        for entry in self._query(wanted.keys()):
            for key_str, content in wanted.get((entry.image_path, entry.feature_type, entry.model_name), []):
                if not content:
                    current_mtime = self._get_image_mtime(entry.image_path)
                    if current_mtime < 0 or entry.image_mtime != current_mtime:
                        # File not found or modified - invalidate cache
                        to_delete.append(entry)
                        logger.debug(f"Cache invalidated for {entry.image_path} "
                                     f"({entry.feature_type}/{entry.model_name})")
                        continue
                    if key_str in from_content:
                        continue              # the content-keyed row wins
                elif key_str in results:
                    from_content.add(key_str)
                results[key_str] = (entry.data_blob, {
                    "model_version": entry.model_version,
                    "image_mtime": entry.image_mtime,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                })
                if content:
                    from_content.add(key_str)

        for entry in {id(e): e for e in to_delete}.values():
            self._session.delete(entry)
        # Always end the transaction: the step computes for minutes after this read, and Postgres (Neon) ends a
        # connection left idle inside a transaction (spec-117 T8.2: triage_content, 8.5 min on the L4, failed).
        self._session.commit()
        return results

    def _query(self, triples) -> List[UniversalCache]:
        """Rows for (image_path, feature_type, model_name) triples: one IN query per (feature, model), chunked
        below the drivers' parameter limits."""
        by_feature: Dict[Tuple[str, str], List[str]] = {}
        for path, feature, model in triples:
            by_feature.setdefault((feature, model), []).append(path)
        out: List[UniversalCache] = []
        for (feature, model), paths in by_feature.items():
            paths = sorted(set(paths))
            for i in range(0, len(paths), _CHUNK):
                out.extend(self._session.query(UniversalCache).filter(
                    UniversalCache.feature_type == feature, UniversalCache.model_name == model,
                    UniversalCache.image_path.in_(paths[i:i + _CHUNK])).all())
        return out

    def search_keys(
        self,
        filter: Dict[str, Any]
    ) -> List[CacheKey]:
        """
        Find keys matching metadata filter.
        
        Args:
            filter: Dict with keys like:
                - image_path: str (exact match)
                - feature_type: str (exact match)
                - model_name: str (exact match)
                - model_version: str (exact match)
        
        Returns:
            List of matching CacheKey objects
        """
        query = self._session.query(UniversalCache)
        
        if "image_path" in filter:
            query = query.filter(UniversalCache.image_path == filter["image_path"])
        if "feature_type" in filter:
            query = query.filter(UniversalCache.feature_type == filter["feature_type"])
        if "model_name" in filter:
            query = query.filter(UniversalCache.model_name == filter["model_name"])
        if "model_version" in filter:
            query = query.filter(UniversalCache.model_version == filter["model_version"])
        
        entries = query.all()
        
        return [
            CacheKey(
                image_path=entry.image_path,
                feature_type=entry.feature_type,
                model_name=entry.model_name
            )
            for entry in entries
        ]
    
    def _get_image_mtime(self, image_path: str) -> float:
        """
        Get image modification time, or -1 if file not found.
        
        Handles composite keys (e.g., "path/to/image.jpg:face_0") by extracting
        the actual image path before checking mtime.
        """
        # Extract actual image path from composite keys
        actual_path = image_path.split(':face_')[0] if ':face_' in image_path else image_path
        
        try:
            return os.path.getmtime(actual_path)
        except (FileNotFoundError, OSError):
            return -1.0
