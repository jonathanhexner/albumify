"""Universal cache handler with metadata + flexible data storage."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from sqlalchemy.orm import Session
from sqlalchemy import and_

from sim_bench.api.database.models import UniversalCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key identifying a cached feature."""
    image_path: str
    feature_type: str
    model_name: str
    
    def __post_init__(self):
        """Validate key components."""
        if not self.image_path:
            raise ValueError("image_path cannot be empty")
        if not self.feature_type:
            raise ValueError("feature_type cannot be empty")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
    
    def to_string(self) -> str:
        """Convert to string for logging/debugging."""
        return f"{self.image_path}:{self.feature_type}:{self.model_name}"


class UniversalCacheHandler:
    """
    Universal cache handler with three-method interface.
    
    Stores opaque bytes with metadata. Steps handle serialization/deserialization.
    """
    
    def __init__(self, session: Session):
        """
        Initialize cache handler.
        
        Args:
            session: SQLAlchemy database session
        """
        self._session = session
    
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
        if not data:
            logger.warning(f"Attempted to store empty data for {key.to_string()}")
            return
        
        # Get image mtime for invalidation
        image_mtime = self._get_image_mtime(key.image_path)
        if image_mtime < 0:
            logger.warning(f"Cannot cache for {key.image_path} - file not found")
            return
        
        model_version = (metadata or {}).get("model_version")
        
        # Check if entry exists
        existing = self._session.query(UniversalCache).filter(
            and_(
                UniversalCache.image_path == key.image_path,
                UniversalCache.feature_type == key.feature_type,
                UniversalCache.model_name == key.model_name
            )
        ).first()
        
        if existing:
            # Update existing entry
            existing.data_blob = data
            existing.model_version = model_version
            existing.image_mtime = image_mtime
            existing.last_accessed = datetime.utcnow()
        else:
            # Create new entry
            entry = UniversalCache(
                image_path=key.image_path,
                feature_type=key.feature_type,
                model_name=key.model_name,
                model_version=model_version,
                data_blob=data,
                image_mtime=image_mtime,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            self._session.add(entry)
        
        self._session.commit()
    
    def load_from_cache(
        self,
        keys: List[CacheKey]
    ) -> Dict[str, Tuple[bytes, Dict[str, Any]]]:
        """
        Load data bytes and metadata for keys.
        
        Validates mtime to ensure cache entries are still valid.
        
        Args:
            keys: List of cache keys to load
        
        Returns:
            Dict mapping key string -> (data_bytes, metadata_dict)
            Only includes valid (mtime matches) entries
        """
        if not keys:
            return {}
        
        # Build query for all keys
        conditions = []
        for key in keys:
            conditions.append(
                and_(
                    UniversalCache.image_path == key.image_path,
                    UniversalCache.feature_type == key.feature_type,
                    UniversalCache.model_name == key.model_name
                )
            )
        
        from sqlalchemy import or_
        entries = self._session.query(UniversalCache).filter(
            or_(*conditions)
        ).all()
        
        results = {}
        to_delete = []
        
        for entry in entries:
            # Check mtime validity
            current_mtime = self._get_image_mtime(entry.image_path)
            if current_mtime < 0 or entry.image_mtime != current_mtime:
                # File not found or modified - invalidate cache
                to_delete.append(entry)
                logger.debug(f"Cache invalidated for {entry.image_path} ({entry.feature_type}/{entry.model_name})")
                continue
            
            # Create key string for lookup
            key_str = f"{entry.image_path}:{entry.feature_type}:{entry.model_name}"
            
            # Build metadata
            metadata = {
                "model_version": entry.model_version,
                "image_mtime": entry.image_mtime,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None
            }
            
            results[key_str] = (entry.data_blob, metadata)
            
            # Update last_accessed
            entry.last_accessed = datetime.utcnow()
        
        # Delete invalid entries
        for entry in to_delete:
            self._session.delete(entry)
        
        if to_delete:
            self._session.commit()
        
        return results
    
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
