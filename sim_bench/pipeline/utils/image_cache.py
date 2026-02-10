"""Global image cache with EXIF normalization.

This module provides a global cache for EXIF-normalized images. Images are
stored once and shared across all albums and pipeline runs.

Cache key strategy (EXIF-first with partial hash fallback):
1. If image has EXIF DateTimeOriginal: SHA256(DateTimeOriginal + Make + Model + FileSize)
2. Otherwise: SHA256(first_64KB + last_64KB + FileSize)
"""

import hashlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".sim_bench" / "image_cache"

# Chunk size for partial hash (64KB)
HASH_CHUNK_SIZE = 65536

# JPEG quality for cached images
CACHE_JPEG_QUALITY = 95


class ImageCache:
    """Global cache for EXIF-normalized images.

    Images are normalized (EXIF transposed, converted to RGB) and stored
    on disk. The cache is keyed by image content, so the same image in
    different locations results in a single cached copy.
    """

    _instance: Optional['ImageCache'] = None
    _lock = threading.Lock()

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the image cache.

        Args:
            cache_dir: Directory for cache storage. Defaults to ~/.sim_bench/image_cache/
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.images_dir = self.cache_dir / "images"
        self.index_path = self.cache_dir / "index.db"

        self._init_cache()
        self._local = threading.local()

    @classmethod
    def get_instance(cls, cache_dir: Optional[Path] = None) -> 'ImageCache':
        """Get singleton instance of ImageCache."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cache_dir)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _init_cache(self):
        """Initialize cache directory and database."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.index_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    cache_key TEXT PRIMARY KEY,
                    original_path TEXT,
                    cached_filename TEXT,
                    width INTEGER,
                    height INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_original_path
                ON image_index(original_path)
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.index_path))
        return self._local.conn

    def _compute_cache_key(self, path: Path) -> str:
        """Compute cache key for an image using EXIF-first strategy.

        Strategy:
        1. If EXIF DateTimeOriginal exists: hash(datetime + make + model + size)
        2. Otherwise: hash(first_64KB + last_64KB + size)
        """
        file_size = path.stat().st_size

        # Try EXIF-based key first
        exif_key = self._try_exif_key(path, file_size)
        if exif_key:
            return exif_key

        # Fallback to partial content hash
        return self._compute_content_key(path, file_size)

    def _try_exif_key(self, path: Path, file_size: int) -> Optional[str]:
        """Try to compute cache key from EXIF metadata."""
        try:
            with Image.open(path) as img:
                exif_data = img._getexif()
                if not exif_data:
                    return None

                # Extract relevant EXIF tags
                exif_dict = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    exif_dict[tag_name] = value

                date_time = exif_dict.get('DateTimeOriginal')
                if not date_time:
                    return None

                make = exif_dict.get('Make', '')
                model = exif_dict.get('Model', '')

                # Create key from EXIF data
                key_data = f"{date_time}|{make}|{model}|{file_size}"
                hash_value = hashlib.sha256(key_data.encode()).hexdigest()[:16]

                return f"exif_{hash_value}"

        except Exception as e:
            logger.debug(f"Could not read EXIF from {path}: {e}")
            return None

    def _compute_content_key(self, path: Path, file_size: int) -> str:
        """Compute cache key from partial file content."""
        hasher = hashlib.sha256()

        with open(path, 'rb') as f:
            # Read first chunk
            first_chunk = f.read(HASH_CHUNK_SIZE)
            hasher.update(first_chunk)

            # Read last chunk (if file is large enough)
            if file_size > HASH_CHUNK_SIZE * 2:
                f.seek(-HASH_CHUNK_SIZE, 2)  # Seek from end
                last_chunk = f.read(HASH_CHUNK_SIZE)
                hasher.update(last_chunk)

            # Include file size
            hasher.update(str(file_size).encode())

        hash_value = hasher.hexdigest()[:16]
        return f"content_{hash_value}"

    def _lookup_cache(self, cache_key: str) -> Optional[Tuple[str, int, int]]:
        """Look up cached image by key.

        Returns:
            Tuple of (cached_filename, width, height) or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT cached_filename, width, height FROM image_index WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        return row if row else None

    def _store_cache_entry(self, cache_key: str, original_path: Path,
                           cached_filename: str, width: int, height: int):
        """Store cache entry in index."""
        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO image_index
               (cache_key, original_path, cached_filename, width, height)
               VALUES (?, ?, ?, ?, ?)""",
            (cache_key, str(original_path), cached_filename, width, height)
        )
        conn.commit()

    def _normalize_and_cache(self, path: Path, cache_key: str) -> Tuple[np.ndarray, int, int]:
        """Normalize image and store in cache.

        Returns:
            Tuple of (image_array, width, height)
        """
        # Load and normalize image
        with Image.open(path) as img:
            # Apply EXIF transpose
            img = ImageOps.exif_transpose(img)

            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size

            # Save to cache
            cached_filename = f"{cache_key}.jpg"
            cached_path = self.images_dir / cached_filename
            img.save(cached_path, 'JPEG', quality=CACHE_JPEG_QUALITY)

            # Store index entry
            self._store_cache_entry(cache_key, path, cached_filename, width, height)

            logger.debug(f"Cached normalized image: {path} -> {cached_filename}")

            return np.array(img), width, height

    def get(self, path: Path) -> np.ndarray:
        """Get normalized image as numpy array.

        Args:
            path: Path to original image

        Returns:
            RGB numpy array with EXIF normalization applied
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        cache_key = self._compute_cache_key(path)

        # Check cache
        cached = self._lookup_cache(cache_key)
        if cached:
            cached_filename, width, height = cached
            cached_path = self.images_dir / cached_filename

            if cached_path.exists():
                with Image.open(cached_path) as img:
                    return np.array(img)
            else:
                logger.warning(f"Cached file missing, regenerating: {cached_path}")

        # Cache miss - normalize and cache
        img_array, _, _ = self._normalize_and_cache(path, cache_key)
        return img_array

    def get_pil(self, path: Path) -> Image.Image:
        """Get normalized image as PIL Image.

        Args:
            path: Path to original image

        Returns:
            RGB PIL Image with EXIF normalization applied
        """
        return Image.fromarray(self.get(path))

    def get_dimensions(self, path: Path) -> Tuple[int, int]:
        """Get dimensions of normalized image without loading pixels.

        Args:
            path: Path to original image

        Returns:
            Tuple of (width, height)
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        cache_key = self._compute_cache_key(path)

        # Check cache for dimensions
        cached = self._lookup_cache(cache_key)
        if cached:
            _, width, height = cached
            return width, height

        # Need to normalize to get dimensions
        _, width, height = self._normalize_and_cache(path, cache_key)
        return width, height

    def clear(self):
        """Clear entire cache."""
        import shutil

        # Close connection
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

        # Remove cache directory
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

        # Reinitialize
        self._init_cache()

        logger.info("Image cache cleared")

    def evict(self, path: Path):
        """Remove specific image from cache.

        Args:
            path: Path to original image
        """
        path = Path(path).resolve()
        cache_key = self._compute_cache_key(path)

        cached = self._lookup_cache(cache_key)
        if cached:
            cached_filename, _, _ = cached
            cached_path = self.images_dir / cached_filename

            # Remove file
            if cached_path.exists():
                cached_path.unlink()

            # Remove index entry
            conn = self._get_connection()
            conn.execute("DELETE FROM image_index WHERE cache_key = ?", (cache_key,))
            conn.commit()

            logger.debug(f"Evicted from cache: {path}")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats (count, size_mb, etc.)
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM image_index")
        count = cursor.fetchone()[0]

        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in self.images_dir.glob("*.jpg")
        )

        return {
            'count': count,
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


# Convenience function for getting the singleton instance
def get_image_cache(cache_dir: Optional[Path] = None) -> ImageCache:
    """Get the global ImageCache instance.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        ImageCache singleton instance
    """
    return ImageCache.get_instance(cache_dir)
