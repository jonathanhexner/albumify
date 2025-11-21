"""
Abstract base class for image processing operations.

Provides common interface for image processors like thumbnail generation,
enhancement, cropping, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class ImageProcessor(ABC):
    """
    Abstract base class for image processors.

    All image processing operations (thumbnails, enhancement, cropping, etc.)
    inherit from this class to ensure consistent API.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cache: bool = True
    ):
        """
        Initialize image processor.

        Args:
            cache_dir: Directory for caching processed images
            enable_cache: Whether to cache processed images
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_cache = enable_cache

        if self.enable_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Image processor cache enabled: {self.cache_dir}")

    @abstractmethod
    def process_image(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> Union[Path, Dict[str, Path]]:
        """
        Process single image.

        Args:
            image_path: Path to input image
            **kwargs: Processor-specific parameters

        Returns:
            Path to processed image or dict of paths (for multi-output processors)
        """
        pass

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        num_workers: int = 4,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Union[Path, Dict[str, Path]]]:
        """
        Process batch of images (can be overridden for parallel processing).

        Args:
            image_paths: List of input image paths
            num_workers: Number of parallel workers
            verbose: Whether to show progress
            **kwargs: Processor-specific parameters

        Returns:
            Dict mapping input paths to output paths
        """
        results = {}

        for i, image_path in enumerate(image_paths):
            if verbose and i % 100 == 0:
                logger.info(f"Processing image {i+1}/{len(image_paths)}")

            try:
                result = self.process_image(image_path, **kwargs)
                results[str(image_path)] = result
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results[str(image_path)] = None

        return results

    def clear_cache(self) -> int:
        """
        Clear cache directory.

        Returns:
            Number of files deleted
        """
        if not self.enable_cache or not self.cache_dir:
            logger.warning("Cache not enabled or cache_dir not set")
            return 0

        if not self.cache_dir.exists():
            return 0

        count = 0
        for file in self.cache_dir.rglob('*'):
            if file.is_file():
                file.unlink()
                count += 1

        logger.info(f"Cleared {count} files from cache: {self.cache_dir}")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache information
        """
        if not self.enable_cache or not self.cache_dir:
            return {
                'enabled': False,
                'cache_dir': None,
                'file_count': 0,
                'total_size_mb': 0
            }

        if not self.cache_dir.exists():
            return {
                'enabled': True,
                'cache_dir': str(self.cache_dir),
                'file_count': 0,
                'total_size_mb': 0
            }

        files = list(self.cache_dir.rglob('*'))
        file_count = sum(1 for f in files if f.is_file())
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'file_count': file_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

    def _compute_hash(self, image_path: Union[str, Path]) -> str:
        """
        Compute content hash for image (for cache key).

        Args:
            image_path: Path to image

        Returns:
            Hash string (MD5 of file content)
        """
        import hashlib

        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cache_dir={self.cache_dir}, "
            f"enable_cache={self.enable_cache})"
        )
