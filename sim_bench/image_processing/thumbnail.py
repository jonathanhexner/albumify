"""
Multi-resolution thumbnail generation with caching.

Generates image pyramids at multiple resolutions for efficient processing:
- tiny (128px): Fast CLIP tagging and routing
- small (512px): UI previews
- medium (1024px): Quality assessment
- large (2048px): Specialized models (face detection, etc.)
"""

from typing import List, Dict, Optional, Union, Literal
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from sim_bench.image_processing.base import ImageProcessor
from sim_bench.config import get_global_config


logger = logging.getLogger(__name__)


SizeType = Literal['tiny', 'small', 'medium', 'large']


class ThumbnailGenerator(ImageProcessor):
    """
    Multi-resolution thumbnail generator with disk caching.

    Generates and caches thumbnails at multiple predefined sizes.
    Uses content hashing to detect when original images change.

    Example:
        >>> from sim_bench.image_processing import ThumbnailGenerator
        >>>
        >>> generator = ThumbnailGenerator(cache_dir=".cache/thumbnails")
        >>> thumbnails = generator.generate("photo.jpg", sizes=['tiny', 'small'])
        >>> # Returns: {'tiny': 'path/to/tiny.jpg', 'small': 'path/to/small.jpg'}
        >>>
        >>> # Process entire directory
        >>> results = generator.process_batch(
        >>>     image_paths=["photo1.jpg", "photo2.jpg"],
        >>>     sizes=['tiny', 'small'],
        >>>     num_workers=4
        >>> )
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cache: bool = True,
        quality: int = 90,
        format: str = 'jpg'
    ):
        """
        Initialize thumbnail generator.

        Args:
            cache_dir: Directory for caching thumbnails (default from global config)
            enable_cache: Whether to cache generated thumbnails
            quality: JPEG quality (1-100, default from global config)
            format: Output format ('jpg' or 'png', default from global config)
        """
        # Load config
        config = get_global_config()

        # Set cache directory
        if cache_dir is None:
            cache_base = config.get_path('cache_dir', '.cache')
            cache_dir = cache_base / 'thumbnails'

        super().__init__(cache_dir=cache_dir, enable_cache=enable_cache)

        # Load thumbnail sizes from config
        self.sizes = config.get('thumbnail_sizes', {
            'tiny': 128,
            'small': 512,
            'medium': 1024,
            'large': 2048
        })

        # Load quality and format from config
        self.quality = config.get_int('thumbnail_quality', quality)
        self.format = config.get('thumbnail_format', format).lower()

        # Validate format
        if self.format not in ['jpg', 'png']:
            logger.warning(f"Invalid format '{self.format}', using 'jpg'")
            self.format = 'jpg'

        logger.info(
            f"ThumbnailGenerator initialized: sizes={list(self.sizes.keys())}, "
            f"quality={self.quality}, format={self.format}"
        )

    def process_image(
        self,
        image_path: Union[str, Path],
        sizes: Optional[List[SizeType]] = None
    ) -> Dict[str, Path]:
        """
        Generate thumbnails for single image.

        Args:
            image_path: Path to input image
            sizes: List of sizes to generate (default: all sizes)

        Returns:
            Dict mapping size names to thumbnail paths
            Example: {'tiny': Path('...'), 'small': Path('...')}
        """
        return self.generate(image_path, sizes=sizes)

    def generate(
        self,
        image_path: Union[str, Path],
        sizes: Optional[List[SizeType]] = None
    ) -> Dict[str, Path]:
        """
        Generate thumbnails at specified sizes.

        Args:
            image_path: Path to input image
            sizes: List of size names ('tiny', 'small', 'medium', 'large')
                  If None, generates all sizes

        Returns:
            Dict mapping size names to thumbnail paths

        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If invalid size name provided
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Default to all sizes
        if sizes is None:
            sizes = list(self.sizes.keys())

        # Validate sizes
        for size in sizes:
            if size not in self.sizes:
                raise ValueError(
                    f"Invalid size '{size}'. Must be one of: {list(self.sizes.keys())}"
                )

        # Compute hash for cache key
        image_hash = self._compute_hash(image_path)

        results = {}

        for size_name in sizes:
            # Check cache first
            cached_path = self._get_cached_thumbnail_path(image_hash, size_name)

            if self.enable_cache and cached_path.exists():
                logger.debug(f"Using cached thumbnail: {size_name} for {image_path.name}")
                results[size_name] = cached_path
                continue

            # Generate thumbnail
            try:
                thumbnail_path = self._generate_thumbnail(
                    image_path,
                    size_name,
                    image_hash
                )
                results[size_name] = thumbnail_path
                logger.debug(f"Generated thumbnail: {size_name} for {image_path.name}")

            except Exception as e:
                logger.error(f"Failed to generate {size_name} thumbnail for {image_path}: {e}")
                raise

        return results

    def _generate_thumbnail(
        self,
        image_path: Path,
        size_name: str,
        image_hash: str
    ) -> Path:
        """
        Generate single thumbnail.

        Args:
            image_path: Original image path
            size_name: Size name ('tiny', 'small', etc.)
            image_hash: Hash of original image (for cache key)

        Returns:
            Path to generated thumbnail
        """
        max_size = self.sizes[size_name]

        # Open and resize image
        with Image.open(image_path) as img:
            # Apply EXIF orientation if present (fixes rotation issues)
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
            
            # Convert RGBA to RGB if saving as JPEG
            if self.format == 'jpg' and img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img

            # Resize using high-quality Lanczos resampling
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Determine output path
            output_path = self._get_cached_thumbnail_path(image_hash, size_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save thumbnail
            save_kwargs = {'quality': self.quality} if self.format == 'jpg' else {}
            img.save(output_path, **save_kwargs)

        return output_path

    def _get_cached_thumbnail_path(self, image_hash: str, size_name: str) -> Path:
        """
        Get path where cached thumbnail would be stored.

        Args:
            image_hash: Hash of original image
            size_name: Size name

        Returns:
            Path to cached thumbnail
        """
        extension = 'jpg' if self.format == 'jpg' else 'png'
        return self.cache_dir / size_name / f"{image_hash}.{extension}"

    def get_thumbnail_path(
        self,
        image_path: Union[str, Path],
        size: SizeType = 'tiny'
    ) -> Optional[Path]:
        """
        Get path to cached thumbnail (if exists).

        Args:
            image_path: Original image path
            size: Size name

        Returns:
            Path to cached thumbnail, or None if not cached
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return None

        image_hash = self._compute_hash(image_path)
        cached_path = self._get_cached_thumbnail_path(image_hash, size)

        return cached_path if cached_path.exists() else None

    def get_batch_paths(
        self,
        image_paths: List[Union[str, Path]],
        size: SizeType = 'tiny',
        generate_missing: bool = False
    ) -> List[Optional[Path]]:
        """
        Get paths to thumbnails for batch of images.

        Args:
            image_paths: List of original image paths
            size: Size name
            generate_missing: If True, generate thumbnails that don't exist

        Returns:
            List of thumbnail paths (None for missing thumbnails if not generated)
        """
        results = []

        for image_path in image_paths:
            cached_path = self.get_thumbnail_path(image_path, size)

            if cached_path is None and generate_missing:
                try:
                    thumbnails = self.generate(image_path, sizes=[size])
                    cached_path = thumbnails[size]
                except Exception as e:
                    logger.error(f"Failed to generate thumbnail for {image_path}: {e}")
                    cached_path = None

            results.append(cached_path)

        return results

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        sizes: Optional[List[SizeType]] = None,
        num_workers: int = 4,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Path]]:
        """
        Generate thumbnails for batch of images in parallel.

        Args:
            image_paths: List of input image paths
            sizes: List of sizes to generate (default: all)
            num_workers: Number of parallel workers
            verbose: Whether to show progress

        Returns:
            Dict mapping original paths to size->thumbnail_path dicts
            Example: {
                'photo1.jpg': {'tiny': Path('...'), 'small': Path('...')},
                'photo2.jpg': {'tiny': Path('...'), 'small': Path('...')}
            }
        """
        results = {}

        if num_workers <= 1:
            # Sequential processing
            for i, image_path in enumerate(image_paths):
                if verbose and i % 100 == 0:
                    logger.info(f"Processing thumbnail {i+1}/{len(image_paths)}")

                try:
                    results[str(image_path)] = self.generate(image_path, sizes=sizes)
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    results[str(image_path)] = {}

        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.generate, img_path, sizes): img_path
                    for img_path in image_paths
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_path):
                    img_path = future_to_path[future]
                    completed += 1

                    if verbose and completed % 100 == 0:
                        logger.info(f"Processing thumbnail {completed}/{len(image_paths)}")

                    try:
                        results[str(img_path)] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to process {img_path}: {e}")
                        results[str(img_path)] = {}

        if verbose:
            logger.info(f"Generated thumbnails for {len(results)} images")

        return results

    def clear_cache(self, size: Optional[SizeType] = None) -> int:
        """
        Clear thumbnail cache.

        Args:
            size: If provided, only clear specific size. Otherwise clear all.

        Returns:
            Number of files deleted
        """
        if not self.enable_cache or not self.cache_dir:
            logger.warning("Cache not enabled or cache_dir not set")
            return 0

        if size:
            # Clear specific size
            size_dir = self.cache_dir / size
            if not size_dir.exists():
                return 0

            count = 0
            for file in size_dir.glob('*'):
                if file.is_file():
                    file.unlink()
                    count += 1

            logger.info(f"Cleared {count} {size} thumbnails from cache")
            return count
        else:
            # Clear all sizes
            return super().clear_cache()

    def __repr__(self) -> str:
        return (
            f"ThumbnailGenerator("
            f"cache_dir={self.cache_dir}, "
            f"sizes={list(self.sizes.keys())}, "
            f"quality={self.quality}, "
            f"format={self.format})"
        )
