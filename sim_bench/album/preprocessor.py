"""
Image preprocessing for efficient album analysis.

Generates thumbnails at appropriate resolutions for different analysis operations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

from sim_bench.image_processing import ThumbnailGenerator

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Generates thumbnails for efficient analysis.
    
    Different operations need different resolutions:
    - IQA: 1024px (medium) - quality assessment
    - Portrait: 2048px (large) - face detection needs detail
    - Features: 1024px (medium) - embeddings resize anyway
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self._config = config
        cache_dir = Path(config.get('cache_dir', '.cache')) / 'album_analysis'
        self._gen = ThumbnailGenerator(cache_dir=cache_dir)
        
        # Map operation types to thumbnail sizes
        self._size_map = {
            'quality': 'medium',   # 1024px for IQA
            'portrait': 'large',   # 2048px for face detection
            'features': 'medium',  # 1024px for embeddings
        }
        
        logger.info(f"ImagePreprocessor initialized (cache: {cache_dir})")
    
    def preprocess_batch(
        self,
        image_paths: List[Path],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[Path, Dict[str, Path]]:
        """
        Generate thumbnails for all images.
        
        Args:
            image_paths: List of original image paths
            progress_callback: Optional callback(stage, progress)
        
        Returns:
            Dictionary mapping original_path to operation-specific thumbnails:
            {
                Path('img1.jpg'): {
                    'quality': Path('.cache/.../medium/hash.jpg'),
                    'portrait': Path('.cache/.../large/hash.jpg'),
                    'features': Path('.cache/.../medium/hash.jpg')
                }
            }
        """
        if progress_callback:
            progress_callback("preprocess", 0.0)
        
        # Get unique sizes needed
        sizes = set(self._size_map.values())
        
        # Generate thumbnails in parallel
        num_workers = self._config.get('album', {}).get('preprocessing', {}).get('num_workers', 4)
        
        logger.info(f"Generating thumbnails for {len(image_paths)} images (sizes: {sizes})")
        
        thumbnails = self._gen.process_batch(
            image_paths,
            sizes=list(sizes),
            num_workers=num_workers,
            verbose=False
        )
        
        # Map thumbnails to operation types
        result = {}
        for orig_path, thumbs in thumbnails.items():
            result[Path(orig_path)] = {
                op: thumbs[size] for op, size in self._size_map.items()
            }
        
        if progress_callback:
            progress_callback("preprocess", 1.0)
        
        cache_hits = sum(1 for t in thumbnails.values() if t)
        logger.info(f"Preprocessing complete: {len(result)} images, {cache_hits} cache hits")
        
        return result
    
    def get_operation_path(
        self,
        original_path: Path,
        operation: str,
        thumbnails: Optional[Dict[Path, Dict[str, Path]]] = None
    ) -> Path:
        """
        Get appropriate image path for an operation.
        
        Args:
            original_path: Original image path
            operation: Operation type ('quality', 'portrait', 'features')
            thumbnails: Preprocessed thumbnail mapping
        
        Returns:
            Path to use (thumbnail if available, original otherwise)
        """
        if thumbnails and original_path in thumbnails:
            return thumbnails[original_path].get(operation, original_path)
        return original_path
