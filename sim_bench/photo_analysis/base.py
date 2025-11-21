"""
Abstract base class for photo analysis.

Provides common interface for photo analyzers that extract high-level
metadata from images using vision-language models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class PhotoAnalyzer(ABC):
    """
    Abstract base class for photo analyzers.

    Photo analyzers extract high-level semantic information from images,
    such as scene type, quality indicators, composition features, and
    routing decisions for specialized models.
    """

    def __init__(
        self,
        enable_cache: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize photo analyzer.

        Args:
            enable_cache: Whether to cache analysis results
            device: Device for computation ('cpu', 'cuda', etc.)
        """
        self.enable_cache = enable_cache
        self.device = device
        self._analysis_cache: Dict[str, Dict] = {}

        logger.info(f"PhotoAnalyzer initialized: device={device}, cache={enable_cache}")

    @abstractmethod
    def analyze_image(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analyze single image.

        Args:
            image_path: Path to image file

        Returns:
            Dict with analysis results:
            {
                'path': str,                  # Original image path
                'tags': Dict[str, float],     # Tag -> confidence scores
                'primary_tags': List[str],    # Top-k tags
                'importance_score': float,    # Overall importance (0-1)
                'routing': Dict[str, bool],   # Which models to apply
                'metadata': Dict[str, Any]    # Additional metadata
            }
        """
        pass

    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze batch of images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            verbose: Whether to show progress

        Returns:
            Dict mapping image paths to analysis results
        """
        results = {}

        for i, image_path in enumerate(image_paths):
            if verbose and i % 100 == 0:
                logger.info(f"Analyzing image {i+1}/{len(image_paths)}")

            # Check cache
            cache_key = str(Path(image_path).resolve())
            if self.enable_cache and cache_key in self._analysis_cache:
                results[str(image_path)] = self._analysis_cache[cache_key]
                continue

            # Analyze image
            try:
                analysis = self.analyze_image(image_path)
                results[str(image_path)] = analysis

                # Cache result
                if self.enable_cache:
                    self._analysis_cache[cache_key] = analysis

            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {e}")
                results[str(image_path)] = self._get_empty_analysis(image_path)

        return results

    def _get_empty_analysis(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Return empty analysis result (for failed analyses).

        Args:
            image_path: Image path

        Returns:
            Empty analysis dict
        """
        return {
            'path': str(image_path),
            'tags': {},
            'primary_tags': [],
            'importance_score': 0.0,
            'routing': {
                'needs_face_detection': False,
                'needs_landmark_detection': False,
                'needs_object_detection': False
            },
            'metadata': {'error': True}
        }

    def clear_cache(self) -> int:
        """
        Clear analysis cache.

        Returns:
            Number of cached items cleared
        """
        count = len(self._analysis_cache)
        self._analysis_cache.clear()
        logger.info(f"Cleared {count} items from analysis cache")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache information
        """
        return {
            'enabled': self.enable_cache,
            'cached_items': len(self._analysis_cache)
        }

    def save_results(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Union[str, Path],
        format: str = 'json'
    ) -> None:
        """
        Save analysis results to file.

        Args:
            results: Analysis results from analyze_batch()
            output_path: Output file path
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved analysis results to: {output_path}")

        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'image_path',
                    'importance_score',
                    'primary_tags',
                    'needs_face_detection',
                    'needs_landmark_detection'
                ])

                # Data
                for img_path, analysis in results.items():
                    writer.writerow([
                        img_path,
                        analysis.get('importance_score', 0.0),
                        ','.join(analysis.get('primary_tags', [])),
                        analysis.get('routing', {}).get('needs_face_detection', False),
                        analysis.get('routing', {}).get('needs_landmark_detection', False)
                    ])

            logger.info(f"Saved analysis results to: {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")

    def load_results(
        self,
        input_path: Union[str, Path],
        format: str = 'json'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load previously saved analysis results.

        Args:
            input_path: Input file path
            format: Input format ('json' or 'csv')

        Returns:
            Analysis results dict
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")

        if format == 'json':
            import json
            with open(input_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} analysis results from: {input_path}")
            return results

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json'")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"cache_enabled={self.enable_cache})"
        )
