"""
Abstract base class for image quality assessment methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import os


class QualityAssessor(ABC):
    """Base class for all quality assessment methods."""

    def __init__(self, device: str = 'cpu', enable_cache: bool = True):
        """
        Initialize quality assessor.

        Args:
            device: Device to run on ('cpu' or 'cuda')
            enable_cache: Whether to cache quality scores for images
        """
        self.device = device
        self.name = self.__class__.__name__
        self.enable_cache = enable_cache
        self._score_cache = {}  # Cache: image_path -> quality_score

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if this method's dependencies are available.

        Returns:
            True if the method can be instantiated, False otherwise
        """
        return True  # Default: always available (override for methods with dependencies)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'QualityAssessor':
        """
        Create instance from configuration dictionary.

        Args:
            config: Configuration parameters (method pulls what it needs)

        Returns:
            Configured QualityAssessor instance
        """
        # Default implementation: pass all config as kwargs
        # Override in subclasses if custom logic needed
        return cls(**config)
        
    @abstractmethod
    def assess_image(self, image_path: str) -> float:
        """
        Assess quality of a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Quality score (higher is better)
        """
        pass
    
    def assess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Assess quality of multiple images.
        Uses caching to avoid re-processing the same image.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of quality scores
        """
        scores = []
        for img_path in image_paths:
            # Normalize path for cache key
            cache_key = os.path.abspath(img_path)
            
            # Check cache
            if self.enable_cache and cache_key in self._score_cache:
                score = self._score_cache[cache_key]
            else:
                # Compute score
                score = self.assess_image(img_path)
                # Store in cache
                if self.enable_cache:
                    self._score_cache[cache_key] = score
            
            scores.append(score)
        return np.array(scores)
    
    def clear_cache(self):
        """Clear the quality score cache."""
        self._score_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._score_cache),
            'cache_enabled': self.enable_cache
        }
    
    def select_best_from_series(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Select best image from a series.
        
        Args:
            image_paths: List of image paths in series
            
        Returns:
            Dictionary with:
                - best_idx: Index of best image
                - best_path: Path to best image
                - scores: All quality scores
        """
        scores = self.assess_batch(image_paths)
        best_idx = int(np.argmax(scores))
        
        return {
            'best_idx': best_idx,
            'best_path': image_paths[best_idx],
            'best_score': float(scores[best_idx]),
            'scores': scores.tolist()
        }
    
    def rank_series(self, image_paths: List[str]) -> List[int]:
        """
        Rank images in series by quality (best to worst).
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of indices sorted by quality (descending)
        """
        scores = self.assess_batch(image_paths)
        return np.argsort(scores)[::-1].tolist()
    
    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            'method': self.name,
            'device': self.device
        }


