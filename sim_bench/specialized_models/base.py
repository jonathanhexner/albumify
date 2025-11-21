"""
Abstract base class for specialized models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np


class SpecializedModel(ABC):
    """Base class for all specialized models (faces, landmarks, etc.)."""
    
    def __init__(self, device: str = 'cpu', enable_cache: bool = True):
        """
        Initialize specialized model.
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
            enable_cache: Whether to cache results
        """
        self.device = device
        self.name = self.__class__.__name__
        self.enable_cache = enable_cache
        self._result_cache = {}
        
    @abstractmethod
    def extract_embeddings(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dict mapping image_path -> embedding array
            For images with multiple detections (e.g., multiple faces),
            return array of shape [n_detections, embedding_dim]
        """
        pass
    
    @abstractmethod
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process single image and return detailed results.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with:
                - embeddings: List of embedding arrays
                - detections: List of detection info (bbox, confidence, etc.)
                - metadata: Additional information
        """
        pass
    
    def process_batch(
        self,
        image_paths: List[str],
        routing_hints: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process batch of images.
        
        Args:
            image_paths: List of image paths
            routing_hints: Optional routing decisions (only process if True)
            
        Returns:
            Dict mapping image_path -> process_image() results
        """
        results = {}
        
        for image_path in image_paths:
            # Check routing hints
            if routing_hints and not routing_hints.get(f'needs_{self._get_routing_key()}', True):
                continue
                
            # Check cache
            cache_key = str(Path(image_path).resolve())
            if self.enable_cache and cache_key in self._result_cache:
                results[image_path] = self._result_cache[cache_key]
                continue
            
            # Process image
            try:
                result = self.process_image(image_path)
                results[image_path] = result
                
                # Cache result
                if self.enable_cache:
                    self._result_cache[cache_key] = result
            except Exception as e:
                results[image_path] = {
                    'error': str(e),
                    'embeddings': [],
                    'detections': []
                }
        
        return results
    
    @abstractmethod
    def _get_routing_key(self) -> str:
        """Get routing key for this model (e.g., 'face_detection')."""
        pass
    
    def clear_cache(self):
        """Clear the result cache."""
        self._result_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._result_cache),
            'cache_enabled': self.enable_cache
        }




