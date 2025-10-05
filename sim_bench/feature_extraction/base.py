"""
Abstract base class for similarity methods in sim-bench.
Uses Strategy pattern for distance computation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from sim_bench.distances import create_distance_strategy


class BaseMethod(ABC):
    """Abstract base class for all similarity methods."""
    
    def __init__(self, method_config: Dict[str, Any]):
        """
        Initialize method with configuration.
        
        Args:
            method_config: Method configuration dictionary loaded from YAML
        """
        self.method_config = method_config
        self.method_name = method_config['method']
        
        # Initialize distance strategy - pass entire config for maximum flexibility
        self.distance_strategy = create_distance_strategy(method_config)
        
    @abstractmethod
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from a list of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Feature matrix of shape [n_images, feature_dim]
        """
        pass
    
    def compute_distances(self, features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances using the configured strategy.
        
        Args:
            features: Feature matrix [n_images, feature_dim]
            
        Returns:
            Distance matrix [n_images, n_images]
        """
        return self.distance_strategy.compute_pairwise_distances(features, features)
    
    def get_cache_dir(self) -> Path:
        """Get cache directory for this method."""
        cache_dir = Path(self.method_config.get('cache_dir', f'artifacts/{self.method_name}'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.method_name})"


def load_method(method_name: str, method_config: Dict[str, Any]) -> BaseMethod:
    """
    Factory function to load a method by name.
    
    Args:
        method_name: Name of the method to load
        method_config: Method configuration dictionary
        
    Returns:
        Instantiated method object
        
    Raises:
        ValueError: If method_name is not recognized
    """
    # Import all feature extraction classes
    from sim_bench.feature_extraction.hsv_histogram import HSVHistogramMethod
    from sim_bench.feature_extraction.resnet50 import ResNet50Method
    from sim_bench.feature_extraction.sift_bovw import SIFTBoVWMethod

    # Method registry - supporting both old and new names for backward compatibility
    method_registry = {
        'chi_square': HSVHistogramMethod,    # Legacy name
        'emd': HSVHistogramMethod,           # Legacy name
        'hsv_histogram': HSVHistogramMethod, # New feature-based name
        'deep': ResNet50Method,              # Legacy name
        'resnet50': ResNet50Method,          # New specific name
        'cnn_feature': ResNet50Method,       # Generic CNN name
        'sift_bovw': SIFTBoVWMethod,
    }
    
    if method_name not in method_registry:
        available_methods = ', '.join(method_registry.keys())
        raise ValueError(f"Unknown method: {method_name}. Available: {available_methods}")
    
    method_class = method_registry[method_name]
    return method_class(method_config)
