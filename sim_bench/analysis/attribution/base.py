"""
Base classes for feature attribution extractors.

Defines the interface that all attribution methods should implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


class BaseAttributionExtractor(ABC):
    """
    Abstract base class for feature attribution extractors.
    
    All attribution methods should inherit from this and implement
    the required methods.
    """
    
    @abstractmethod
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from an image (same as used in benchmarking).
        
        Args:
            image_path: Path to image
            
        Returns:
            Feature vector
        """
        pass
    
    @abstractmethod
    def compute_attribution(
        self, 
        image_path: str,
        feature_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attribution map showing which image regions contribute to features.
        
        Args:
            image_path: Path to image
            feature_indices: Specific feature dimensions to visualize (None = all)
            
        Returns:
            Tuple of (attribution_map, original_image)
            attribution_map: [H, W] array with values in [0, 1]
            original_image: [H, W, 3] RGB image array
        """
        pass
    
    @abstractmethod
    def analyze_feature_importance(
        self,
        image_path: str,
        top_k: int = 10
    ) -> dict:
        """
        Analyze which feature dimensions are most important for an image.
        
        Args:
            image_path: Path to image
            top_k: Number of top features to return
            
        Returns:
            Dictionary with feature analysis results
        """
        pass

