"""
Abstract base class for distance measures.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class DistanceMeasure(ABC):
    """Abstract base class for computing distances between feature vectors."""
    
    def __init__(self, method_config: Dict[str, Any]):
        """
        Initialize strategy with method configuration.
        
        Args:
            method_config: Method configuration dictionary
        """
        self.method_config = method_config
    
    @abstractmethod
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between two feature matrices.
        
        Args:
            features_a: Feature matrix [n_samples_a, n_features]
            features_b: Feature matrix [n_samples_b, n_features]
            
        Returns:
            Distance matrix [n_samples_a, n_samples_b]
        """
        pass
