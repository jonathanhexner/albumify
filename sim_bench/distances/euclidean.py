"""
Euclidean distance measure.
"""

import numpy as np
from typing import Dict, Any
from .base import DistanceMeasure


class EuclideanMeasure(DistanceMeasure):
    """Euclidean distance measure."""
    
    def __init__(self, method_config: Dict[str, Any]):
        super().__init__(method_config)
    
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between feature vectors."""
        # Using broadcasting: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        squared_norms_a = np.sum(features_a * features_a, axis=1, keepdims=True)  # [num_samples_a, 1]
        squared_norms_b = np.sum(features_b * features_b, axis=1, keepdims=True).T  # [1, num_samples_b]
        dot_product = features_a @ features_b.T  # [num_samples_a, num_samples_b]
        
        squared_distances = squared_norms_a + squared_norms_b - 2 * dot_product
        return np.maximum(squared_distances, 0.0)  # Ensure non-negative
