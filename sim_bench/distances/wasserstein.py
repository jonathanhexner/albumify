"""
Wasserstein (Earth Mover's Distance) measure for histograms.
"""

import numpy as np
from typing import Dict, Any
from .base import DistanceMeasure


class WassersteinMeasure(DistanceMeasure):
    """Earth Mover's Distance (Wasserstein) measure for histograms.
    
    Note: This is a simplified implementation. For true multi-dimensional EMD,
    you would need proper ground distance matrices and optimization solvers.
    """
    
    def __init__(self, method_config: Dict[str, Any]):
        """
        Initialize Wasserstein strategy.
        
        Args:
            method_config: Method configuration dictionary
        """
        super().__init__(method_config)
        
        # Extract histogram shape from config
        self.histogram_shape = None
        if 'features' in method_config and 'bins' in method_config['features']:
            self.histogram_shape = tuple(method_config['features']['bins'])
        
        self.ground_metric = method_config.get('ground_metric', 'euclidean')
        
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """
        Compute simplified Wasserstein distances between histograms.
        
        For multi-dimensional histograms (like HSV), this treats them as 1D
        by using bin indices as coordinates. This is a simplification but
        works reasonably well in practice.
        """
        from scipy.stats import wasserstein_distance
        
        num_samples_a, feature_dim = features_a.shape
        num_samples_b, _ = features_b.shape
        
        # Validate histogram shape if provided
        if self.histogram_shape is not None and np.prod(self.histogram_shape) != feature_dim:
            raise ValueError(f"Histogram shape {self.histogram_shape} doesn't match feature dimension {feature_dim}")
        
        # Create 1D coordinate array (bin indices)
        bin_coordinates = np.arange(feature_dim, dtype=np.float32)
        
        # Compute pairwise Wasserstein distances
        distance_matrix = np.empty((num_samples_a, num_samples_b), dtype=np.float32)
        for i in range(num_samples_a):
            for j in range(num_samples_b):
                distance_matrix[i, j] = wasserstein_distance(
                    bin_coordinates, bin_coordinates, 
                    features_a[i], features_b[j]
                )
        
        return distance_matrix
