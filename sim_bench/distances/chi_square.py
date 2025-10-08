"""
Chi-square distance measure for histograms.
"""

import numpy as np
from typing import Dict, Any
from tqdm import tqdm
from .base import DistanceMeasure


class ChiSquareMeasure(DistanceMeasure):
    """Chi-square distance measure for histograms."""
    
    def __init__(self, method_config: Dict[str, Any]):
        super().__init__(method_config)
        self.eps = method_config.get('eps', 1e-10)
    
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """Compute chi-square distances between histograms."""
        num_samples_a, feature_dim = features_a.shape
        num_samples_b, _ = features_b.shape
        distance_matrix = np.empty((num_samples_a, num_samples_b), dtype=np.float32)
        
        # Chunked computation for memory efficiency
        chunk_size = max(1, 1024 // max(1, feature_dim // 256))
        num_chunks = (num_samples_a + chunk_size - 1) // chunk_size
        
        for i in tqdm(range(0, num_samples_a, chunk_size), 
                     total=num_chunks, 
                     desc=f"Chi-square distances ({num_samples_a} images, {num_chunks} chunks)", 
                     unit="chunk"):
            chunk_a = features_a[i:i+chunk_size][:, None, :]  # [chunk_size, 1, feature_dim]
            chunk_b = features_b[None, :, :]                  # [1, num_samples_b, feature_dim]
            numerator = (chunk_a - chunk_b) ** 2
            denominator = (chunk_a + chunk_b + self.eps)
            chi_square_dist = 0.5 * np.sum(numerator / denominator, axis=2)  # [chunk_size, num_samples_b]
            distance_matrix[i:i+chunk_size] = chi_square_dist
        
        return distance_matrix
