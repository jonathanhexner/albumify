"""
Cosine distance measure.
"""

import numpy as np
from typing import Dict, Any
from .base import DistanceMeasure


class CosineMeasure(DistanceMeasure):
    """Cosine distance measure (1 - cosine_similarity)."""
    
    def __init__(self, method_config: Dict[str, Any]):
        super().__init__(method_config)
    
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """Compute cosine distances between feature vectors."""
        # L2 normalize both feature matrices
        features_a_normalized = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-12)
        features_b_normalized = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-12)
        
        # Cosine similarity then convert to distance
        cosine_similarity = features_a_normalized @ features_b_normalized.T
        return 1.0 - cosine_similarity
