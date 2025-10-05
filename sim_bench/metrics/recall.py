"""
Recall@k metric implementation.
"""

import numpy as np
from typing import Dict, List, Set, Any
from .base import BaseMetric


class RecallAtK(BaseMetric):
    """Recall@k - fraction of queries with at least one relevant result in top-k."""
    
    def __init__(self, metric_name: str = None, metric_config: Dict[str, Any] = None, **kwargs):
        """
        Initialize Recall@k metric.
        
        Args:
            metric_name: Name of the metric (for consistent interface)
            metric_config: Dictionary containing metric-specific configuration
            **kwargs: Additional parameters (for backward compatibility)
        """
        super().__init__(metric_name=metric_name, metric_config=metric_config, **kwargs)
        
        # Extract k from metric_name
        if metric_name and metric_name.startswith('recall@'):
            self.k = int(metric_name.split('@')[1])
        else:
            self.k = kwargs.get('k', 1)  # Fallback for backward compatibility
    
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> float:
        """
        Compute Recall@k.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        hits = 0
        n_queries = len(relevance_sets)
        
        for query_idx in range(n_queries):
            relevant_images = relevance_sets[query_idx]
            if not relevant_images:
                continue
                
            # Check top-k results (excluding self at rank 0)
            topk_results = set(ranking_indices[query_idx][1:self.k+1])
            if topk_results & relevant_images:  # Intersection
                hits += 1
        
        return hits / n_queries if n_queries > 0 else 0.0
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return f"recall@{self.k}"
    
    @property
    def description(self) -> str:
        """Get metric description."""
        return f"Fraction of queries with at least one relevant result in top-{self.k}"
