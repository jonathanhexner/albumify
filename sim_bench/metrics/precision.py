"""
Precision@k metric implementation.
"""

import numpy as np
from typing import Dict, List, Set, Any
from .base import BaseMetric


class PrecisionAtK(BaseMetric):
    """Precision@k - average precision in top-k results."""
    
    def __init__(self, metric_name: str = None, metric_config: Dict[str, Any] = None, **kwargs):
        """
        Initialize Precision@k metric.
        
        Args:
            metric_name: Name of the metric (for consistent interface)
            metric_config: Dictionary containing metric-specific configuration
            **kwargs: Additional parameters (for backward compatibility)
        """
        super().__init__(metric_name=metric_name, metric_config=metric_config, **kwargs)
        
        # Extract k from metric_name
        if metric_name and metric_name.startswith('precision@'):
            self.k = int(metric_name.split('@')[1])
        else:
            self.k = kwargs.get('k', 10)  # Fallback for backward compatibility
    
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> float:
        """
        Compute Precision@k.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Average Precision@k score (0.0 to 1.0)
        """
        precision_sum = 0.0
        n_queries = len(relevance_sets)
        
        for query_idx in range(n_queries):
            relevant_images = relevance_sets[query_idx]
            if not relevant_images:
                continue
                
            # Count relevant images in top-k (excluding self at rank 0)
            topk_results = ranking_indices[query_idx][1:self.k+1]
            relevant_in_topk = sum(1 for idx in topk_results if idx in relevant_images)
            
            precision_sum += relevant_in_topk / min(self.k, len(topk_results))
        
        return precision_sum / n_queries if n_queries > 0 else 0.0
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return f"precision@{self.k}"
    
    @property
    def description(self) -> str:
        """Get metric description."""
        return f"Average precision in top-{self.k} results"
