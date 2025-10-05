"""
Mean Average Precision (mAP) metric implementation.
"""

import numpy as np
from typing import Dict, List, Set, Optional, Any
from .base import BaseMetric


class MeanAveragePrecision(BaseMetric):
    """Mean Average Precision (mAP) metric."""
    
    def __init__(self, metric_name: str = None, metric_config: Dict[str, Any] = None, **kwargs):
        """
        Initialize mAP metric.
        
        Args:
            metric_name: Name of the metric (for consistent interface)
            metric_config: Dictionary containing metric-specific configuration
            **kwargs: Additional parameters (for backward compatibility)
        """
        super().__init__(metric_name=metric_name, metric_config=metric_config, **kwargs)
        
        # Extract k from metric_name
        if metric_name and metric_name.startswith('map@'):
            self.k = int(metric_name.split('@')[1])
        elif metric_name in ['map', 'map_full']:
            self.k = None
        else:
            self.k = kwargs.get('k', None)  # Fallback for backward compatibility
    
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> float:
        """
        Compute Mean Average Precision.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Mean Average Precision score (0.0 to 1.0)
        """
        ap_scores = self._compute_average_precision_per_query(ranking_indices, relevance_sets)
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def _compute_average_precision_per_query(self, ranking_indices: np.ndarray, 
                                           relevance_sets: List[Set[int]]) -> List[float]:
        """
        Compute Average Precision for each query.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            List of AP scores for each query
        """
        ap_scores = []
        
        for query_idx in range(len(relevance_sets)):
            relevant_images = relevance_sets[query_idx]
            if not relevant_images:
                ap_scores.append(0.0)
                continue
                
            # Determine evaluation depth
            eval_k = self.k if self.k is not None else len(relevant_images)
            eval_k = min(eval_k, len(ranking_indices[query_idx]) - 1)  # -1 to skip self
            
            # Calculate AP for this query
            relevant_count = 0
            precision_sum = 0.0
            
            for rank, result_idx in enumerate(ranking_indices[query_idx][1:eval_k+1], start=1):
                if result_idx in relevant_images:
                    relevant_count += 1
                    precision_sum += relevant_count / rank
            
            # Normalize by number of relevant images
            ap = precision_sum / len(relevant_images) if relevant_images else 0.0
            ap_scores.append(ap)
        
        return ap_scores
    
    @property
    def name(self) -> str:
        """Get metric name."""
        if self.k is None:
            return "map"
        else:
            return f"map@{self.k}"
    
    @property
    def description(self) -> str:
        """Get metric description."""
        if self.k is None:
            return "Mean Average Precision using all relevant images"
        else:
            return f"Mean Average Precision at {self.k}"
