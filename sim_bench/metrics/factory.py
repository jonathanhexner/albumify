"""
Proper factory for metrics creation.
Encapsulates all metric creation logic and hides concrete implementations.
"""

import numpy as np
from typing import Dict, List, Set, Any, Union
from .base import BaseMetric, create_relevance_sets_from_groups


class MetricFactory:
    """
    Factory for creating and managing metrics.
    Provides a clean interface without exposing concrete metric classes.
    """
    
    @staticmethod
    def create_metric(metric_name: str, metric_config: Dict[str, Any] = None) -> BaseMetric:
        """
        Create a metric instance by name.
        
        Args:
            metric_name: Name of the metric to create
            metric_config: Dictionary containing metric-specific configuration
            
        Returns:
            Metric instance
            
        Raises:
            ValueError: If metric name is not recognized
        """
        if metric_config is None:
            metric_config = {}
        
        if metric_name.startswith('recall@'):
            from .recall import RecallAtK
            return RecallAtK(metric_name=metric_name, metric_config=metric_config)
        
        elif metric_name.startswith('precision@'):
            from .precision import PrecisionAtK
            return PrecisionAtK(metric_name=metric_name, metric_config=metric_config)
        
        elif metric_name.startswith('map@'):
            from .average_precision import MeanAveragePrecision
            return MeanAveragePrecision(metric_name=metric_name, metric_config=metric_config)
        
        elif metric_name in ['map', 'map_full']:
            from .average_precision import MeanAveragePrecision
            return MeanAveragePrecision(metric_name=metric_name, metric_config=metric_config)
        
        elif metric_name in ['ns', 'ns_score']:
            from .normalized_score import NormalizedScore
            return NormalizedScore(metric_name=metric_name, metric_config=metric_config)
        
        elif metric_name == 'accuracy':
            from .accuracy import Accuracy
            return Accuracy(metric_name=metric_name, metric_config=metric_config)
        
        else:
            available = ['recall@k', 'precision@k', 'map[@k]', 'ns_score', 'accuracy']
            raise ValueError(f"Unknown metric: {metric_name}. Available: {', '.join(available)}")
    
    @staticmethod
    def get_available_metrics() -> List[str]:
        """
        Get list of all available metric types.
        
        Returns:
            List of available metric names/patterns
        """
        return [
            'recall@k',      # e.g., recall@1, recall@4, recall@10
            'precision@k',   # e.g., precision@10, precision@50
            'map',           # Mean Average Precision (full)
            'map@k',         # e.g., map@10, map@50
            'ns_score',      # Normalized Score
            'accuracy'       # Accuracy (same as Recall@1)
        ]
    
    @staticmethod
    def compute_all_metrics(ranking_indices: np.ndarray, evaluation_data: Dict[str, Any], 
                           config: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all requested metrics using the factory.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            evaluation_data: Dataset evaluation data (groups, relevance_map, etc.)
            config: Configuration specifying which metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert evaluation data to universal relevance sets
        if 'groups' not in evaluation_data:
            raise ValueError("Evaluation data must contain 'groups'")
        
        relevance_sets = create_relevance_sets_from_groups(evaluation_data['groups'])
        
        results = {}
        requested_metrics = config.get('metrics', [])
        
        # Extract metric-specific configuration
        metric_config = MetricFactory._extract_metric_config(config)
        
        # Create and compute each requested metric using factory
        for metric_name in requested_metrics:
            try:
                metric = MetricFactory.create_metric(metric_name, metric_config)
                value = metric.compute(ranking_indices, relevance_sets)
                results[metric_name] = value
            except Exception as e:
                print(f"Warning: Failed to compute {metric_name}: {e}")
                continue
        
        # Add metadata
        results['num_queries'] = len(relevance_sets)
        if 'total_images' in evaluation_data:
            results['num_images'] = evaluation_data['total_images']
        
        return results
    
    @staticmethod
    def _extract_metric_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metric-specific configuration from the full config.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Dictionary containing only metric-relevant parameters
        """
        # Define which config keys are relevant for metrics
        metric_relevant_keys = {
            'k',  # For N-S score and other k-based metrics
            'threshold',  # For threshold-based metrics (if any)
            'weights',  # For weighted metrics (if any)
        }
        
        return {key: value for key, value in config.items() 
                if key in metric_relevant_keys}
    
    @staticmethod
    def describe_metric(metric_name: str) -> str:
        """
        Get description of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Human-readable description of the metric
        """
        try:
            metric = MetricFactory.create_metric(metric_name, {})
            return metric.description
        except ValueError:
            return f"Unknown metric: {metric_name}"
