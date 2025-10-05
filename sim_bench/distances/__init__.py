"""
Distance computation package for sim-bench.
Provides various distance/similarity measures for feature comparison.
"""

from typing import Dict, Any
from .base import DistanceMeasure
from .chi_square import ChiSquareMeasure
from .cosine import CosineMeasure
from .euclidean import EuclideanMeasure
from .wasserstein import WassersteinMeasure

# Factory function for creating distance measures
def create_distance_strategy(method_config: Dict[str, Any]) -> DistanceMeasure:
    """
    Factory function for distance measures.
    
    Args:
        method_config: Full method configuration dictionary containing distance type and parameters
        
    Returns:
        Instantiated distance measure
        
    Raises:
        ValueError: If distance measure is not recognized
    """
    # Distance measure registry mapping names to classes
    measure_registry = {
        'chi_square': ChiSquareMeasure,
        'chi2': ChiSquareMeasure,
        'cosine': CosineMeasure,
        'euclidean': EuclideanMeasure,
        'wasserstein': WassersteinMeasure,
        'emd': WassersteinMeasure,
    }
    
    # Extract distance type from config
    measure_name = method_config.get('distance', 'cosine')
    
    # Look up measure class and instantiate
    measure_class = measure_registry.get(measure_name)
    if measure_class is None:
        available = ', '.join(sorted(set(measure_registry.keys())))
        raise ValueError(f"Unknown distance measure: {measure_name}. Available: {available}")
    
    return measure_class(method_config)
