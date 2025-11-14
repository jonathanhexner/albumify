"""
Image Quality Assessment module for selecting best images from series.
Supports rule-based, CNN, and transformer-based methods.
"""

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.rule_based import RuleBasedQuality
from sim_bench.quality_assessment.evaluator import QualityEvaluator
from sim_bench.quality_assessment.benchmark import (
    QualityBenchmark,
    load_benchmark_config,
    run_benchmark_from_config
)

# Conditional imports for methods with heavy dependencies
try:
    from sim_bench.quality_assessment.cnn_methods import NIMAQuality
    _HAS_CNN = True
except ImportError:
    _HAS_CNN = False
    NIMAQuality = None

try:
    from sim_bench.quality_assessment.transformer_methods import ViTQuality
    _HAS_TRANSFORMER = True
except ImportError:
    _HAS_TRANSFORMER = False
    ViTQuality = None

__all__ = [
    'QualityAssessor',
    'RuleBasedQuality', 
    'NIMAQuality',
    'ViTQuality',
    'QualityEvaluator',
    'QualityBenchmark',
    'load_benchmark_config',
    'run_benchmark_from_config',
    'load_quality_method'
]


def load_quality_method(method_name: str, config: dict = None):
    """
    Factory function to load quality assessment method.
    
    Args:
        method_name: Name of method ('rule_based', 'nima', 'vit', 'musiq')
        config: Method-specific configuration
        
    Returns:
        QualityAssessor instance
    """
    config = config or {}
    
    if method_name == 'rule_based':
        return RuleBasedQuality(**config)
    
    elif method_name == 'nima':
        if not _HAS_CNN:
            raise ImportError(
                "NIMA requires PyTorch. Install with: pip install torch torchvision"
            )
        return NIMAQuality(**config)
    
    elif method_name == 'vit':
        if not _HAS_TRANSFORMER:
            raise ImportError(
                "ViT requires PyTorch and Transformers. "
                "Install with: pip install torch torchvision transformers"
            )
        return ViTQuality(**config)
    
    elif method_name == 'musiq':
        if not _HAS_TRANSFORMER:
            raise ImportError(
                "MUSIQ requires PyTorch and Transformers. "
                "Install with: pip install torch torchvision transformers"
            )
        # Import here to avoid circular dependency
        from sim_bench.quality_assessment.transformer_methods import MUSIQQuality
        return MUSIQQuality(**config)
    
    else:
        raise ValueError(f"Unknown quality method: {method_name}")

