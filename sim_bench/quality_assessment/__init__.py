"""
Image Quality Assessment module for selecting best images from series.
Supports rule-based, CNN, and transformer-based methods.
"""

# Import base classes
from sim_bench.quality_assessment.base import QualityAssessor

# Import registry (must be imported before methods to enable decorator)
from sim_bench.quality_assessment.registry import (
    QualityMethodRegistry,
    register_method,
    create_quality_assessor
)

# Import all methods to trigger registration
# (The @register_method decorator runs on import)
from sim_bench.quality_assessment.rule_based import RuleBasedQuality
from sim_bench.quality_assessment.clip_aesthetic import CLIPAestheticAssessor

# Import CLIP attribute-specific methods (7 separate methods, one per attribute)
try:
    import sim_bench.quality_assessment.clip_attribute_methods
except ImportError:
    pass  # CLIP dependencies not available

# Optional methods (only import if dependencies available)
try:
    from sim_bench.quality_assessment.cnn_methods import NIMAQuality
except ImportError:
    pass  # PyTorch not available

try:
    from sim_bench.quality_assessment.transformer_methods import ViTQuality
except ImportError:
    pass  # Transformers not available

try:
    from sim_bench.quality_assessment.musiq import MUSIQQuality
except ImportError:
    pass  # PyTorch not available

# Evaluators and benchmarks
from sim_bench.quality_assessment.evaluator import QualityEvaluator
from sim_bench.quality_assessment.benchmark import QualityBenchmark
from sim_bench.quality_assessment.pairwise_evaluator import (
    PairwiseEvaluator,
    compare_methods_pairwise
)
from sim_bench.quality_assessment.pairwise_benchmark import (
    PairwiseBenchmark,
    run_pairwise_benchmark_from_config
)

# Export main API
__all__ = [
    # Base classes
    'QualityAssessor',

    # Registry
    'QualityMethodRegistry',
    'register_method',
    'create_quality_assessor',

    # Methods
    'RuleBasedQuality',
    'CLIPAestheticAssessor',

    # Evaluators
    'QualityEvaluator',
    'QualityBenchmark',
    'PairwiseEvaluator',
    'compare_methods_pairwise',
    'PairwiseBenchmark',
    'run_pairwise_benchmark_from_config',
]
