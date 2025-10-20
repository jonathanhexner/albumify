"""
DEPRECATED: This module has been reorganized.

Please use the new modular attribution package with direct imports:

    from sim_bench.analysis.attribution.resnet50 import ResNet50AttributionExtractor
    from sim_bench.analysis.attribution.visualization import (
        plot_attribution_overlay,
        plot_attribution_comparison,
        visualize_feature_dimensions
    )

The attribution functionality has been split into:
    - attribution/base.py: Base classes and interfaces
    - attribution/resnet50.py: ResNet-50 specific attribution
    - attribution/visualization.py: General visualization utilities

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

warnings.warn(
    "sim_bench.analysis.feature_attribution is deprecated. "
    "Use direct imports from sim_bench.analysis.attribution.resnet50 and "
    "sim_bench.analysis.attribution.visualization instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility with direct imports (no __init__ tricks)
try:
    from .attribution.resnet50 import ResNet50AttributionExtractor as ResNet50FeatureExtractor
    from .attribution.visualization import (
        plot_attribution_overlay,
        plot_attribution_comparison as compare_feature_activations,
        visualize_feature_dimensions
    )
except ImportError as e:
    raise ImportError(
        "Could not import from attribution submodules. "
        f"Error: {e}"
    )

__all__ = [
    'ResNet50FeatureExtractor',
    'plot_attribution_overlay',
    'visualize_feature_dimensions',
    'compare_feature_activations'
]
