"""
Quality Assessment Analysis Module

Functions for analyzing quality assessment benchmark results.
"""

from .load_results import load_quality_results, merge_per_series_metrics
from .correlation import compute_correlation_matrix
from .visualization import (
    plot_performance_comparison,
    plot_runtime_comparison,
    plot_correlation_heatmap,
    plot_efficiency_comparison
)
from .method_wins import find_method_wins, visualize_method_wins, get_top_methods_by_accuracy
from .failure_analysis import analyze_failures, plot_failure_analysis

__all__ = [
    'load_quality_results',
    'merge_per_series_metrics',
    'compute_correlation_matrix',
    'plot_performance_comparison',
    'plot_runtime_comparison',
    'plot_correlation_heatmap',
    'plot_efficiency_comparison',
    'find_method_wins',
    'visualize_method_wins',
    'get_top_methods_by_accuracy',
    'analyze_failures',
    'plot_failure_analysis',
]

