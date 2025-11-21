"""
Failure case analysis for quality assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def analyze_failures(
    merged_df: pd.DataFrame,
    series_data_dict: Dict[str, pd.DataFrame],
    methods: List[str]
) -> Dict:
    """
    Analyze failure cases across methods.
    
    Args:
        merged_df: Merged DataFrame from merge_per_series_metrics()
        series_data_dict: Dict mapping method_name -> per-series DataFrame
        methods: List of methods to analyze
        
    Returns:
        Dict with failure statistics
    """
    stats = {}
    
    # All methods fail
    method_cols = [f'{m}_correct' for m in methods if f'{m}_correct' in merged_df.columns]
    if method_cols:
        all_fail = (merged_df[method_cols].sum(axis=1) == 0)
        stats['all_fail_count'] = all_fail.sum()
        stats['all_fail_series'] = merged_df[all_fail]['group_id'].tolist()
    
    # Single method succeeds
    stats['single_success'] = {}
    for method in methods:
        method_col = f'{method}_correct'
        if method_col not in merged_df.columns:
            continue
        
        other_methods = [m for m in methods if m != method]
        other_cols = [f'{m}_correct' for m in other_methods if f'{m}_correct' in merged_df.columns]
        
        if other_cols:
            method_success = merged_df[merged_df[method_col] == 1]
            others_fail = (method_success[other_cols].sum(axis=1) == 0)
            single_success = method_success[others_fail]
            stats['single_success'][method] = {
                'count': len(single_success),
                'series': single_success['group_id'].tolist()
            }
    
    # Failure rate by series size
    if methods:
        first_method = methods[0]
        if first_method in series_data_dict:
            series_df = series_data_dict[first_method]
            failure_by_size = {}
            
            for method in methods:
                method_col = f'{method}_correct'
                if method_col not in merged_df.columns:
                    continue
                
                # Merge with series size
                merged_with_size = merged_df.merge(
                    series_df[['group_id', 'num_images']],
                    on='group_id',
                    how='left'
                )
                
                # Group by series size
                size_groups = merged_with_size.groupby('num_images')
                failure_rates = {}
                
                for size, group in size_groups:
                    failures = (group[method_col] == 0).sum()
                    total = len(group)
                    failure_rates[size] = failures / total if total > 0 else 0
                
                failure_by_size[method] = failure_rates
            
            stats['failure_by_size'] = failure_by_size
    
    return stats


def plot_failure_analysis(
    failure_stats: Dict,
    methods: List[str],
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Plot failure analysis visualizations.
    
    Args:
        failure_stats: Stats from analyze_failures()
        methods: List of methods
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Subplot 1: All methods fail count
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(['All Methods Fail'], [failure_stats.get('all_fail_count', 0)], color='red', alpha=0.7)
    ax1.set_ylabel('Number of Series')
    ax1.set_title('Series Where All Methods Fail', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Single method success
    ax2 = plt.subplot(2, 2, 2)
    single_success = failure_stats.get('single_success', {})
    if single_success:
        method_names = list(single_success.keys())
        counts = [single_success[m]['count'] for m in method_names]
        ax2.bar(method_names, counts, color='green', alpha=0.7)
        ax2.set_ylabel('Number of Series')
        ax2.set_title('Series Where Only One Method Succeeds', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Failure rate by series size
    ax3 = plt.subplot(2, 2, 3)
    failure_by_size = failure_stats.get('failure_by_size', {})
    if failure_by_size:
        for method in methods[:4]:  # Limit to 4 methods for readability
            if method in failure_by_size:
                sizes = sorted(failure_by_size[method].keys())
                rates = [failure_by_size[method][s] for s in sizes]
                ax3.plot(sizes, rates, marker='o', label=method, linewidth=2)
        
        ax3.set_xlabel('Series Size (Number of Images)')
        ax3.set_ylabel('Failure Rate')
        ax3.set_title('Failure Rate by Series Size', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # Subplot 4: Summary statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = "Failure Analysis Summary\n\n"
    summary_text += f"All methods fail: {failure_stats.get('all_fail_count', 0)} series\n\n"
    
    single_success = failure_stats.get('single_success', {})
    if single_success:
        summary_text += "Single method success:\n"
        for method, data in single_success.items():
            summary_text += f"  {method}: {data['count']} series\n"
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace')
    
    plt.suptitle('Failure Case Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig




