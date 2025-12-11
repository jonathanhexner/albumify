"""
Visualization functions for quality assessment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def plot_performance_comparison(
    methods_summary_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot bar charts comparing methods across multiple metrics.
    
    Args:
        methods_summary_df: DataFrame with method performance metrics
        metrics: List of metrics to plot (None = auto-detect)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        # Auto-detect metrics (exclude method name and non-metric columns)
        exclude_cols = {'method', 'datasets_tested'}
        metrics = [col for col in methods_summary_df.columns if col not in exclude_cols]
        # Focus on key metrics
        key_metrics = ['avg_top1_accuracy', 'avg_top2_accuracy', 'avg_mrr']
        metrics = [m for m in key_metrics if m in metrics] or metrics[:3]
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df_sorted = methods_summary_df.sort_values(metric, ascending=False)
        
        bars = ax.bar(range(len(df_sorted)), df_sorted[metric], color='steelblue')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['method'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Method Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_runtime_comparison(
    detailed_results_df: pd.DataFrame,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot runtime comparison across methods.
    
    Args:
        detailed_results_df: DataFrame with runtime data
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df_sorted = detailed_results_df.sort_values('avg_time_ms', ascending=True)
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['avg_time_ms'], color='coral')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['method'])
    ax.set_xlabel('Average Time per Image (ms)')
    ax.set_title('Method Runtime Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['avg_time_ms'])):
        ax.text(bar.get_width() + max(df_sorted['avg_time_ms']) * 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}ms', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def plot_efficiency_comparison(
    methods_summary_df: pd.DataFrame,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot efficiency (accuracy vs time) scatter plot.
    
    Args:
        methods_summary_df: DataFrame with method metrics
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate efficiency (accuracy / time)
    methods_summary_df = methods_summary_df.copy()
    methods_summary_df['efficiency'] = methods_summary_df['avg_top1_accuracy'] / methods_summary_df['avg_time_ms']
    
    scatter = ax.scatter(
        methods_summary_df['avg_time_ms'],
        methods_summary_df['avg_top1_accuracy'],
        s=200,
        alpha=0.6,
        c=methods_summary_df['efficiency'],
        cmap='viridis'
    )
    
    # Add method labels
    for idx, row in methods_summary_df.iterrows():
        ax.annotate(row['method'], 
                   (row['avg_time_ms'], row['avg_top1_accuracy']),
                   fontsize=9, ha='center', va='bottom')
    
    ax.set_xlabel('Average Time per Image (ms)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Speed Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Efficiency (Accuracy/Time)', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    methods: Optional[List[str]] = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot correlation heatmap between methods.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        methods: List of methods to include (None = all)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if methods is not None:
        corr_matrix = corr_matrix.loc[methods, methods]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title('Method Correlation Matrix\n(Top-1 Accuracy Agreement)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig






