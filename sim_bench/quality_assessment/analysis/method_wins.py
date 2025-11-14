"""
Find and visualize method wins (cases where one method succeeds but others fail).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend with proper window controls
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def find_method_wins(
    merged_df: pd.DataFrame,
    methods: List[str],
    top_n_series: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Find series where each method succeeds but others fail.
    
    Args:
        merged_df: Merged DataFrame from merge_per_series_metrics()
        methods: List of methods to analyze
        top_n_series: Number of winning series to return per method
        
    Returns:
        Dict mapping method_name -> DataFrame of winning series
    """
    wins_dict = {}
    
    for method in methods:
        method_col = f'{method}_correct'
        if method_col not in merged_df.columns:
            continue
        
        # Find series where this method is correct
        method_correct = merged_df[merged_df[method_col] == 1]
        
        # Find series where all other methods are wrong
        other_methods = [m for m in methods if m != method]
        other_cols = [f'{m}_correct' for m in other_methods if f'{m}_correct' in merged_df.columns]
        
        if other_cols:
            # All other methods should be 0 (incorrect)
            all_others_wrong = (method_correct[other_cols].sum(axis=1) == 0)
            wins = method_correct[all_others_wrong].copy()
        else:
            wins = method_correct.copy()
        
        # Take top N
        wins = wins.head(top_n_series)
        wins_dict[method] = wins
    
    return wins_dict


def get_top_methods_by_accuracy(
    methods_summary_df: pd.DataFrame,
    top_n: int = 4
) -> List[str]:
    """
    Get top N methods by Top-1 accuracy.
    
    Args:
        methods_summary_df: Methods summary DataFrame
        top_n: Number of top methods to return
        
    Returns:
        List of method names
    """
    df_sorted = methods_summary_df.sort_values('avg_top1_accuracy', ascending=False)
    return df_sorted['method'].head(top_n).tolist()


def visualize_method_wins(
    wins_dict: Dict[str, pd.DataFrame],
    series_data_dict: Dict[str, pd.DataFrame],
    top_n_methods: int = 4,
    top_n_images: int = 3,
    output_dir: Optional[Path] = None,
    show_plots: bool = True
) -> List[Path]:
    """
    Visualize method wins showing ground truth and top predictions.
    
    Args:
        wins_dict: Dict from find_method_wins()
        series_data_dict: Dict mapping method_name -> per-series DataFrame
        top_n_methods: Number of top methods to show (default: 4)
        top_n_images: Number of top images to show per method (default: 3)
        output_dir: Directory to save figures (None = don't save)
        show_plots: Whether to display plots
        
    Returns:
        List of saved figure paths
    """
    saved_figs = []
    
    # Get top methods (will use methods from wins_dict, but limit to top_n_methods)
    methods = list(wins_dict.keys())[:top_n_methods]
    
    for method, wins_df in wins_dict.items():
        if method not in methods:
            continue
        
        if len(wins_df) == 0:
            continue
        
        method_series_df = series_data_dict.get(method)
        if method_series_df is None:
            continue
        
        # Process each winning series
        for idx, win_row in wins_df.iterrows():
            group_id = win_row['group_id']
            
            # Get series data for this method
            series_row = method_series_df[method_series_df['group_id'] == group_id].iloc[0]
            
            # Parse data
            scores = [float(s) for s in str(series_row['scores']).split(',')]
            image_paths = str(series_row['image_paths']).split(',')
            ground_truth_idx = series_row['ground_truth_idx']
            
            # Get ground truth image
            gt_image_path = image_paths[ground_truth_idx]
            
            # Create figure: Ground truth row + methods rows (each method shows top 3 images)
            n_cols = top_n_methods + 1  # +1 for ground truth
            n_rows = 1 + top_n_images  # 1 for ground truth, top_n_images for method images
            fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
            
            # Plot ground truth (spans full width in first row)
            ax_gt = fig.add_subplot(gs[0, :])
            try:
                img = Image.open(gt_image_path)
                ax_gt.imshow(img)
                ax_gt.axis('off')
                ax_gt.set_title(f'Ground Truth Best Image\n{Path(gt_image_path).name}', 
                              fontsize=12, fontweight='bold')
            except Exception as e:
                ax_gt.text(0.5, 0.5, f'Error loading image:\n{e}', 
                          ha='center', va='center', transform=ax_gt.transAxes)
                ax_gt.set_title('Ground Truth (Error)', fontsize=12)
            
            # Plot top 3 images for each method
            for method_idx, method_name in enumerate(methods):
                if method_name not in series_data_dict:
                    continue
                
                method_series = series_data_dict[method_name]
                method_series_row = method_series[method_series['group_id'] == group_id]
                
                if len(method_series_row) == 0:
                    continue
                
                method_series_row = method_series_row.iloc[0]
                method_scores = [float(s) for s in str(method_series_row['scores']).split(',')]
                method_paths = str(method_series_row['image_paths']).split(',')
                
                # Get top N images by score
                top_indices = sorted(range(len(method_scores)), 
                                   key=lambda i: method_scores[i], reverse=True)[:top_n_images]
                
                # Plot top images for this method (one per row, in method's column)
                for img_idx, top_idx in enumerate(top_indices):
                    row = 1 + img_idx  # Start from row 1 (row 0 is ground truth)
                    col = method_idx + 1  # Column 0 is reserved, methods start at 1
                    ax = fig.add_subplot(gs[row, col])
                    
                    try:
                        img = Image.open(method_paths[top_idx])
                        ax.imshow(img)
                        ax.axis('off')
                        
                        score = method_scores[top_idx]
                        filename = Path(method_paths[top_idx]).name
                        rank = img_idx + 1
                        
                        # Method name only on first image
                        if img_idx == 0:
                            title = f'{method_name}\nRank {rank}: {score:.3f}\n{filename}'
                            fontweight = 'bold'
                        else:
                            title = f'Rank {rank}: {score:.3f}\n{filename}'
                            fontweight = 'normal'
                        
                        ax.set_title(title, fontsize=9, fontweight=fontweight)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                              transform=ax.transAxes, fontsize=8)
                        ax.set_title(f'{method_name} (Error)' if img_idx == 0 else 'Error', fontsize=9)
            
            # Add overall title
            fig.suptitle(f'Method Win: {method} (Series {group_id})', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                fig_path = output_dir / f'{method}_win_series_{group_id}.png'
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                saved_figs.append(fig_path)
            
            if show_plots:
                plt.show(block=False)
                # Ensure figure window has proper controls
                try:
                    fig.canvas.manager.set_window_title(f'Method Win: {method} (Series {group_id})')
                except AttributeError:
                    pass  # Some backends don't support set_window_title
            else:
                plt.close(fig)
    
    return saved_figs

