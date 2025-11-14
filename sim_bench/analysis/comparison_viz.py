"""Visualization utilities for comparing methods side-by-side."""

from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend with proper window controls
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import logging
import warnings
import sys
import os
from contextlib import contextmanager
from io import StringIO

from .plotting import load_query_data


@contextmanager
def suppress_output():
    """Context manager to suppress all output (stdout, stderr, logging, warnings)."""
    # Save old settings
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_log_level = logging.root.level

    try:
        # Redirect stdout and stderr to string buffers
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # Suppress logging and warnings
        logging.root.setLevel(logging.CRITICAL)
        warnings.filterwarnings('ignore')

        yield
    finally:
        # Restore
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.root.setLevel(old_log_level)
        warnings.filterwarnings('default')


def plot_query_comparison_grid(
    methods: List[str],
    query_idx: int,
    query_path: str,
    metric_scores: Dict[str, float],
    metric_name: str,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    method_to_exp_dir: Dict[str, Path],
    top_n: int = 5,
    figsize_per_col: float = 3.0,
    figsize_per_row: float = 2.5,
    save_path: Path = None
) -> plt.Figure:
    """
    Plot query and top-N results for multiple methods side-by-side.

    Creates a grid with:
    - Row 0: Query image (centered, spanning all columns)
    - Rows 1-N: Top-N results for each method (one column per method)

    Args:
        methods: List of method names to compare
        query_idx: Query image index
        query_path: Path to query image
        metric_scores: Dict mapping method -> metric score (e.g., {'deep': 0.95, 'dinov2': 0.32})
        metric_name: Name of metric being compared (e.g., 'ap@10')
        dataset_name: Dataset name
        dataset_config: Dataset configuration dict
        method_to_exp_dir: Dict mapping method -> experiment directory for that method
        top_n: Number of top results to show per method
        figsize_per_col: Figure width per column (method)
        figsize_per_row: Figure height per row
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_query_comparison_grid(
        ...     methods=['deep', 'dinov2', 'openclip'],
        ...     query_idx=123,
        ...     query_path='/path/to/query.jpg',
        ...     metric_scores={'deep': 0.95, 'dinov2': 0.32, 'openclip': 0.28},
        ...     metric_name='ap@10',
        ...     dataset_name='holidays',
        ...     dataset_config={'name': 'holidays', 'root': '...'},
        ...     experiment_dir=Path('outputs/.../2025-11-02_00-12-26'),
        ...     top_n=5
        ... )
    """
    num_methods = len(methods)
    num_rows = top_n + 1  # +1 for query row

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(num_methods * figsize_per_col, num_rows * figsize_per_row))
    gs = fig.add_gridspec(num_rows, num_methods, hspace=0.3, wspace=0.3)

    # Row 0: Query image (spans all columns)
    ax_query = fig.add_subplot(gs[0, :])
    query_filename = Path(query_path).name
    img_query = Image.open(query_path)
    ax_query.imshow(img_query)
    ax_query.set_title(
        f"QUERY: {query_filename} (idx={query_idx})",
        fontsize=14,
        fontweight='bold',
        color='blue'
    )
    ax_query.axis('off')

    # Load data and plot results for each method
    for col_idx, method in enumerate(methods):
        # Load query data for this method using its specific experiment directory
        method_exp_dir = method_to_exp_dir.get(method)
        if not method_exp_dir:
            print(f"Warning: No experiment directory found for method '{method}'")
            continue

        query_data = load_query_data(
            method=method,
            query_idx=query_idx,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            experiment_dir=method_exp_dir,
            top_n=top_n
        )

        # Get score and determine color
        score = metric_scores.get(method, 0.0)
        score_color = 'green' if score >= 0.7 else 'red' if score <= 0.4 else 'orange'

        # Plot top-N results for this method
        for row_idx in range(top_n):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])

            if row_idx < len(query_data.top_results):
                # Get result info
                result_row = query_data.top_results.iloc[row_idx]
                result_idx = int(result_row['result_idx'])
                rank = int(result_row['rank'])
                distance = float(result_row['distance']) if pd.notna(result_row['distance']) else float('nan')
                result_group = query_data.group_map.get(result_idx, -1)
                result_path = Path(query_data.image_index_to_path[result_idx])
                result_filename = result_path.name

                # Render image
                img = Image.open(result_path)
                ax.imshow(img)

                # Title with match indicator (use simple text instead of Unicode)
                is_match = result_group == query_data.query_group
                match_indicator = "[OK]" if is_match else "[X]"
                title_color = 'green' if is_match else 'red'

                title = f"#{rank} {match_indicator}\n{result_filename}\nd={distance:.3f}"
                ax.set_title(title, fontsize=8, color=title_color)

                # Add method label and score on first row
                if row_idx == 0:
                    ax.text(
                        0.5, -0.2,
                        f"{method.upper()}\n{metric_name}={score:.3f}",
                        transform=ax.transAxes,
                        ha='center',
                        fontsize=11,
                        fontweight='bold',
                        color=score_color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=score_color, linewidth=2)
                    )

            ax.axis('off')

    # Overall title
    fig.suptitle(
        f"Method Comparison: Query {query_idx} on {dataset_name}",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_all_method_wins(
    all_wins: Dict[str, pd.DataFrame],
    methods: List[str],
    metric_name: str,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    experiment_infos: List[Dict[str, Any]],
    output_dir: Path,
    top_n: int = 5,
    show_plots: bool = True
) -> Dict[str, List[Path]]:
    """
    Generate comparison visualizations for all winning queries.

    Args:
        all_wins: Dict mapping method -> DataFrame of winning queries (from find_all_method_wins)
        methods: List of all method names
        metric_name: Name of metric being compared (e.g., 'ap@10')
        dataset_name: Dataset name
        dataset_config: Dataset configuration dict
        experiment_infos: List of experiment info dicts (from load_experiments)
        output_dir: Directory to save output figures
        top_n: Number of top results to show per method
        show_plots: If True, call plt.show() for each figure

    Returns:
        Dict mapping method -> list of saved figure paths

    Example:
        >>> saved_figs = visualize_all_method_wins(
        ...     all_wins=all_wins,
        ...     methods=['deep', 'dinov2', 'openclip'],
        ...     metric_name='ap@10',
        ...     dataset_name='holidays',
        ...     dataset_config={'name': 'holidays', 'root': '...'},
        ...     experiment_infos=experiment_infos,
        ...     output_dir=Path('sim_bench/analysis/reports/method_wins'),
        ...     top_n=5
        ... )
    """
    # Build method -> experiment_dir mapping for this dataset
    method_to_exp_dir = {}
    for info in experiment_infos:
        if info['dataset'] == dataset_name:
            method_to_exp_dir[info['method']] = info['experiment_dir']

    saved_figures = {}

    for winning_method, df_wins in all_wins.items():
        method_figures = []

        if len(df_wins) == 0:
            saved_figures[winning_method] = method_figures
            continue

        print(f"\n{winning_method.upper()}: {len(df_wins)} winning queries")

        for idx, row in df_wins.iterrows():
            query_idx = int(row['query_idx'])
            query_path = str(row['query_path'])

            # Get metric scores for all methods
            metric_scores = {
                method: row[f"{metric_name}_{method}"]
                for method in methods
            }

            # Create comparison grid (suppress all output)
            save_path = output_dir / f"comparison_{winning_method}_wins_q{query_idx}.png"

            with suppress_output():
                fig = plot_query_comparison_grid(
                    methods=methods,
                    query_idx=query_idx,
                    query_path=query_path,
                    metric_scores=metric_scores,
                    metric_name=metric_name,
                    dataset_name=dataset_name,
                    dataset_config=dataset_config,
                    method_to_exp_dir=method_to_exp_dir,
                    top_n=top_n,
                    save_path=save_path
                )

            method_figures.append(save_path)

            if show_plots:
                plt.show(block=False)
                # Ensure figure window has proper controls
                try:
                    fig.canvas.manager.set_window_title(f'Method Comparison: {winning_method}')
                except AttributeError:
                    pass  # Some backends don't support set_window_title
            else:
                plt.close(fig)

        saved_figures[winning_method] = method_figures

    total_figs = sum(len(figs) for figs in saved_figures.values())
    print(f"\n✓ Generated {total_figs} comparison figures")

    return saved_figures
