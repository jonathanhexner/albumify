from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from .config import PlotGridConfig, get_global_config
from .io import build_index_to_path_via_dataset, load_per_query, load_rankings


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class QueryData:
    """Container for query-related data."""
    query_idx: int
    query_path: str
    query_group: int
    top_results: pd.DataFrame  # Rankings for this query
    image_index_to_path: Dict[int, str]
    group_map: Dict[int, int]  # result_idx -> group_id


@dataclass
class GroundTruthData:
    """Container for ground truth images."""
    indices: List[int]
    group_id: int


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_query_data(
    method: str,
    query_idx: int,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    experiment_dir: Path,
    top_n: int = 10
) -> QueryData:
    """
    Load all data needed for visualizing a query and its results.
    
    Args:
        method: Method name
        query_idx: Query image index
        dataset_name: Dataset name
        dataset_config: Dataset configuration
        experiment_dir: Experiment directory
        top_n: Number of top results to load
    
    Returns:
        QueryData object with all necessary information
    """
    # Load CSVs
    per_query = load_per_query(method, experiment_dir)
    rankings = load_rankings(method, experiment_dir)
    
    # Get query info
    query_row = per_query[per_query["query_idx"] == query_idx].iloc[0]
    query_group = int(query_row["group_id"])
    query_path = str(query_row["query_path"])
    
    # Get top results for this query
    query_results = rankings[rankings["query_idx"] == query_idx].sort_values("rank")
    query_results = query_results[query_results["rank"] > 0].head(top_n)
    
    # Build mappings
    image_index_to_path = build_index_to_path_via_dataset(dataset_name, dataset_config)
    group_map = _build_group_map_for_results(query_results, per_query)
    
    return QueryData(
        query_idx=query_idx,
        query_path=query_path,
        query_group=query_group,
        top_results=query_results,
        image_index_to_path=image_index_to_path,
        group_map=group_map
    )


def load_ground_truth_for_query(
    query_idx: int,
    query_group: int,
    per_query: pd.DataFrame,
    max_gt_images: int = 3
) -> GroundTruthData:
    """
    Find ground truth images for a query (same group, excluding query itself).
    
    Args:
        query_idx: Query image index
        query_group: Query's group ID
        per_query: per_query.csv DataFrame
        max_gt_images: Maximum number of ground truth images to return
    
    Returns:
        GroundTruthData with indices of ground truth images
    """
    gt_indices = []
    
    # Find all images in the same group (excluding the query)
    for _, row in per_query.iterrows():
        idx = int(row["query_idx"])
        if idx != query_idx and int(row["group_id"]) == query_group:
            gt_indices.append(idx)
            if len(gt_indices) >= max_gt_images:
                break
    
    return GroundTruthData(indices=gt_indices, group_id=query_group)


def _build_group_map_for_results(results_df: pd.DataFrame, per_query: pd.DataFrame) -> Dict[int, int]:
    """Build mapping from result_idx to group_id for quick lookups."""
    group_map = {}
    for _, r in results_df.iterrows():
        result_idx = int(r["result_idx"])
        result_row = per_query[per_query["query_idx"] == result_idx]
        if not result_row.empty:
            group_map[result_idx] = int(result_row.iloc[0]["group_id"])
    return group_map


# ============================================================================
# Rendering Utilities
# ============================================================================

def _render_image(ax: plt.Axes, image_path: Path, title: str, title_color: str = "black", fontsize: int = 10):
    """Render an image with title on a matplotlib axis."""
    img = Image.open(image_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=fontsize, color=title_color)
    ax.axis("off")


def _format_result_title(
    rank: int,
    result_idx: int,
    distance: float,
    result_group: int,
    query_group: int,
    filename: str = ""
) -> str:
    """Format a compact title for a result image."""
    match_indicator = "✓" if result_group == query_group else ""
    if filename:
        return f"#{rank} {filename}\nidx={result_idx} grp={result_group}{match_indicator} d={distance:.3f}"
    return f"#{rank} idx={result_idx} grp={result_group}{match_indicator} d={distance:.3f}"


# ============================================================================
# Main Plotting Function
# ============================================================================

def plot_query_topn_grid(
    method: str,
    query_idx: int,
    config: PlotGridConfig,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    experiment_dir: Optional[Path] = None,
    suptitle: Optional[str] = None,
    show_ground_truth: bool = False,
) -> plt.Figure:
    """
    Plot query and top-N results in a grid with group/distance information.
    
    Args:
        method: Method name (e.g., 'deep', 'sift_bovw')
        query_idx: Query image index
        config: Plot configuration (figsize, fonts, etc.)
        dataset_name: Dataset name (e.g., 'ukbench', 'holidays')
        dataset_config: Dataset configuration dict
        experiment_dir: Experiment directory (uses global config if None)
        suptitle: Custom figure title (default: "{method} — Query {query_idx}")
        show_ground_truth: If True, adds a row with ground truth images
    
    Returns:
        matplotlib Figure object
    """
    experiment_dir = Path(experiment_dir or get_global_config().experiment_dir)
    
    # Determine top_n based on mode
    top_n = 5 if show_ground_truth else config.top_n
    
    # Load data
    query_data = load_query_data(method, query_idx, dataset_name, dataset_config, experiment_dir, top_n=top_n)
    
    # Load ground truth if needed
    ground_truth = None
    if show_ground_truth:
        per_query = load_per_query(method, experiment_dir)
        ground_truth = load_ground_truth_for_query(query_idx, query_data.query_group, per_query)
    
    # Calculate grid size
    num_results = len(query_data.top_results)
    cols = config.max_per_row
    
    if show_ground_truth:
        # Fixed 3x3 grid for ground truth mode
        rows = 3
        cols = 3
        total_cells = 1 + num_results + len(ground_truth.indices)
    else:
        # Dynamic grid
        total_cells = 1 + num_results
        rows = (total_cells + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * config.figsize_per_cell_w, rows * config.figsize_per_cell_h))
    axes_flat = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]
    
    cell_idx = 0
    
    # Render query
    query_filename = Path(query_data.query_path).name
    _render_image(
        axes_flat[cell_idx],
        Path(query_data.query_path),
        f"QUERY: {query_filename}\n(idx={query_data.query_idx}, grp={query_data.query_group})",
        title_color="blue",
        fontsize=config.label_fontsize
    )
    cell_idx += 1

    # Render results
    for _, r in query_data.top_results.iterrows():
        result_idx = int(r["result_idx"])
        distance = float(r["distance"]) if pd.notna(r["distance"]) else float("nan")
        rank = int(r['rank'])
        result_group = query_data.group_map.get(result_idx, -1)
        result_path = Path(query_data.image_index_to_path[result_idx])
        result_filename = result_path.name

        title = _format_result_title(rank, result_idx, distance, result_group, query_data.query_group, result_filename)
        _render_image(
            axes_flat[cell_idx],
            result_path,
            title,
            fontsize=config.label_fontsize
        )
        cell_idx += 1

    # Render ground truth (if enabled)
    if ground_truth:
        for gt_idx in ground_truth.indices:
            gt_path = Path(query_data.image_index_to_path[gt_idx])
            gt_filename = gt_path.name
            _render_image(
                axes_flat[cell_idx],
                gt_path,
                f"GT: {gt_filename}\n(idx={gt_idx}, grp={ground_truth.group_id})",
                title_color="green",
                fontsize=config.label_fontsize
            )
            cell_idx += 1
    
    # Hide unused cells
    for idx in range(cell_idx, len(axes_flat)):
        axes_flat[idx].axis("off")
    
    # Set title and layout
    fig.suptitle(suptitle or f"{method} — Query {query_idx}", fontsize=config.title_fontsize)
    plt.tight_layout()
    
    # Save
    save_dir = Path(get_global_config().save_dir or experiment_dir / "analysis_outputs")
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_with_gt" if show_ground_truth else config.save_filename_suffix
    fig.savefig(save_dir / f"{method}_q{query_idx}{suffix}.png", dpi=150)
    
    return fig
