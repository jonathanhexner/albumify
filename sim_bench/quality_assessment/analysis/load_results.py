"""
Load quality assessment benchmark results.

Supports flexible loading from single folder or auto-scanning multiple folders.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def find_benchmark_folders(base_dir: Path) -> List[Path]:
    """
    Find all benchmark folders in base directory.
    
    Args:
        base_dir: Base directory containing benchmark_* folders
        
    Returns:
        List of benchmark folder paths, sorted by modification time (newest first)
    """
    if not base_dir.exists():
        return []
    
    benchmark_folders = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('benchmark_')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    return benchmark_folders


def load_quality_results(
    benchmark_dir: Optional[Path] = None,
    auto_scan: bool = False,
    base_dir: Optional[Path] = None,
    use_latest: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Path]:
    """
    Load quality assessment benchmark results.
    
    Args:
        benchmark_dir: Path to specific benchmark folder (if not auto_scan)
        auto_scan: If True, scan base_dir for benchmark folders
        base_dir: Base directory to scan (if auto_scan)
        use_latest: If True and auto_scan, use latest folder; else return all
        
    Returns:
        Tuple of:
        - methods_summary_df: Overall method comparison
        - detailed_results_df: Per-dataset, per-method results
        - series_data_dict: Dict mapping method_name -> per-series DataFrame
        - actual_benchmark_dir: Path to the benchmark folder used
    """
    if auto_scan:
        if base_dir is None:
            raise ValueError("base_dir required when auto_scan=True")
        
        base_dir = Path(base_dir)
        benchmark_folders = find_benchmark_folders(base_dir)
        
        if not benchmark_folders:
            raise ValueError(f"No benchmark folders found in {base_dir}")
        
        if use_latest:
            benchmark_dir = benchmark_folders[0]
            print(f"Using latest benchmark: {benchmark_dir.name}")
        else:
            # For now, use latest (can extend to support multiple later)
            benchmark_dir = benchmark_folders[0]
            print(f"Found {len(benchmark_folders)} benchmark folders, using latest: {benchmark_dir.name}")
    else:
        if benchmark_dir is None:
            raise ValueError("benchmark_dir required when auto_scan=False")
    
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        raise ValueError(f"Benchmark directory not found: {benchmark_dir}")
    
    # Load methods_summary.csv
    methods_summary_path = benchmark_dir / "methods_summary.csv"
    if not methods_summary_path.exists():
        raise ValueError(f"methods_summary.csv not found in {benchmark_dir}")
    
    methods_summary_df = pd.read_csv(methods_summary_path)
    
    # Load detailed_results.csv
    detailed_results_path = benchmark_dir / "detailed_results.csv"
    if not detailed_results_path.exists():
        raise ValueError(f"detailed_results.csv not found in {benchmark_dir}")
    
    detailed_results_df = pd.read_csv(detailed_results_path)
    
    # Load all per-series CSV files
    series_data_dict = {}
    dataset_name = None
    
    for csv_file in benchmark_dir.glob("*_series.csv"):
        # Parse method name from filename: phototriage_sharpness_only_series.csv -> sharpness_only
        parts = csv_file.stem.split('_')
        if len(parts) >= 3 and parts[-1] == 'series':
            dataset_name = parts[0]  # e.g., 'phototriage'
            method_name = '_'.join(parts[1:-1])  # e.g., 'sharpness_only'
            
            df = pd.read_csv(csv_file)
            series_data_dict[method_name] = df
            print(f"Loaded {method_name}: {len(df)} series")
    
    if not series_data_dict:
        print("Warning: No per-series CSV files found")
    
    return methods_summary_df, detailed_results_df, series_data_dict, benchmark_dir


def parse_series_data(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse per-series CSV data to extract scores and image paths.
    
    Args:
        series_df: DataFrame with columns: group_id, scores, image_paths, etc.
        
    Returns:
        DataFrame with parsed scores and paths
    """
    df = series_df.copy()
    
    # Parse scores (comma-separated string to list)
    df['scores_list'] = df['scores'].apply(lambda x: [float(s) for s in str(x).split(',')])
    
    # Parse image paths (comma-separated string to list)
    df['image_paths_list'] = df['image_paths'].apply(lambda x: str(x).split(','))
    
    # Parse ranking (comma-separated indices)
    df['ranking_list'] = df['ranking'].apply(lambda x: [int(s) for s in str(x).split(',')])
    
    # Get top 3 scores and corresponding image paths
    def get_top_n_scores_and_paths(row, n=3):
        scores = row['scores_list']
        paths = row['image_paths_list']
        ranking = row['ranking_list']
        
        # Get top N indices by score (highest first)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        
        top_scores = [scores[i] for i in top_indices]
        top_paths = [paths[i] for i in top_indices]
        top_ranks = [ranking[i] for i in top_indices]  # Original rank in series
        
        return {
            'top_scores': top_scores,
            'top_paths': top_paths,
            'top_ranks': top_ranks
        }
    
    top_n_data = df.apply(lambda row: get_top_n_scores_and_paths(row, n=3), axis=1)
    df['top_3_scores'] = top_n_data.apply(lambda x: x['top_scores'])
    df['top_3_paths'] = top_n_data.apply(lambda x: x['top_paths'])
    df['top_3_ranks'] = top_n_data.apply(lambda x: x['top_ranks'])
    
    return df


def merge_per_series_metrics(
    series_data_dict: Dict[str, pd.DataFrame],
    methods: Optional[List[str]] = None,
    metric: str = 'correct'
) -> pd.DataFrame:
    """
    Merge per-series metrics across methods for correlation analysis.
    
    Args:
        series_data_dict: Dict mapping method_name -> per-series DataFrame
        methods: List of methods to include (None = all)
        metric: Metric to merge ('correct' for binary, or 'predicted_idx' for rank)
        
    Returns:
        Merged DataFrame with one row per series, columns for each method's metric
    """
    if methods is None:
        methods = list(series_data_dict.keys())
    
    # Start with first method's group_ids
    first_method = methods[0]
    merged_df = series_data_dict[first_method][['group_id']].copy()
    
    # Add metric column for each method
    for method in methods:
        if method not in series_data_dict:
            print(f"Warning: Method {method} not found in series data")
            continue
        
        method_df = series_data_dict[method]
        
        if metric == 'correct':
            # Binary: correct (1) or incorrect (0)
            merged_df[f'{method}_correct'] = method_df['correct'].astype(int)
        elif metric == 'predicted_idx':
            # Predicted index (0-based)
            merged_df[f'{method}_predicted'] = method_df['predicted_idx']
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return merged_df

