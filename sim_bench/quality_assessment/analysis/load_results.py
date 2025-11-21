"""
Load quality assessment benchmark results.

Supports flexible loading from single folder or auto-scanning multiple folders.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(('pairwise_', 'benchmark_'))],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    return benchmark_folders


def _resolve_benchmark_dirs(
    benchmark_dir: Optional[Path],
    auto_scan: bool,
    base_dir: Optional[Path],
    use_latest: bool
) -> List[Path]:
    """
    Resolve benchmark directory/directories from various input modes.
    
    Args:
        benchmark_dir: Path to specific benchmark folder (if not auto_scan)
        auto_scan: If True, scan base_dir for benchmark folders
        base_dir: Base directory to scan (if auto_scan)
        use_latest: If True and auto_scan, use only latest folder; else use all
        
    Returns:
        List of resolved benchmark directory paths (single item if use_latest=True)
        
    Raises:
        ValueError: If required parameters are missing or no folders found
    """
    if auto_scan:
        if base_dir is None:
            raise ValueError("base_dir required when auto_scan=True")
        
        benchmark_folders = find_benchmark_folders(Path(base_dir))
        if not benchmark_folders:
            raise ValueError(f"No benchmark folders found in {base_dir}")
        
        if use_latest:
            selected_dirs = [benchmark_folders[0]]
            logger.info(f"Using latest benchmark: {selected_dirs[0].name}")
        else:
            selected_dirs = benchmark_folders
            logger.info(f"Found {len(benchmark_folders)} benchmark folders, loading all")
        
        return selected_dirs
    
    if benchmark_dir is None:
        raise ValueError("benchmark_dir required when auto_scan=False")
    
    return [Path(benchmark_dir)]


def _validate_benchmark_dir(benchmark_dir: Path) -> None:
    """Validate that benchmark directory exists and contains required files."""
    if not benchmark_dir.exists():
        raise ValueError(f"Benchmark directory not found: {benchmark_dir}")
    
    required_files = ["methods_summary.csv", "detailed_results.csv"]
    missing_files = [f for f in required_files if not (benchmark_dir / f).exists()]
    
    if missing_files:
        raise ValueError(f"Missing required files in {benchmark_dir}: {', '.join(missing_files)}")


def _load_summary_files(benchmark_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load methods_summary.csv and detailed_results.csv from a single directory."""
    methods_summary_df = pd.read_csv(benchmark_dir / "methods_summary.csv")
    detailed_results_df = pd.read_csv(benchmark_dir / "detailed_results.csv")
    return methods_summary_df, detailed_results_df


def _load_and_concatenate_summary_files(benchmark_dirs: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate summary files from multiple benchmark directories.
    
    Args:
        benchmark_dirs: List of benchmark directory paths
        
    Returns:
        Tuple of concatenated DataFrames (methods_summary_df, detailed_results_df)
    """
    all_methods_summary = []
    all_detailed_results = []
    
    for benchmark_dir in benchmark_dirs:
        _validate_benchmark_dir(benchmark_dir)
        methods_df, detailed_df = _load_summary_files(benchmark_dir)
        
        # Add source folder identifier
        methods_df['source_folder'] = benchmark_dir.name
        detailed_df['source_folder'] = benchmark_dir.name
        
        all_methods_summary.append(methods_df)
        all_detailed_results.append(detailed_df)
    
    # Concatenate and reset index
    methods_summary_df = pd.concat(all_methods_summary, ignore_index=True)
    detailed_results_df = pd.concat(all_detailed_results, ignore_index=True)
    
    logger.info(f"Loaded and concatenated results from {len(benchmark_dirs)} benchmark folders")
    
    return methods_summary_df, detailed_results_df


def _parse_method_name_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse dataset and method name from series CSV filename.
    
    Args:
        filename: Filename like "phototriage_sharpness_only_series.csv"
        
    Returns:
        Tuple of (dataset_name, method_name) or None if parsing fails
    """
    parts = Path(filename).stem.split('_')
    if len(parts) >= 3 and parts[-1] == 'series':
        dataset_name = parts[0]  # e.g., 'phototriage'
        method_name = '_'.join(parts[1:-1])  # e.g., 'sharpness_only'
        return dataset_name, method_name
    return None


def _load_series_data(benchmark_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all per-series CSV files from a single benchmark directory."""
    series_data_dict = {}
    
    for csv_file in benchmark_dir.glob("*_series.csv"):
        parsed = _parse_method_name_from_filename(csv_file.name)
        if parsed is None:
            logger.warning(f"Could not parse method name from: {csv_file.name}")
            continue
        
        dataset_name, method_name = parsed
        df = pd.read_csv(csv_file)
        series_data_dict[method_name] = df
        logger.info(f"Loaded {method_name}: {len(df)} series")
    
    if not series_data_dict:
        logger.warning("No per-series CSV files found")
    
    return series_data_dict


def _load_and_concatenate_series_data(benchmark_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    """
    Load and concatenate per-series data from multiple benchmark directories.
    
    Args:
        benchmark_dirs: List of benchmark directory paths
        
    Returns:
        Dict mapping method_name -> concatenated per-series DataFrame
    """
    all_series_data = {}
    
    for benchmark_dir in benchmark_dirs:
        series_data = _load_series_data(benchmark_dir)
        
        for method_name, df in series_data.items():
            # Add source folder identifier
            df = df.copy()
            df['source_folder'] = benchmark_dir.name
            
            if method_name in all_series_data:
                # Concatenate if method already exists
                all_series_data[method_name] = pd.concat(
                    [all_series_data[method_name], df],
                    ignore_index=True
                )
            else:
                all_series_data[method_name] = df
    
    logger.info(f"Loaded series data for {len(all_series_data)} methods from {len(benchmark_dirs)} folders")
    
    return all_series_data


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
        use_latest: If True and auto_scan, use only latest folder; 
                   If False and auto_scan, load and concatenate all folders
        
    Returns:
        Tuple of:
        - methods_summary_df: Overall method comparison (concatenated if multiple folders)
        - detailed_results_df: Per-dataset, per-method results (concatenated if multiple folders)
        - series_data_dict: Dict mapping method_name -> per-series DataFrame (concatenated if multiple folders)
        - actual_benchmark_dir: Path to the benchmark folder used (or first folder if multiple)
        
    Note:
        When loading multiple folders (use_latest=False), all DataFrames include a 'source_folder' 
        column indicating which benchmark folder each row came from.
    """
    # Resolve benchmark directory/directories
    resolved_dirs = _resolve_benchmark_dirs(benchmark_dir, auto_scan, base_dir, use_latest)
    
    # Validate all directories
    for resolved_dir in resolved_dirs:
        _validate_benchmark_dir(resolved_dir)
    
    # Load summary files (concatenate if multiple folders)
    if len(resolved_dirs) == 1:
        methods_summary_df, detailed_results_df = _load_summary_files(resolved_dirs[0])
    else:
        methods_summary_df, detailed_results_df = _load_and_concatenate_summary_files(resolved_dirs)
    
    # Load per-series data (concatenate if multiple folders)
    if len(resolved_dirs) == 1:
        series_data_dict = _load_series_data(resolved_dirs[0])
    else:
        series_data_dict = _load_and_concatenate_series_data(resolved_dirs)
    
    # Return first directory as the "actual" benchmark_dir for backward compatibility
    return methods_summary_df, detailed_results_df, series_data_dict, resolved_dirs[0]


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

