"""
Utilities for loading and comparing results across multiple experiments, datasets, and methods.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json
import warnings
from .io import load_metrics, load_per_query


def detect_dataset_from_manifest(method_dir: Path) -> Optional[str]:
    """
    Detect dataset name from manifest.json in method directory.

    For new manifests: reads dataset.name from JSON
    For old manifests (no dataset field): infers from parent directory name

    Args:
        method_dir: Path to method directory

    Returns:
        Dataset name or None if not found
    """
    manifest_file = method_dir / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)

            # New format: has dataset.name field
            if 'dataset' in manifest and 'name' in manifest['dataset']:
                return manifest['dataset']['name']

            # Old format: infer from parent directory name
            # e.g., "methods_sift_resnet_emd_datasets_holidays" -> "holidays"
            parent_name = method_dir.parent.name
            if 'datasets_' in parent_name:
                # Extract dataset name after "datasets_" prefix
                dataset_part = parent_name.split('datasets_')[-1]
                # Handle multiple datasets separated by underscore (take first one)
                dataset_name = dataset_part.split('_')[0]
                return dataset_name

        except Exception:
            pass
    return None


def scan_experiment_directories(base_dir: Path, verbose: bool = True) -> List[Dict[str, any]]:
    """
    Recursively scan for experiment directories and extract method/dataset info.

    Args:
        base_dir: Top-level directory to scan
        verbose: Print scanning progress

    Returns:
        List of dicts with keys: experiment_dir, method, dataset, run_name
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    results = []
    seen_combinations = set()  # Track (method, dataset) to detect duplicates

    if verbose:
        print(f"Scanning: {base_dir}")

    # Look for directories that contain method subdirectories
    for potential_run_dir in base_dir.rglob("*"):
        if not potential_run_dir.is_dir():
            continue

        # Check if this directory contains method subdirectories with metrics.csv
        method_dirs = [d for d in potential_run_dir.iterdir()
                      if d.is_dir() and (d / "metrics.csv").exists()]

        if method_dirs:
            run_name = potential_run_dir.name

            for method_dir in method_dirs:
                method_name = method_dir.name
                dataset_name = detect_dataset_from_manifest(method_dir)

                if dataset_name:
                    combination = (method_name, dataset_name)

                    if combination in seen_combinations:
                        warnings.warn(
                            f"\n{'='*80}\n"
                            f"[!] WARNING: DUPLICATE DETECTED!\n"
                            f"{'='*80}\n"
                            f"Method: {method_name}\n"
                            f"Dataset: {dataset_name}\n"
                            f"Already found in previous run, skipping: {potential_run_dir}\n"
                            f"{'='*80}\n",
                            UserWarning,
                            stacklevel=2
                        )
                        continue

                    seen_combinations.add(combination)

                    results.append({
                        'experiment_dir': potential_run_dir,
                        'method': method_name,
                        'dataset': dataset_name,
                        'run_name': run_name
                    })

                    if verbose:
                        print(f"  Found: {method_name} on {dataset_name} in {run_name}")

    if verbose:
        print(f"\n[OK] Found {len(results)} method-dataset combinations")

        # Summary by dataset
        datasets = set(r['dataset'] for r in results)
        for dataset in sorted(datasets):
            methods = [r['method'] for r in results if r['dataset'] == dataset]
            print(f"  {dataset}: {len(methods)} methods ({', '.join(sorted(methods))})")

    return results


def load_multi_experiment_metrics(
    experiment_infos: List[Dict[str, any]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load metrics from multiple experiments into a single DataFrame.

    Args:
        experiment_infos: List of experiment info dicts from scan_experiment_directories
        verbose: Print loading progress

    Returns:
        DataFrame with columns: method, dataset, run_name, and all metric columns
    """
    all_data = []

    for info in experiment_infos:
        try:
            df = load_metrics(info['method'], info['experiment_dir'])

            # Add metadata columns
            df['method'] = info['method']
            df['dataset'] = info['dataset']
            df['run_name'] = info['run_name']

            # Handle backward compatibility for OLD metric names
            if 'map_full' in df.columns and 'map' not in df.columns:
                df['map'] = df['map_full']
                df = df.drop(columns=['map_full'])  # Remove old name
            if 'prec@10' in df.columns and 'precision@10' not in df.columns:
                df['precision@10'] = df['prec@10']
                df = df.drop(columns=['prec@10'])  # Remove old name

            all_data.append(df)

            if verbose:
                print(f"[OK] Loaded: {info['method']:15} on {info['dataset']:10} - {len(df.columns)} metrics")

        except Exception as e:
            warnings.warn(f"Failed to load {info['method']} on {info['dataset']}: {e}")

    if not all_data:
        raise ValueError("No metrics data could be loaded")

    # Concatenate with outer join to preserve all columns (some datasets have different metrics)
    result_df = pd.concat(all_data, ignore_index=True, sort=False)

    if verbose:
        print(f"\n[OK] Loaded {len(result_df)} total records")
        print(f"  Available metrics: {[c for c in result_df.columns if c not in ['method', 'dataset', 'run_name', 'created_at', 'num_queries', 'num_gallery']]}")

    return result_df


def load_multi_experiment_per_query(
    experiment_infos: List[Dict[str, any]],
    verbose: bool = True
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Load per-query data from multiple experiments.

    Args:
        experiment_infos: List of experiment info dicts from scan_experiment_directories
        verbose: Print loading progress

    Returns:
        Dict mapping (method, dataset) tuples to per-query DataFrames
    """
    per_query_data = {}

    for info in experiment_infos:
        try:
            df = load_per_query(info['method'], info['experiment_dir'])
            key = (info['method'], info['dataset'])
            per_query_data[key] = df

            if verbose:
                print(f"[OK] Loaded per-query: {info['method']} on {info['dataset']} ({len(df)} queries)")

        except Exception as e:
            warnings.warn(f"Failed to load per-query for {info['method']} on {info['dataset']}: {e}")

    return per_query_data


def load_experiments(
    base_dir: Optional[Path] = None,
    experiment_dirs: Optional[List[Path]] = None,
    auto_scan: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame], List[Dict[str, any]]]:
    """
    Unified function to load experiment data with auto-scan or manual specification.

    Args:
        base_dir: Base directory for auto-scanning (used if auto_scan=True)
        experiment_dirs: List of specific experiment directories (used if auto_scan=False)
        auto_scan: If True, recursively scan base_dir; if False, use experiment_dirs list
        verbose: Print loading progress

    Returns:
        Tuple of (metrics_df, per_query_dict, experiment_infos)
    """
    if auto_scan:
        if base_dir is None:
            raise ValueError("base_dir must be provided when auto_scan=True")
        experiment_infos = scan_experiment_directories(base_dir, verbose=verbose)
    else:
        if experiment_dirs is None:
            raise ValueError("experiment_dirs must be provided when auto_scan=False")

        # Build experiment_infos from manual list
        experiment_infos = []
        for exp_dir in experiment_dirs:
            if not exp_dir.exists():
                warnings.warn(f"Experiment directory not found: {exp_dir}")
                continue

            # Find method directories
            method_dirs = [d for d in exp_dir.iterdir()
                          if d.is_dir() and (d / "metrics.csv").exists()]

            for method_dir in method_dirs:
                method_name = method_dir.name
                dataset_name = detect_dataset_from_manifest(method_dir)

                if dataset_name:
                    experiment_infos.append({
                        'experiment_dir': exp_dir,
                        'method': method_name,
                        'dataset': dataset_name,
                        'run_name': exp_dir.name
                    })

    if not experiment_infos:
        raise ValueError("No valid experiments found")

    # Load metrics and per-query data
    metrics_df = load_multi_experiment_metrics(experiment_infos, verbose=verbose)
    per_query_dict = load_multi_experiment_per_query(experiment_infos, verbose=verbose)

    return metrics_df, per_query_dict, experiment_infos


def merge_per_query_metrics(
    per_query_dict: Dict[Tuple[str, str], pd.DataFrame],
    methods: List[str],
    dataset: str,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Merge per-query metrics from multiple methods into a single DataFrame.

    Creates a wide-format DataFrame with one row per query and columns for each
    method-metric combination (e.g., 'ap@10_deep', 'ap@10_dinov2').

    Args:
        per_query_dict: Dictionary mapping (method, dataset) to per-query DataFrames
        methods: List of method names to merge
        dataset: Dataset name to use
        metrics: List of metric column names to merge (e.g., ['ap@10', 'recall@10'])

    Returns:
        DataFrame with columns:
        - query_idx, query_path, group_id (from first method)
        - {metric}_{method} for each metric-method combination

    Example:
        >>> merged = merge_per_query_metrics(
        ...     per_query_dict,
        ...     methods=['deep', 'dinov2', 'openclip'],
        ...     dataset='holidays',
        ...     metrics=['ap@10', 'recall@10']
        ... )
        >>> merged.columns
        ['query_idx', 'query_path', 'group_id',
         'ap@10_deep', 'recall@10_deep', 'ap@10_dinov2', 'recall@10_dinov2', ...]
    """
    df_merged = pd.DataFrame()

    for n_method, method in enumerate(methods):
        # Get per-query data for this method
        key = (method, dataset)
        if key not in per_query_dict:
            warnings.warn(f"Method-dataset combination not found: {key}")
            continue

        df_method = per_query_dict[key]

        # Rename metric columns to include method suffix
        metrics_method = [f"{metric}_{method}" for metric in metrics]
        rename_map = dict(zip(metrics, metrics_method))

        if n_method == 0:
            # First method: include all metadata columns + metrics
            cols = ['query_idx', 'query_path', 'group_id'] + metrics
            df_merged = df_method[cols].copy()
            df_merged.rename(columns=rename_map, inplace=True)
        else:
            # Subsequent methods: only merge metrics
            cols = ['query_idx'] + metrics
            df_query = df_method[cols].copy()
            df_query.rename(columns=rename_map, inplace=True)
            df_merged = df_merged.merge(df_query, on='query_idx', how='left')

    return df_merged


def find_method_wins(
    df_merged: pd.DataFrame,
    winning_method: str,
    competing_methods: List[str],
    metric: str = 'ap@10',
    threshold_high: float = 0.9,
    threshold_low: float = 0.3,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Find queries where one method excels but competitors fail.

    Args:
        df_merged: Merged per-query DataFrame from merge_per_query_metrics()
        winning_method: Method that should perform well
        competing_methods: List of methods that should perform poorly
        metric: Base metric name (without method suffix, e.g., 'ap@10')
        threshold_high: Minimum score for winning method
        threshold_low: Maximum score for competing methods
        top_n: Number of queries to return

    Returns:
        DataFrame with top_n queries sorted by score difference, including:
        - Original columns from df_merged
        - score_diff: winning_score - mean(competing_scores)

    Example:
        >>> wins = find_method_wins(
        ...     df_merged,
        ...     winning_method='deep',
        ...     competing_methods=['dinov2', 'openclip'],
        ...     metric='ap@10',
        ...     threshold_high=0.9,
        ...     threshold_low=0.3,
        ...     top_n=3
        ... )
        >>> wins[['query_idx', 'ap@10_deep', 'ap@10_dinov2', 'ap@10_openclip', 'score_diff']]
    """
    # Build column names
    winning_col = f"{metric}_{winning_method}"
    competing_cols = [f"{metric}_{method}" for method in competing_methods]

    # Check columns exist
    missing_cols = [c for c in [winning_col] + competing_cols if c not in df_merged.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_merged: {missing_cols}")

    # Filter: winning method is high, all competitors are low
    mask = (df_merged[winning_col] >= threshold_high)
    for competing_col in competing_cols:
        mask &= (df_merged[competing_col] <= threshold_low)

    df_wins = df_merged[mask].copy()

    if len(df_wins) == 0:
        warnings.warn(
            f"No queries found where {winning_method} wins. "
            f"Try lowering threshold_high or raising threshold_low."
        )
        return df_wins

    # Calculate score difference (winning - average of competitors)
    df_wins['score_diff'] = df_wins[winning_col] - df_wins[competing_cols].mean(axis=1)

    # Sort by score difference and take top_n
    df_wins = df_wins.sort_values('score_diff', ascending=False).head(top_n)

    return df_wins


def find_all_method_wins(
    df_merged: pd.DataFrame,
    methods: List[str],
    metric: str = 'ap@10',
    threshold_high: float = 0.9,
    threshold_low: float = 0.3,
    top_n: int = 3,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Find winning queries for each method (where that method excels but others fail).

    Args:
        df_merged: Merged per-query DataFrame from merge_per_query_metrics()
        methods: List of all methods to compare
        metric: Base metric name (without method suffix, e.g., 'ap@10')
        threshold_high: Minimum score for winning method
        threshold_low: Maximum score for competing methods
        top_n: Number of queries to return per method
        verbose: Print summary of findings

    Returns:
        Dictionary mapping method -> DataFrame of winning queries

    Example:
        >>> all_wins = find_all_method_wins(
        ...     df_merged,
        ...     methods=['deep', 'dinov2', 'openclip'],
        ...     metric='ap@10',
        ...     top_n=3
        ... )
        >>> for method, wins in all_wins.items():
        ...     print(f"{method}: {len(wins)} winning queries")
    """
    all_wins = {}

    for method in methods:
        competing_methods = [m for m in methods if m != method]

        df_wins = find_method_wins(
            df_merged,
            winning_method=method,
            competing_methods=competing_methods,
            metric=metric,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            top_n=top_n
        )

        all_wins[method] = df_wins

        if verbose:
            print(f"\n{method.upper()} wins ({len(df_wins)} queries):")
            if len(df_wins) > 0:
                method_col = f"{metric}_{method}"
                competing_cols = [f"{metric}_{m}" for m in competing_methods]
                display_cols = ['query_idx', method_col] + competing_cols
                print(df_wins[display_cols].to_string(index=False))
            else:
                print("  (none found)")

    return all_wins
