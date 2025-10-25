"""
Enhanced IO utilities that work with image filenames instead of array indices.

This provides a cleaner, more intuitive interface for analysis.
"""

from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, List
import yaml
from sim_bench.datasets import load_dataset


def load_rankings_by_filename(method: str, experiment_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load rankings CSV with filename-based columns.
    
    Args:
        method: Method name
        experiment_dir: Experiment directory
    
    Returns:
        DataFrame with columns: query_filename, rank, result_filename, distance
    """
    if experiment_dir is None:
        from .config import get_global_config
        if cfg := get_global_config():
            experiment_dir = Path(cfg.experiment_dir)
        else:
            raise ValueError("experiment_dir not provided and no GlobalAnalysisConfig set")
    
    rankings_file = experiment_dir / method / "rankings.csv"
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_file}")
    
    df = pd.read_csv(rankings_file)
    
    # Check if already using filenames
    if 'query_filename' in df.columns:
        return df
    
    # Convert from old format (indices) to new format (filenames)
    print(f"[INFO] Converting rankings from indices to filenames...")
    return _convert_rankings_to_filenames(df, method, experiment_dir)


def _convert_rankings_to_filenames(
    rankings_df: pd.DataFrame, 
    method: str, 
    experiment_dir: Path
) -> pd.DataFrame:
    """Convert rankings from array indices to filenames."""
    
    # Load dataset to get image mappings
    dataset_name = _detect_dataset_from_experiment(experiment_dir)
    config_file = Path(__file__).parent.parent.parent / "configs" / f"dataset.{dataset_name}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    dataset = load_dataset(dataset_name, dataset_config)
    images = dataset.get_images()
    
    # Convert indices to filenames
    df = rankings_df.copy()
    df['query_filename'] = df['query_idx'].apply(lambda idx: Path(images[idx]).name)
    df['result_filename'] = df['result_idx'].apply(lambda idx: Path(images[idx]).name)
    
    # Drop old columns and reorder
    df = df.drop(columns=['query_idx', 'result_idx'])
    df = df[['query_filename', 'rank', 'result_filename', 'distance']]
    
    return df


def _detect_dataset_from_experiment(experiment_dir: Path) -> str:
    """Detect dataset name from experiment directory structure."""
    # Try to find manifest.json with dataset info
    for method_dir in experiment_dir.iterdir():
        if method_dir.is_dir():
            manifest_file = method_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    import json
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    if 'dataset' in manifest and 'name' in manifest['dataset']:
                        return manifest['dataset']['name']
                except:
                    continue
    
    # Fallback: try to detect from per_query.csv structure
    for method_dir in experiment_dir.iterdir():
        if method_dir.is_dir():
            per_query_file = method_dir / "per_query.csv"
            if per_query_file.exists():
                df = pd.read_csv(per_query_file, nrows=5)
                if 'group_id' in df.columns:
                    # Check if it looks like Holidays (high group IDs) or UKBench (low group IDs)
                    sample_groups = df['group_id'].head().tolist()
                    if any(g > 1000 for g in sample_groups):
                        return 'holidays'
                    else:
                        return 'ukbench'
    
    # Default fallback
    return 'holidays'


def get_rankings_for_query(
    query_filename: str,
    method: str,
    experiment_dir: Optional[Path] = None,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Get rankings for a specific query image by filename.
    
    Args:
        query_filename: Image filename (e.g., '111001.jpg')
        method: Method name
        experiment_dir: Experiment directory
        top_k: Number of top results to return
    
    Returns:
        DataFrame with top-k results for the query
    """
    rankings_df = load_rankings_by_filename(method, experiment_dir)
    
    # Filter for this query
    query_results = rankings_df[rankings_df['query_filename'] == query_filename]
    
    # Sort by rank and limit to top_k
    query_results = query_results.sort_values('rank').head(top_k)
    
    return query_results


def get_queries_in_group(
    group_id: int,
    method: str,
    experiment_dir: Optional[Path] = None
) -> List[str]:
    """
    Get all query filenames in a specific group.
    
    Args:
        group_id: Group ID
        method: Method name
        experiment_dir: Experiment directory
    
    Returns:
        List of query filenames in the group
    """
    from .io import load_per_query
    
    per_query_df = load_per_query(method, experiment_dir)
    group_queries = per_query_df[per_query_df['group_id'] == group_id]
    
    # Extract filenames from paths
    filenames = []
    for _, row in group_queries.iterrows():
        filename = Path(row['query_path']).name
        filenames.append(filename)
    
    return filenames


def analyze_query_performance(
    query_filename: str,
    method: str,
    experiment_dir: Optional[Path] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Analyze performance for a specific query.
    
    Args:
        query_filename: Image filename
        method: Method name
        experiment_dir: Experiment directory
        top_k: Number of top results to analyze
    
    Returns:
        Dictionary with performance metrics
    """
    from .io import load_per_query
    
    # Get rankings for this query
    rankings = get_rankings_for_query(query_filename, method, experiment_dir, top_k)
    
    # Get query info from per_query.csv
    per_query_df = load_per_query(method, experiment_dir)
    query_row = per_query_df[per_query_df['query_path'].str.contains(query_filename)]
    
    if len(query_row) == 0:
        raise ValueError(f"Query {query_filename} not found in per_query.csv")
    
    query_row = query_row.iloc[0]
    query_group = query_row['group_id']
    num_relevant = query_row.get('num_relevant', 0)
    
    # Count relevant results in top-k
    relevant_in_topk = 0
    for _, result in rankings.iterrows():
        result_filename = result['result_filename']
        # Check if result is in same group (relevant)
        result_group = _get_group_from_filename(result_filename, experiment_dir)
        if result_group == query_group:
            relevant_in_topk += 1
    
    # Calculate metrics
    precision_at_k = relevant_in_topk / min(top_k, len(rankings))
    recall_at_k = relevant_in_topk / num_relevant if num_relevant > 0 else 0.0
    
    return {
        'query_filename': query_filename,
        'query_group': query_group,
        'num_relevant': num_relevant,
        'relevant_in_topk': relevant_in_topk,
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'top_results': rankings[['result_filename', 'distance']].to_dict('records')
    }


def _get_group_from_filename(filename: str, experiment_dir: Path) -> int:
    """Get group ID from image filename."""
    # Extract image ID from filename
    image_id = int(Path(filename).stem)
    
    # Use same logic as dataset loading
    return image_id // 100


# Backward compatibility functions
def load_rankings(method: str, experiment_dir: Optional[Path] = None) -> pd.DataFrame:
    """Backward compatibility wrapper."""
    return load_rankings_by_filename(method, experiment_dir)

