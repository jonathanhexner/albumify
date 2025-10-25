from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import yaml

from .config import get_global_config
from sim_bench.datasets import load_dataset


def get_metrics_path(method: str, experiment_dir: Optional[Path] = None) -> Path:
    """Get path to metrics.csv for a method."""
    exp = _resolve_experiment_dir(experiment_dir)
    return exp / method / "metrics.csv"


def get_per_query_path(method: str, experiment_dir: Optional[Path] = None) -> Path:
    """Get path to per_query.csv for a method."""
    exp = _resolve_experiment_dir(experiment_dir)
    return exp / method / "per_query.csv"


def get_rankings_path(method: str, experiment_dir: Optional[Path] = None) -> Path:
    """Get path to rankings.csv for a method."""
    exp = _resolve_experiment_dir(experiment_dir)
    return exp / method / "rankings.csv"


def load_metrics(method: str, experiment_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load metrics.csv for a method."""
    path = get_metrics_path(method, experiment_dir)
    return pd.read_csv(path)


def load_per_query(method: str, experiment_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load per_query.csv for a method."""
    path = get_per_query_path(method, experiment_dir)
    return pd.read_csv(path)


def load_rankings(method: str, experiment_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load rankings.csv with backward compatibility for both old and new formats."""
    path = get_rankings_path(method, experiment_dir)
    df = pd.read_csv(path)
    
    # Check if using new format (filenames) and convert to old format for compatibility
    if 'query_filename' in df.columns:
        print(f"[INFO] Converting rankings from filenames to indices for compatibility...")
        # Ensure experiment_dir is resolved
        if experiment_dir is None:
            try:
                experiment_dir = _resolve_experiment_dir(None)
            except ValueError:
                # If global config is not set, try to detect from the rankings file path
                experiment_dir = path.parent.parent
                print(f"[INFO] Using experiment directory: {experiment_dir}")
        return _convert_rankings_to_indices(df, method, experiment_dir)
    
    return df


def _convert_rankings_to_indices(
    rankings_df: pd.DataFrame, 
    method: str, 
    experiment_dir: Path
) -> pd.DataFrame:
    """Convert rankings from filenames back to indices for backward compatibility."""
    
    # Load dataset to get image mappings
    dataset_name = _detect_dataset_from_experiment(experiment_dir)
    config_file = Path(__file__).parent.parent.parent / "configs" / f"dataset.{dataset_name}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    from sim_bench.datasets import load_dataset
    dataset = load_dataset(dataset_name, dataset_config)
    images = dataset.get_images()
    
    # Create filename to index mapping
    filename_to_idx = {Path(img).name: idx for idx, img in enumerate(images)}
    
    # Convert filenames back to indices
    df = rankings_df.copy()
    df['query_idx'] = df['query_filename'].apply(lambda f: filename_to_idx[f])
    df['result_idx'] = df['result_filename'].apply(lambda f: filename_to_idx[f])
    
    # Drop filename columns and reorder
    df = df.drop(columns=['query_filename', 'result_filename'])
    df = df[['query_idx', 'rank', 'result_idx', 'distance']]
    
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


def build_image_index_to_path(per_query_df: pd.DataFrame) -> Dict[int, str]:
    """
    Build a mapping from image index (query_idx) to absolute image path using per_query.csv.
    Note: This covers only query images by index; non-query gallery images may need a dataset map.
    """
    mapping: Dict[int, str] = {}
    for _, row in per_query_df.iterrows():
        mapping[int(row["query_idx"])] = str(row["query_path"])
    return mapping


def build_index_to_path_via_dataset(dataset_name: str, dataset_config: Dict) -> Dict[int, str]:
    """
    Build index->path mapping by instantiating the dataset via the dataset factory.
    Respects the dataset's own ordering and any sampling previously applied if used
    within analysis runs that load sampled configs.
    """
    ds = load_dataset(dataset_name, dataset_config)
    # Ensure data is loaded
    if ds.images is None:
        ds.load_data()
    return ds.get_index_to_path()


def load_enriched_per_query(
    method: str,
    k_values: Optional[List[int]] = None,
    experiment_dir: Optional[Path] = None,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Load enriched per-query data with additional metrics (recall@k, precision@k).
    Results are cached to avoid recomputation.
    
    Args:
        method: Method name
        k_values: List of k values for recall@k (default: [1, 2, 3, 4, 5])
        experiment_dir: Experiment directory (uses global config if None)
        force_recompute: If True, ignore cache and recompute
    
    Returns:
        DataFrame with enriched metrics
    """
    from .metrics import compute_enriched_per_query
    
    if k_values is None:
        k_values = [1, 2, 3, 4, 5]
    
    exp = _resolve_experiment_dir(experiment_dir)
    cache_dir = exp / "enriched_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache key based on k_values
    k_str = "_".join(map(str, sorted(k_values)))
    cache_file = cache_dir / f"{method}_enriched_k{k_str}.csv"
    
    if cache_file.exists() and not force_recompute:
        return pd.read_csv(cache_file)
    
    # Compute and cache
    df_enriched = compute_enriched_per_query(method, k_values, experiment_dir=exp)
    df_enriched.to_csv(cache_file, index=False)
    return df_enriched


def _resolve_experiment_dir(experiment_dir: Optional[Path]) -> Path:
    if experiment_dir is not None:
        return Path(experiment_dir)
    cfg = get_global_config()
    if cfg.experiment_dir is not None:
        return Path(cfg.experiment_dir)
    raise ValueError("experiment_dir not provided and no GlobalAnalysisConfig with experiment_dir is set")


