from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd

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
    """Load rankings.csv for a method."""
    path = get_rankings_path(method, experiment_dir)
    return pd.read_csv(path)


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


