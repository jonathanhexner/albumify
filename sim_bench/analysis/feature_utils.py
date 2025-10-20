"""
Utilities for loading and analyzing feature representations.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def list_feature_caches(cache_dir: Path = Path("artifacts/feature_cache")) -> pd.DataFrame:
    """
    List all available feature cache files.
    
    Args:
        cache_dir: Directory containing feature caches
        
    Returns:
        DataFrame with cache file information (filename, method, size, etc.)
    """
    cache_files = list(cache_dir.glob("*.pkl"))
    
    data = []
    for cache_file in cache_files:
        # Extract method name from filename (format: method_hash.pkl)
        method_name = cache_file.stem.split('_')[0]
        file_size_mb = cache_file.stat().st_size / (1024 * 1024)
        
        data.append({
            'filename': cache_file.name,
            'method': method_name,
            'size_mb': file_size_mb,
            'path': str(cache_file)
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(['method', 'filename'])
    return df


def load_features_from_cache(
    cache_file: Path,
    return_metadata: bool = False
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load features from a cache file.
    
    Args:
        cache_file: Path to .pkl cache file
        return_metadata: If True, return (features, metadata_dict)
        
    Returns:
        Feature matrix [n_images, feature_dim] or (features, metadata)
    """
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        features = cached_data['features']
        
        if return_metadata:
            metadata = {
                'method_name': cached_data.get('method_name'),
                'method_config': cached_data.get('method_config'),
                'image_paths': cached_data.get('image_paths'),
                'n_images': len(cached_data.get('image_paths', [])),
                'feature_dim': features.shape[1] if len(features.shape) > 1 else features.shape[0]
            }
            return features, metadata
        
        return features
        
    except Exception as e:
        logger.error(f"Failed to load cache file {cache_file}: {e}")
        raise


def find_cache_for_method(
    method_name: str,
    cache_dir: Path = Path("artifacts/feature_cache"),
    return_all: bool = False
) -> Optional[Path] | List[Path]:
    """
    Find cache file(s) for a specific method.
    
    Args:
        method_name: Name of the method (e.g., 'deep', 'chi_square')
        cache_dir: Directory containing feature caches
        return_all: If True, return list of all matches; if False, return most recent
        
    Returns:
        Path to cache file or list of paths (if return_all=True)
    """
    pattern = f"{method_name}_*.pkl"
    cache_files = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not cache_files:
        return None if not return_all else []
    
    if return_all:
        return cache_files
    
    return cache_files[0]  # Most recent


def get_features_for_images(
    image_paths: List[str] | List[Path],
    cache_file: Path,
    return_indices: bool = False
) -> np.ndarray | Tuple[np.ndarray, List[int]]:
    """
    Extract features for specific images from a cache file.
    
    Args:
        image_paths: List of image paths to query (can be full paths or just filenames)
        cache_file: Path to feature cache file
        return_indices: If True, also return the indices in the original cache
        
    Returns:
        Feature matrix for queried images [n_query, feature_dim]
        If return_indices=True: (features, indices)
        
    Raises:
        ValueError: If any of the queried images are not found in cache
    """
    # Load full cache
    features, metadata = load_features_from_cache(cache_file, return_metadata=True)
    cached_paths = metadata['image_paths']
    
    # Normalize query paths to just filenames for matching
    query_filenames = [Path(p).name for p in image_paths]
    cached_filenames = [Path(p).name for p in cached_paths]
    
    # Find indices of queried images
    indices = []
    missing = []
    
    for i, query_filename in enumerate(query_filenames):
        try:
            idx = cached_filenames.index(query_filename)
            indices.append(idx)
        except ValueError:
            missing.append(image_paths[i])
    
    if missing:
        raise ValueError(
            f"Could not find {len(missing)} image(s) in cache:\n" +
            "\n".join([f"  - {p}" for p in missing[:5]]) +
            (f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )
    
    # Extract features for queried images
    queried_features = features[indices]
    
    if return_indices:
        return queried_features, indices
    
    return queried_features


def get_features_by_index(
    indices: List[int] | int,
    cache_file: Path
) -> np.ndarray:
    """
    Extract features by index (e.g., query_idx from per_query.csv).
    
    Args:
        indices: Single index or list of indices to extract
        cache_file: Path to feature cache file
        
    Returns:
        Feature matrix for specified indices [n_indices, feature_dim]
        
    Raises:
        IndexError: If any index is out of bounds
    """
    # Load full cache
    features, metadata = load_features_from_cache(cache_file, return_metadata=True)
    n_images = metadata['n_images']
    
    # Handle single index
    if isinstance(indices, int):
        indices = [indices]
    
    # Validate indices
    for idx in indices:
        if idx < 0 or idx >= n_images:
            raise IndexError(
                f"Index {idx} out of bounds for cache with {n_images} images"
            )
    
    return features[indices]


def get_image_path_by_index(
    index: int,
    cache_file: Path
) -> str:
    """
    Get image path by index.
    
    Args:
        index: Image index
        cache_file: Path to feature cache file
        
    Returns:
        Image path at the specified index
        
    Raises:
        IndexError: If index is out of bounds
    """
    _, metadata = load_features_from_cache(cache_file, return_metadata=True)
    image_paths = metadata['image_paths']
    
    if index < 0 or index >= len(image_paths):
        raise IndexError(
            f"Index {index} out of bounds for cache with {len(image_paths)} images"
        )
    
    return image_paths[index]


def search_images_by_filename(
    search_term: str,
    cache_file: Path,
    case_sensitive: bool = False
) -> Tuple[List[str], List[int]]:
    """
    Search for images in cache by filename pattern.
    
    Args:
        search_term: Search string to match in filenames
        cache_file: Path to feature cache file
        case_sensitive: Whether search is case-sensitive
        
    Returns:
        Tuple of (matching_paths, matching_indices)
    """
    # Load metadata
    _, metadata = load_features_from_cache(cache_file, return_metadata=True)
    cached_paths = metadata['image_paths']
    
    # Search
    matches = []
    indices = []
    
    search_lower = search_term if case_sensitive else search_term.lower()
    
    for i, path in enumerate(cached_paths):
        filename = Path(path).name
        compare_name = filename if case_sensitive else filename.lower()
        
        if search_lower in compare_name:
            matches.append(path)
            indices.append(i)
    
    return matches, indices


def get_query_feature_matrix(
    query_indices: List[int],
    cache_file: Path
) -> np.ndarray:
    """
    Get feature matrix for queries in shape [feature_dim, n_queries].
    
    This is useful for analyzing feature patterns across queries, where each
    column represents a query and each row represents a feature dimension.
    
    Args:
        query_indices: List of query indices
        cache_file: Path to feature cache file
        
    Returns:
        Feature matrix [feature_dim, n_queries] (transposed)
    """
    features = get_features_by_index(query_indices, cache_file)
    # Transpose to [feature_dim, n_queries]
    return features.T


def analyze_within_group_feature_diversity(
    query_indices: List[int],
    query_group_ids: List[int],
    cache_file: Path,
    top_k: int = 20,
    metrics: List[str] = ['variance', 'range']
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze which feature dimensions have highest diversity within each group.
    
    High diversity within a group indicates features that are NOT consistent
    for images in the same group - these may be causing retrieval errors.
    
    Args:
        query_indices: List of query indices
        query_group_ids: Group ID for each query
        cache_file: Path to feature cache file
        top_k: Number of top diverse features to return per group
        metrics: Which diversity metrics to compute ['variance', 'range', 'std', 'iqr']
        
    Returns:
        Dictionary mapping group_id to analysis results:
            {
                group_id: {
                    'query_indices_in_group': [...],
                    'n_images': int,
                    'feature_matrix': [feature_dim, n_in_group],
                    'variances': [feature_dim],
                    'ranges': [feature_dim],
                    'top_diverse_dims_by_metric': {metric: [top_k]},
                    'top_diverse_values_by_metric': {metric: [top_k]},
                    'mean_variance': float,
                    'mean_range': float
                }
            }
    """
    # Get features for all queries
    query_features = get_features_by_index(query_indices, cache_file)
    
    # Group queries by group_id
    from collections import defaultdict
    groups = defaultdict(list)
    for i, gid in enumerate(query_group_ids):
        groups[gid].append(i)
    
    results = {}
    
    for gid, indices_in_group in groups.items():
        if len(indices_in_group) < 2:
            # Need at least 2 images to compute diversity
            continue
        
        # Get features for this group [n_in_group, feature_dim]
        group_features = query_features[indices_in_group]
        
        # Transpose to [feature_dim, n_in_group]
        feature_matrix = group_features.T
        
        result = {
            'query_indices_in_group': [query_indices[i] for i in indices_in_group],
            'n_images': len(indices_in_group),
            'feature_matrix': feature_matrix,
            'top_diverse_dims_by_metric': {},
            'top_diverse_values_by_metric': {}
        }
        
        # Compute requested metrics
        if 'variance' in metrics:
            variances = np.var(feature_matrix, axis=1)
            top_dims = np.argsort(variances)[-top_k:][::-1]
            result['variances'] = variances
            result['top_diverse_dims_by_metric']['variance'] = top_dims
            result['top_diverse_values_by_metric']['variance'] = variances[top_dims]
            result['mean_variance'] = variances.mean()
        
        if 'range' in metrics:
            ranges = np.ptp(feature_matrix, axis=1)  # peak-to-peak (max - min)
            top_dims = np.argsort(ranges)[-top_k:][::-1]
            result['ranges'] = ranges
            result['top_diverse_dims_by_metric']['range'] = top_dims
            result['top_diverse_values_by_metric']['range'] = ranges[top_dims]
            result['mean_range'] = ranges.mean()
        
        if 'std' in metrics:
            stds = np.std(feature_matrix, axis=1)
            top_dims = np.argsort(stds)[-top_k:][::-1]
            result['stds'] = stds
            result['top_diverse_dims_by_metric']['std'] = top_dims
            result['top_diverse_values_by_metric']['std'] = stds[top_dims]
            result['mean_std'] = stds.mean()
        
        if 'iqr' in metrics:
            q75 = np.percentile(feature_matrix, 75, axis=1)
            q25 = np.percentile(feature_matrix, 25, axis=1)
            iqrs = q75 - q25
            top_dims = np.argsort(iqrs)[-top_k:][::-1]
            result['iqrs'] = iqrs
            result['top_diverse_dims_by_metric']['iqr'] = top_dims
            result['top_diverse_values_by_metric']['iqr'] = iqrs[top_dims]
            result['mean_iqr'] = iqrs.mean()
        
        # Keep backward compatibility: default to variance
        if 'variance' in metrics:
            result['top_diverse_dims'] = result['top_diverse_dims_by_metric']['variance']
            result['top_diverse_variances'] = result['top_diverse_values_by_metric']['variance']
            result['max_variance'] = result['variances'].max()
        
        results[gid] = result
    
    return results


def analyze_feature_discriminability(
    query_indices: List[int],
    query_group_ids: List[int],
    cache_file: Path,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Analyze which features are good at discriminating between groups.
    
    Good features have:
    - Low within-group variance (similar images have similar values)
    - High between-group variance (different images have different values)
    
    This is similar to the Fisher criterion used in feature selection.
    
    Args:
        query_indices: List of query indices
        query_group_ids: Group ID for each query
        cache_file: Path to feature cache file
        top_k: Number of top discriminative features to return
        
    Returns:
        Dictionary with:
            'fisher_scores': Fisher criterion for each dimension
            'top_discriminative_dims': Indices of top-k features
            'top_fisher_scores': Fisher scores for top features
            'within_group_variances': Mean within-group variance per dimension
            'between_group_variances': Between-group variance per dimension
    """
    # Get features for all queries
    query_features = get_features_by_index(query_indices, cache_file)
    n_features = query_features.shape[1]
    
    # Group features by group_id
    from collections import defaultdict
    groups = defaultdict(list)
    for i, gid in enumerate(query_group_ids):
        groups[gid].append(i)
    
    # Compute overall mean
    overall_mean = query_features.mean(axis=0)  # [feature_dim]
    
    # Compute within-group and between-group variances
    within_group_var = np.zeros(n_features)
    between_group_var = np.zeros(n_features)
    
    for gid, indices_in_group in groups.items():
        if len(indices_in_group) < 2:
            continue
        
        group_features = query_features[indices_in_group]
        group_mean = group_features.mean(axis=0)
        group_size = len(indices_in_group)
        
        # Within-group variance for this group
        within_var = np.var(group_features, axis=0)
        within_group_var += within_var * group_size
        
        # Between-group variance contribution
        mean_diff = (group_mean - overall_mean) ** 2
        between_group_var += mean_diff * group_size
    
    # Normalize by total number of samples
    n_samples = len(query_indices)
    within_group_var /= n_samples
    between_group_var /= n_samples
    
    # Compute Fisher score (ratio of between to within variance)
    # Add small epsilon to avoid division by zero
    fisher_scores = between_group_var / (within_group_var + 1e-10)
    
    # Find top discriminative features
    top_dims = np.argsort(fisher_scores)[-top_k:][::-1]
    
    return {
        'fisher_scores': fisher_scores,
        'top_discriminative_dims': top_dims,
        'top_fisher_scores': fisher_scores[top_dims],
        'within_group_variances': within_group_var,
        'between_group_variances': between_group_var,
        'mean_fisher_score': fisher_scores.mean(),
        'median_fisher_score': np.median(fisher_scores)
    }


def compute_feature_statistics(features: np.ndarray) -> pd.DataFrame:
    """
    Compute comprehensive statistics for feature matrix.
    
    Args:
        features: Feature matrix [n_images, feature_dim]
        
    Returns:
        DataFrame with per-dimension statistics
    """
    stats = {
        'dimension': list(range(features.shape[1])),
        'mean': features.mean(axis=0),
        'std': features.std(axis=0),
        'min': features.min(axis=0),
        'max': features.max(axis=0),
        'median': np.median(features, axis=0),
        'q25': np.percentile(features, 25, axis=0),
        'q75': np.percentile(features, 75, axis=0),
    }
    
    df = pd.DataFrame(stats)
    
    # Add derived statistics
    df['range'] = df['max'] - df['min']
    df['iqr'] = df['q75'] - df['q25']
    df['coef_var'] = df['std'] / (df['mean'] + 1e-10)  # Avoid division by zero
    
    return df


def compute_feature_correlations(features: np.ndarray, max_dims: int = 100) -> np.ndarray:
    """
    Compute correlation matrix for features (optionally limited to first N dimensions).
    
    Args:
        features: Feature matrix [n_images, feature_dim]
        max_dims: Maximum number of dimensions to compute (for efficiency)
        
    Returns:
        Correlation matrix [max_dims, max_dims]
    """
    if features.shape[1] > max_dims:
        features_subset = features[:, :max_dims]
    else:
        features_subset = features
    
    return np.corrcoef(features_subset.T)


def compute_sparsity_metrics(features: np.ndarray, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Compute sparsity metrics for feature matrix.
    
    Args:
        features: Feature matrix [n_images, feature_dim]
        threshold: Values below this are considered zero
        
    Returns:
        Dictionary with sparsity metrics
    """
    near_zero = np.abs(features) < threshold
    
    return {
        'fraction_zeros': near_zero.sum() / features.size,
        'avg_nonzeros_per_image': (~near_zero).sum(axis=1).mean(),
        'avg_nonzeros_per_dim': (~near_zero).sum(axis=0).mean(),
        'max_nonzeros_per_image': (~near_zero).sum(axis=1).max(),
        'min_nonzeros_per_image': (~near_zero).sum(axis=1).min(),
    }


def compute_distance_statistics(
    features: np.ndarray,
    metric: str = 'euclidean',
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute statistics on pairwise distances.
    
    Args:
        features: Feature matrix [n_images, feature_dim]
        metric: Distance metric ('euclidean', 'cosine', etc.)
        sample_size: If provided, compute on random sample of pairs
        random_seed: Random seed for sampling
        
    Returns:
        Dictionary with distance statistics
    """
    from scipy.spatial.distance import pdist
    
    if sample_size and len(features) > sample_size:
        np.random.seed(random_seed)
        indices = np.random.choice(len(features), size=sample_size, replace=False)
        features_sample = features[indices]
    else:
        features_sample = features
    
    distances = pdist(features_sample, metric=metric)
    
    return {
        'mean_distance': distances.mean(),
        'std_distance': distances.std(),
        'min_distance': distances.min(),
        'max_distance': distances.max(),
        'median_distance': np.median(distances),
        'q25_distance': np.percentile(distances, 25),
        'q75_distance': np.percentile(distances, 75),
    }


def extract_group_labels_from_paths(
    image_paths: List[str],
    dataset_type: str = 'ukbench'
) -> np.ndarray:
    """
    Extract ground-truth group labels from image paths.
    
    Args:
        image_paths: List of image file paths
        dataset_type: 'ukbench' or 'holidays'
        
    Returns:
        Array of group labels
    """
    if dataset_type == 'ukbench':
        # UKBench: groups of 4, extract from filename (e.g., ukbench00000.jpg -> group 0)
        import re
        labels = []
        for path in image_paths:
            match = re.search(r'ukbench(\d+)', path)
            if match:
                image_id = int(match.group(1))
                group_id = image_id // 4
                labels.append(group_id)
            else:
                labels.append(-1)  # Unknown
        return np.array(labels)
    
    elif dataset_type == 'holidays':
        # Holidays: first 2 digits indicate group (e.g., 100000.jpg -> group 10)
        import re
        labels = []
        for path in image_paths:
            match = re.search(r'(\d{2})\d{4}', path)
            if match:
                group_id = int(match.group(1))
                labels.append(group_id)
            else:
                labels.append(-1)  # Unknown
        return np.array(labels)
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def compute_intra_inter_class_distances(
    features: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
    sample_per_class: int = 100
) -> Dict[str, Any]:
    """
    Compute intra-class and inter-class distance statistics.
    
    Args:
        features: Feature matrix [n_images, feature_dim]
        labels: Group labels [n_images]
        metric: Distance metric
        sample_per_class: Max samples per class to compute
        
    Returns:
        Dictionary with intra/inter-class statistics and separability ratio
    """
    from scipy.spatial.distance import cdist
    
    unique_labels = np.unique(labels[labels >= 0])  # Exclude unknown (-1)
    
    intra_distances = []
    inter_distances = []
    
    for label in unique_labels[:50]:  # Limit to 50 classes for efficiency
        # Get samples from this class
        class_mask = labels == label
        class_features = features[class_mask]
        
        if len(class_features) < 2:
            continue
        
        # Sample if too many
        if len(class_features) > sample_per_class:
            indices = np.random.choice(len(class_features), sample_per_class, replace=False)
            class_features = class_features[indices]
        
        # Intra-class distances
        intra_dist = cdist(class_features, class_features, metric=metric)
        intra_distances.extend(intra_dist[np.triu_indices_from(intra_dist, k=1)])
        
        # Inter-class distances (to other classes)
        other_mask = labels != label
        other_features = features[other_mask]
        
        if len(other_features) > sample_per_class:
            indices = np.random.choice(len(other_features), sample_per_class, replace=False)
            other_features = other_features[indices]
        
        inter_dist = cdist(class_features, other_features, metric=metric)
        inter_distances.extend(inter_dist.flatten()[:1000])  # Sample to avoid memory issues
    
    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)
    
    return {
        'intra_mean': intra_distances.mean(),
        'intra_std': intra_distances.std(),
        'inter_mean': inter_distances.mean(),
        'inter_std': inter_distances.std(),
        'separability_ratio': inter_distances.mean() / (intra_distances.mean() + 1e-10),
        'intra_distances': intra_distances,  # Full arrays for plotting
        'inter_distances': inter_distances,
    }


