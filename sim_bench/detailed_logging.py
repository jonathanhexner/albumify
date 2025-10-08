"""
Detailed logging for debugging and analysis.
Separate logger for verbose operation details.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def setup_detailed_logger(log_file: Path, level: str = "DEBUG") -> logging.Logger:
    """
    Set up detailed logger for verbose operation logging.
    
    Args:
        log_file: Path to detailed log file (usually 'detailed.log' in output dir)
        level: Logging level for detailed logger
        
    Returns:
        Configured detailed logger
    """
    logger = logging.getLogger("sim_bench.detailed")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create detailed formatter with more info
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler only (don't spam console)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Don't propagate to root logger (avoid duplication)
    logger.propagate = False
    
    return logger


def log_sampling_details(logger: logging.Logger, sampling_config: Dict[str, Any], 
                         groups: List[int], images: List[str]) -> None:
    """
    Log detailed sampling information.
    
    Args:
        logger: Detailed logger instance
        sampling_config: Sampling configuration
        groups: List of group IDs
        images: List of image paths
    """
    logger.info("=" * 80)
    logger.info("SAMPLING DETAILS")
    logger.info("=" * 80)
    logger.info(f"Sampling config: {sampling_config}")
    logger.info(f"Total images after sampling: {len(images)}")
    logger.info(f"Total groups: {len(set(groups))}")
    
    # Group distribution
    from collections import Counter
    group_counts = Counter(groups)
    logger.info(f"Group distribution:")
    for group_id in sorted(group_counts.keys())[:10]:  # First 10 groups
        logger.info(f"  Group {group_id}: {group_counts[group_id]} images")
    if len(group_counts) > 10:
        logger.info(f"  ... and {len(group_counts) - 10} more groups")
    
    # Selected groups
    selected_groups = sorted(set(groups))
    logger.info(f"Selected groups: {selected_groups[:20]}")  # First 20
    if len(selected_groups) > 20:
        logger.info(f"  ... and {len(selected_groups) - 20} more")
    
    # Sample images
    logger.info(f"Sample image paths (first 5):")
    for i, img_path in enumerate(images[:5]):
        logger.info(f"  [{i}] Group {groups[i]}: {img_path}")


def log_feature_extraction_details(
    logger: logging.Logger, 
    method_name: str, 
    image_paths: List[str], 
    features: np.ndarray
):
    """
    Log detailed feature extraction information.
    
    Args:
        logger: Logger instance
        method_name: Name of the feature extraction method
        image_paths: List of image file paths
        features: Extracted feature matrix
    """
    try:
        logger.debug(f"Feature Extraction Details for {method_name}")
        logger.debug(f"Total Images: {len(image_paths)}")
        logger.debug(f"Feature Matrix Shape: {features.shape}")
        
        # Safely log details for first few images
        for i in range(min(5, len(image_paths))):
            try:
                logger.debug(f"  Image {i}: {image_paths[i]}")
                logger.debug(f"    Feature vector shape: {features[i].shape}")
                logger.debug(f"    First 10 values: {features[i][:10].tolist()}")
                logger.debug(f"    Min: {float(features[i].min())}, Max: {float(features[i].max())}")
            except Exception as img_error:
                logger.debug(f"  Error logging details for image {i}: {img_error}")
    except Exception as e:
        logger.debug(f"Error in feature extraction logging: {e}")


def log_distance_computation_details(logger: logging.Logger, method_name: str,
                                     distance_matrix: np.ndarray) -> None:
    """
    Log detailed distance computation information.
    
    Args:
        logger: Detailed logger instance
        method_name: Name of the method
        distance_matrix: Computed distance matrix
    """
    logger.info("=" * 80)
    logger.info(f"DISTANCE COMPUTATION DETAILS: {method_name}")
    logger.info("=" * 80)
    logger.info(f"Distance matrix shape: {distance_matrix.shape}")
    logger.info(f"Distance matrix dtype: {distance_matrix.dtype}")
    logger.info(f"Memory size: {distance_matrix.nbytes / (1024**2):.2f} MB")
    
    # Distance statistics
    # Exclude diagonal (self-distances)
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    non_self_distances = distance_matrix[mask]
    
    logger.info(f"Distance statistics (excluding self-distances):")
    logger.info(f"  Min: {non_self_distances.min():.6f}")
    logger.info(f"  Max: {non_self_distances.max():.6f}")
    logger.info(f"  Mean: {non_self_distances.mean():.6f}")
    logger.info(f"  Median: {np.median(non_self_distances):.6f}")
    logger.info(f"  Std: {non_self_distances.std():.6f}")
    
    # Distance distribution (percentiles)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    logger.info(f"Distance percentiles:")
    for p in percentiles:
        val = np.percentile(non_self_distances, p)
        logger.info(f"  {p}%: {val:.6f}")
    
    # Sample distances for first query
    logger.debug(f"Sample: First query's distances to first 10 images:")
    for i in range(min(10, distance_matrix.shape[1])):
        logger.debug(f"  Query 0 -> Image {i}: {distance_matrix[0, i]:.6f}")


def log_ranking_details(logger: logging.Logger, ranking_indices: np.ndarray,
                        groups: List[int], k: int = 10) -> None:
    """
    Log detailed ranking information.
    
    Args:
        logger: Detailed logger instance
        ranking_indices: Ranking indices matrix
        groups: Group labels
        k: Number of top results to log
    """
    logger.info("=" * 80)
    logger.info(f"RANKING DETAILS (Top-{k})")
    logger.info("=" * 80)
    logger.info(f"Number of queries: {ranking_indices.shape[0]}")
    logger.info(f"Number of candidates: {ranking_indices.shape[1]}")
    
    # Sample rankings for first 3 queries
    logger.info(f"Sample rankings (first 3 queries):")
    for query_idx in range(min(3, len(ranking_indices))):
        query_group = groups[query_idx]
        logger.info(f"  Query {query_idx} (Group {query_group}):")
        
        # Top-k results
        for rank, result_idx in enumerate(ranking_indices[query_idx][:k], 1):
            result_group = groups[result_idx]
            is_relevant = (result_group == query_group)
            relevance_marker = "[RELEVANT]" if is_relevant else ""
            logger.info(f"    Rank {rank}: Image {result_idx} (Group {result_group}) {relevance_marker}")
        
        # Count relevant in top-k
        top_k_groups = [groups[idx] for idx in ranking_indices[query_idx][:k]]
        num_relevant = sum(1 for g in top_k_groups if g == query_group)
        logger.info(f"    Relevant in top-{k}: {num_relevant}")


def log_cache_operation(logger: logging.Logger, operation: str, method_name: str,
                        cache_path: Path, success: bool, details: str = "") -> None:
    """
    Log cache operation details.
    
    Args:
        logger: Detailed logger instance
        operation: Operation type ('load', 'save', 'miss', 'hit')
        method_name: Name of the method
        cache_path: Path to cache file
        success: Whether operation was successful
        details: Additional details
    """
    logger.info(f"CACHE {operation.upper()}: {method_name}")
    logger.info(f"  Cache file: {cache_path.name}")
    logger.info(f"  Full path: {cache_path}")
    logger.info(f"  Exists: {cache_path.exists()}")
    if cache_path.exists():
        logger.info(f"  Size: {cache_path.stat().st_size / (1024**2):.2f} MB")
    logger.info(f"  Success: {success}")
    if details:
        logger.info(f"  Details: {details}")

