"""
Individual workflow stages for album organization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

from sim_bench.model_hub import ImageMetrics

logger = logging.getLogger(__name__)


def discover_images(source_directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Stage 1: Discover all image files in source directory.
    
    Args:
        source_directory: Directory to scan for images
        extensions: List of file extensions to include
    
    Returns:
        List of image paths
    """
    extensions = extensions or ['.jpg', '.jpeg', '.png', '.heic', '.raw']
    
    images = []
    for ext in extensions:
        images.extend(source_directory.rglob(f'*{ext}'))
        images.extend(source_directory.rglob(f'*{ext.upper()}'))
    
    logger.info(f"Discovered {len(images)} images in {source_directory}")
    return images


def filter_by_quality(
    metrics: Dict[str, ImageMetrics],
    quality_config: Dict[str, float]
) -> List[str]:
    """
    Stage 2: Filter images by quality thresholds.
    
    Args:
        metrics: Image metrics dictionary
        quality_config: Quality threshold configuration
    
    Returns:
        List of image paths that pass quality thresholds
    """
    min_iqa = quality_config.get('min_iqa_score', 0.0)
    min_ava = quality_config.get('min_ava_score', 0.0)
    min_sharpness = quality_config.get('min_sharpness', 0.0)
    
    passed = []
    for path, metric in metrics.items():
        iqa_ok = metric.iqa_score is None or metric.iqa_score >= min_iqa
        ava_ok = metric.ava_score is None or metric.ava_score >= min_ava
        sharpness_ok = metric.sharpness is None or metric.sharpness >= min_sharpness
        
        if iqa_ok and ava_ok and sharpness_ok:
            passed.append(path)
    
    logger.info(f"Quality filter: {len(passed)}/{len(metrics)} images passed")
    return passed


def filter_by_portrait(
    metrics: Dict[str, ImageMetrics],
    portrait_config: Dict[str, Any]
) -> List[str]:
    """
    Stage 3: Filter portraits by requirements (eyes open, etc.).
    
    Args:
        metrics: Image metrics dictionary
        portrait_config: Portrait filter configuration
    
    Returns:
        List of image paths that pass portrait requirements
    """
    require_eyes_open = portrait_config.get('require_eyes_open', False)
    
    passed = []
    for path, metric in metrics.items():
        if not metric.is_portrait:
            passed.append(path)
            continue
        
        if require_eyes_open and not metric.eyes_open:
            continue
        
        passed.append(path)
    
    logger.info(f"Portrait filter: {len(passed)}/{len(metrics)} images passed")
    return passed


def organize_clusters(
    image_paths: List[str],
    labels: List[int]
) -> Dict[int, List[str]]:
    """
    Stage 4: Organize images into cluster dictionary.
    
    Args:
        image_paths: List of image paths
        labels: Cluster labels (one per image)
    
    Returns:
        Dictionary mapping cluster_id -> list of image paths
    """
    clusters = {}
    for path, label in zip(image_paths, labels):
        if label == -1:
            continue
        
        clusters.setdefault(label, []).append(path)
    
    logger.info(f"Organized into {len(clusters)} clusters")
    return clusters


def compute_cluster_stats(clusters: Dict[int, List[str]]) -> Dict[str, Any]:
    """
    Compute statistics about clusters.
    
    Args:
        clusters: Dictionary mapping cluster_id -> image paths
    
    Returns:
        Statistics dictionary
    """
    sizes = [len(images) for images in clusters.values()]
    
    stats = {
        'num_clusters': len(clusters),
        'total_images': sum(sizes),
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'avg_size': sum(sizes) / len(sizes) if sizes else 0,
    }
    
    return stats
