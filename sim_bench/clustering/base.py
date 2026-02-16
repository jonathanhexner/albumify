"""
Abstract base class for clustering methods in sim-bench.
Uses Strategy pattern with factory function.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ClusteringMethod(ABC):
    """Abstract base class for all clustering methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clustering method with configuration.
        
        Args:
            config: Clustering configuration dictionary
        """
        self.config = config
        self.algorithm = config.get('algorithm', 'unknown')
        self.params = config.get('params', {})
        self.output_config = config.get('output', {})
    
    @abstractmethod
    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster features.
        
        Args:
            features: Feature matrix [n_samples, n_features]
            
        Returns:
            labels: Cluster labels
            stats: Dictionary with clustering statistics
        """
        pass
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        L2-normalize feature vectors (for cosine distance).
        
        Args:
            features: Feature matrix [n_samples, n_features]
            
        Returns:
            Normalized feature matrix
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return features / norms
    
    def save_results(
        self,
        output_dir: Path,
        image_paths: list,
        labels: np.ndarray,
        stats: Dict[str, Any],
        experiment_name: str = "Clustering Experiment"
    ) -> None:
        """
        Save clustering results to disk.

        Args:
            output_dir: Directory to save results
            image_paths: List of image paths
            labels: Cluster labels
            stats: Clustering statistics
            experiment_name: Name of the experiment for HTML gallery
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Save clusters CSV
        if self.output_config.get('save_csv', True):
            csv_path = output_dir / 'clusters.csv'
            with open(csv_path, 'w') as f:
                f.write('image_path,cluster_id\n')
                for img_path, label in zip(image_paths, labels):
                    f.write(f'{img_path},{label}\n')
            saved_files.append(f"clusters: {csv_path}")

        # Save statistics JSON
        if self.output_config.get('save_stats', True):
            stats_path = output_dir / 'cluster_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            saved_files.append(f"stats: {stats_path}")

        # Generate HTML gallery
        if self.output_config.get('save_galleries', False):
            from sim_bench.clustering.gallery import generate_cluster_gallery, open_in_browser

            html_path = generate_cluster_gallery(
                output_dir,
                image_paths,
                labels,
                stats,
                experiment_name
            )
            saved_files.append(f"gallery: {html_path}")

            if self.output_config.get('open_browser', False):
                open_in_browser(html_path)

        logger.info(f"Saved clustering results: {', '.join(saved_files)}")
        self._log_summary(stats)
    
    def _log_summary(self, stats: Dict[str, Any]) -> None:
        """Log clustering summary."""
        sizes = sorted(stats['cluster_sizes'].values(), reverse=True) if stats['cluster_sizes'] else []

        noise_info = ""
        if 'n_noise' in stats:
            noise_info = f", noise={stats['n_noise']} ({stats['noise_ratio']:.1%})"

        size_info = "no clusters"
        if sizes:
            size_info = f"min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}"

        logger.info(
            f"Clustering complete: algorithm={stats['algorithm']}, "
            f"n_clusters={stats['n_clusters']}{noise_info}, sizes=[{size_info}]"
        )
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.algorithm})"


def load_clustering_method(config: Dict[str, Any]) -> ClusteringMethod:
    """
    Factory function to load a clustering method by algorithm name.
    
    Args:
        config: Clustering configuration dictionary with 'algorithm' key
        
    Returns:
        Instantiated clustering method object
        
    Raises:
        ValueError: If algorithm is not recognized
    """
    from sim_bench.clustering.dbscan import DBSCANClusterer
    from sim_bench.clustering.kmeans import KMeansClusterer
    from sim_bench.clustering.hdbscan import HDBSCANClusterer
    from sim_bench.clustering.hierarchical import HierarchicalClusterer
    from sim_bench.clustering.hybrid_hdbscan_knn import HybridHDBSCANKNN
    from sim_bench.clustering.hybrid_closest_face import HybridHDBSCANClosestFace

    algorithm = config.get('algorithm', 'dbscan').lower()

    # Method registry
    clustering_registry = {
        'dbscan': DBSCANClusterer,
        'kmeans': KMeansClusterer,
        'hdbscan': HDBSCANClusterer,
        'hierarchical': HierarchicalClusterer,
        'hybrid_hdbscan_knn': HybridHDBSCANKNN,
        'hybrid_closest_face': HybridHDBSCANClosestFace,
    }
    
    if algorithm not in clustering_registry:
        available = ', '.join(clustering_registry.keys())
        raise ValueError(f"Unknown clustering algorithm: {algorithm}. Available: {available}")
    
    clustering_class = clustering_registry[algorithm]
    return clustering_class(config)

