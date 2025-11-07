"""
Abstract base class for clustering methods in sim-bench.
Uses Strategy pattern with factory function.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path
import json


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
        
        # Save clusters CSV
        if self.output_config.get('save_csv', True):
            csv_path = output_dir / 'clusters.csv'
            with open(csv_path, 'w') as f:
                f.write('image_path,cluster_id\n')
                for img_path, label in zip(image_paths, labels):
                    f.write(f'{img_path},{label}\n')
            print(f"[OK] Saved clusters to: {csv_path}")
        
        # Save statistics JSON
        if self.output_config.get('save_stats', True):
            stats_path = output_dir / 'cluster_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"[OK] Saved statistics to: {stats_path}")
        
        # Generate HTML gallery
        if self.output_config.get('save_galleries', False):
            try:
                from sim_bench.clustering.gallery import generate_cluster_gallery, open_in_browser
                
                html_path = generate_cluster_gallery(
                    output_dir,
                    image_paths,
                    labels,
                    stats,
                    experiment_name
                )
                print(f"[OK] Saved HTML gallery to: {html_path}")
                
                # Optionally open in browser
                if self.output_config.get('open_browser', False):
                    open_in_browser(html_path)
                    print(f"[OK] Opened gallery in browser")
                    
            except ImportError as e:
                print(f"[WARNING] Could not generate HTML gallery: {e}")
                print(f"          Install jinja2: pip install jinja2")
            except Exception as e:
                print(f"[WARNING] Error generating HTML gallery: {e}")
        
        # Print summary
        self._print_summary(stats)
    
    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """Print clustering summary."""
        print(f"\n{'='*60}")
        print(f"CLUSTERING RESULTS")
        print(f"{'='*60}")
        print(f"Algorithm: {stats['algorithm']}")
        print(f"Number of clusters: {stats['n_clusters']}")
        if 'n_noise' in stats:
            print(f"Noise points: {stats['n_noise']} ({stats['noise_ratio']:.1%})")
        print(f"Cluster size distribution:")
        sizes = sorted(stats['cluster_sizes'].values(), reverse=True) if stats['cluster_sizes'] else []
        if sizes:
            print(f"  Min: {min(sizes)}")
            print(f"  Max: {max(sizes)}")
            print(f"  Mean: {np.mean(sizes):.1f}")
            print(f"  Median: {np.median(sizes):.1f}")
        else:
            print(f"  No clusters found")
        print(f"{'='*60}")
    
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
    
    algorithm = config.get('algorithm', 'dbscan').lower()
    
    # Method registry
    clustering_registry = {
        'dbscan': DBSCANClusterer,
        'kmeans': KMeansClusterer,
        'hdbscan': HDBSCANClusterer,
    }
    
    if algorithm not in clustering_registry:
        available = ', '.join(clustering_registry.keys())
        raise ValueError(f"Unknown clustering algorithm: {algorithm}. Available: {available}")
    
    clustering_class = clustering_registry[algorithm]
    return clustering_class(config)

