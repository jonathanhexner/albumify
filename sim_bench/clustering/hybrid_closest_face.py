"""
Hybrid HDBSCAN+kNN with closest-face matching (instead of centroids).

Key differences from centroid-based:
1. Merge: If ANY 2 faces between clusters are close
2. Attach: If singleton is close to ANY face in cluster

Allows intra-person variation (hair, angle, lighting) while maintaining separation.
"""

import logging
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.spatial.distance import cdist
from sim_bench.clustering.base import ClusteringMethod

logger = logging.getLogger(__name__)


class HybridHDBSCANClosestFace(ClusteringMethod):
    """Hybrid HDBSCAN+kNN using closest-face matching."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # HDBSCAN parameters
        self.min_cluster_size = self.params.get('min_cluster_size', 2)
        self.min_samples = self.params.get('min_samples', 2)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.3)

        # Merge parameters
        self.merge_threshold = self.params.get('merge_threshold', 0.45)
        self.merge_min_close_pairs = self.params.get('merge_min_close_pairs', 2)

        # Attach parameters
        self.attach_threshold = self.params.get('attach_threshold', 0.40)
        self.attach_relaxation = self.params.get('attach_relaxation', 1.2)
    
    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run hybrid clustering with closest-face matching.

        Args:
            features: Face embedding vectors (N x D)
            collect_debug_data: If True, collect debug data (not implemented for this method)
        """
        n_samples = len(features)
        
        if n_samples == 0:
            return np.array([]), {'n_clusters': 0, 'n_noise': 0}
        
        if n_samples == 1:
            return np.array([0]), {'n_clusters': 1, 'n_noise': 0}
        
        features_norm = self.normalize_features(features)
        
        # Stage 1: HDBSCAN
        logger.info(f"Stage 1: HDBSCAN (min_cluster_size={self.min_cluster_size})")
        labels, hdbscan_stats = self._run_hdbscan(features_norm)
        
        # Stage 2: Merge clusters (closest-face)
        logger.info(f"Stage 2: Merging (threshold={self.merge_threshold}, min_pairs={self.merge_min_close_pairs})")
        labels, merge_info = self._merge_closest_face(labels, features_norm)
        
        # Stage 3: Attach singletons (closest-face)
        logger.info(f"Stage 3: Attaching singletons (threshold={self.attach_threshold})")
        labels, attach_info = self._attach_closest_face(labels, features_norm)
        
        stats = self._compute_stats(labels, hdbscan_stats, merge_info, attach_info)
        return labels, stats
    
    def _run_hdbscan(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run HDBSCAN."""
        import hdbscan
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        labels = clusterer.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise")
        return labels, {'n_clusters': n_clusters, 'n_noise': n_noise}
    
    def _merge_closest_face(
        self, 
        labels: np.ndarray, 
        features: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Merge clusters if enough face pairs are close."""
        merged_labels = labels.copy()
        
        # Get unique clusters
        cluster_ids = sorted(set(labels))
        cluster_ids = [c for c in cluster_ids if c != -1]
        
        if len(cluster_ids) <= 1:
            return merged_labels, {'n_merges': 0}
        
        # Union-find for merging
        parent = {c: c for c in cluster_ids}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        merge_count = 0
        
        # Check all pairs
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                mask1 = labels == c1
                mask2 = labels == c2
                
                features1 = features[mask1]
                features2 = features[mask2]
                
                # Compute all pairwise distances
                distances = cdist(features1, features2, metric='euclidean')
                
                # Count close pairs
                n_close = np.sum(distances < self.merge_threshold)
                
                if n_close >= self.merge_min_close_pairs:
                    if union(c1, c2):
                        merge_count += 1
                        min_dist = float(np.min(distances))
                        logger.debug(f"  Merged {c1}+{c2}: {n_close} close pairs, min_dist={min_dist:.3f}")
        
        # Apply merges
        label_mapping = {}
        for c in cluster_ids:
            root = find(c)
            if root not in label_mapping:
                label_mapping[root] = len(label_mapping)
        
        for i, label in enumerate(merged_labels):
            if label >= 0:
                merged_labels[i] = label_mapping[find(label)]
        
        logger.info(f"  Merged {merge_count} cluster pairs")
        return merged_labels, {'n_merges': merge_count}
    
    def _attach_closest_face(
        self, 
        labels: np.ndarray, 
        features: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Attach singletons to closest face in any cluster."""
        final_labels = labels.copy()
        
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0]
        
        if len(noise_indices) == 0:
            return final_labels, {'n_attached': 0, 'n_singletons': 0}
        
        cluster_labels = sorted(set(labels))
        cluster_labels = [c for c in cluster_labels if c != -1]
        
        if len(cluster_labels) == 0:
            # No clusters, make all singletons
            for i, idx in enumerate(noise_indices):
                final_labels[idx] = i
            return final_labels, {'n_attached': 0, 'n_singletons': len(noise_indices)}
        
        # Get all non-noise faces
        non_noise_mask = labels != -1
        non_noise_features = features[non_noise_mask]
        non_noise_labels = labels[non_noise_mask]
        
        n_attached = 0
        next_singleton_id = max(cluster_labels) + 1
        
        for noise_idx in noise_indices:
            noise_feature = features[noise_idx:noise_idx+1]
            
            # Find closest face across all clusters
            distances = cdist(noise_feature, non_noise_features, metric='euclidean')[0]
            closest_idx = np.argmin(distances)
            closest_dist = distances[closest_idx]
            
            if closest_dist < self.attach_threshold:
                # Attach to same cluster as closest face
                final_labels[noise_idx] = non_noise_labels[closest_idx]
                n_attached += 1
            else:
                # Create singleton
                final_labels[noise_idx] = next_singleton_id
                next_singleton_id += 1
        
        n_singletons = len(noise_indices) - n_attached
        logger.info(f"  Attached {n_attached} noise points, {n_singletons} singletons")
        
        return final_labels, {'n_attached': n_attached, 'n_singletons': n_singletons}
    
    def _compute_stats(
        self,
        labels: np.ndarray,
        hdbscan_stats: Dict[str, Any],
        merge_info: Dict[str, Any],
        attach_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute statistics."""
        unique_labels = set(labels)
        n_clusters = len(unique_labels)
        
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(labels == label))
        
        return {
            'algorithm': 'hybrid_closest_face',
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'hdbscan': hdbscan_stats,
            'merges': merge_info,
            'singletons': attach_info,
            'params': {
                'min_cluster_size': self.min_cluster_size,
                'merge_threshold': self.merge_threshold,
                'merge_min_close_pairs': self.merge_min_close_pairs,
                'attach_threshold': self.attach_threshold,
            }
        }
