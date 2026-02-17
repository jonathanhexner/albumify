"""
Hybrid HDBSCAN+kNN with cross-cluster d3 matching.

Algorithm:
1. HDBSCAN → initial clusters
2. For each cluster:
   - Compute d3 (distance to 3rd nearest neighbor) for all faces
   - T = median(d3) + iqr_multiplier×IQR(d3)
3. Merge clusters A and B:
   - Stage 1: Early exit if min(exemplar distances) > 2×max(T_A, T_B)
   - Stage 2: For each face, compute d3 using other cluster's neighbors
   - Merge if ≥2 faces have cross-cluster d3 < min(T_A, T_B)
4. Attach: If noise point's d3 (using cluster neighbors) < T, otherwise leave as noise
"""

import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sim_bench.clustering.base import ClusteringMethod

logger = logging.getLogger(__name__)


@dataclass
class ClusterD3State:
    """State for a cluster with d3-based threshold."""
    label: int
    indices: np.ndarray
    features: np.ndarray
    threshold: float
    d3_values: np.ndarray
    exemplar_indices: np.ndarray
    exemplar_features: np.ndarray


class HybridHDBSCANClosestFace(ClusteringMethod):
    """Hybrid HDBSCAN+kNN using closest-face matching."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # HDBSCAN parameters
        self.min_cluster_size = self.params.get('min_cluster_size', 2)
        self.min_samples = self.params.get('min_samples', 2)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.3)

        # d3 parameters
        self.knn_k = self.params.get('knn_k', 3)
        self.iqr_multiplier = self.params.get('iqr_multiplier', 2.0)
        
        # Threshold parameters
        self.threshold_floor = self.params.get('threshold_floor', 0.30)
        self.threshold_ceiling = self.params.get('threshold_ceiling', 0.90)
        
        # Exemplar parameters (for early exit check)
        self.max_exemplars = self.params.get('max_exemplars', 10)
        
        # Merge parameters
        self.merge_min_faces = self.params.get('merge_min_faces', 2)
        self.early_exit_multiplier = self.params.get('early_exit_multiplier', 2.0)
        
        # Attach parameters
        self.attach_min_neighbors = self.params.get('attach_min_neighbors', 1)
    
    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run hybrid clustering with cross-cluster d3 matching.

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
        
        # Compute cluster states (d3-based thresholds)
        cluster_states = self._compute_cluster_states(labels, features_norm)
        
        # Stage 2: Merge clusters (cross-cluster d3)
        logger.info(f"Stage 2: Merging (d3-based, min_faces={self.merge_min_faces})")
        labels, merge_info = self._merge_closest_face(labels, features_norm, cluster_states)
        
        # Recompute cluster states after merge
        cluster_states = self._compute_cluster_states(labels, features_norm)
        
        # Stage 3: Attach noise points (cross-cluster d3)
        logger.info(f"Stage 3: Attaching noise (d3-based)")
        labels, attach_info = self._attach_closest_face(labels, features_norm, cluster_states)
        
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
    
    def _compute_cluster_states(
        self,
        labels: np.ndarray,
        features: np.ndarray
    ) -> Dict[int, ClusterD3State]:
        """Compute d3-based threshold for each cluster."""
        cluster_states = {}
        
        for label in set(labels):
            if label == -1:
                continue
            
            indices = np.where(labels == label)[0]
            n_faces = len(indices)
            cluster_features = features[indices]
            
            if n_faces < 2:
                cluster_states[label] = ClusterD3State(
                    label=label,
                    indices=indices,
                    features=cluster_features,
                    threshold=self.threshold_floor,
                    d3_values=np.array([0.0]),
                    exemplar_indices=indices,
                    exemplar_features=cluster_features
                )
                continue
            
            # Compute pairwise distances within cluster
            distances = cdist(cluster_features, cluster_features, metric='euclidean')
            
            # Compute d3 for each face
            k = min(self.knn_k, n_faces - 1)
            d3_values = []
            
            for i in range(n_faces):
                sorted_dists = np.sort(distances[i])[1:k + 1]  # Exclude self
                d3_values.append(sorted_dists[-1] if len(sorted_dists) > 0 else 0)
            
            d3_values = np.array(d3_values)
            
            # Threshold = median + iqr_multiplier×IQR
            median_d3 = np.median(d3_values)
            q1, q3 = np.percentile(d3_values, [25, 75])
            iqr = q3 - q1
            raw_threshold = median_d3 + self.iqr_multiplier * iqr
            
            # Clamp to [floor, ceiling]
            threshold = max(raw_threshold, self.threshold_floor)
            threshold = min(threshold, self.threshold_ceiling)
            
            # Select exemplars (for early exit check)
            n_exemplars = min(self.max_exemplars, n_faces)
            exemplar_local_indices = np.argsort(d3_values)[:n_exemplars]
            exemplar_global_indices = indices[exemplar_local_indices]
            
            cluster_states[label] = ClusterD3State(
                label=label,
                indices=indices,
                features=cluster_features,
                threshold=float(threshold),
                d3_values=d3_values,
                exemplar_indices=exemplar_global_indices,
                exemplar_features=features[exemplar_global_indices]
            )
            
            logger.debug(f"  Cluster {label}: {n_faces} faces, median_d3={median_d3:.3f}, "
                        f"IQR={iqr:.3f}, T={threshold:.3f}")
        
        return cluster_states
    
    def _merge_closest_face(
        self, 
        labels: np.ndarray, 
        features: np.ndarray,
        cluster_states: Dict[int, ClusterD3State]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Merge clusters if enough faces fit other cluster's d3-based threshold."""
        merged_labels = labels.copy()
        cluster_ids = sorted(cluster_states.keys())
        
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
                if find(c1) == find(c2):
                    continue
                
                state_a = cluster_states[c1]
                state_b = cluster_states[c2]
                
                # Stage 1: Early exit if exemplars are very far
                exemplar_dists = cdist(state_a.exemplar_features, state_b.exemplar_features, metric='euclidean')
                min_exemplar_dist = float(np.min(exemplar_dists))
                early_exit_threshold = self.early_exit_multiplier * max(state_a.threshold, state_b.threshold)
                
                if min_exemplar_dist > early_exit_threshold:
                    logger.debug(f"  Skip {c1}+{c2}: min_exemplar_dist={min_exemplar_dist:.3f} > {early_exit_threshold:.3f}")
                    continue
                
                # Stage 2: Check cross-cluster d3 for all faces
                merge_threshold = min(state_a.threshold, state_b.threshold)
                n_fits = 0
                
                # Check faces from A fitting into B
                for face_feat in state_a.features:
                    dists_to_b = cdist([face_feat], state_b.features, metric='euclidean')[0]
                    k = min(self.knn_k, len(state_b.features))
                    d3_cross = np.sort(dists_to_b)[:k][-1] if k > 0 else float('inf')
                    if d3_cross <= merge_threshold:
                        n_fits += 1
                
                # Check faces from B fitting into A
                for face_feat in state_b.features:
                    dists_to_a = cdist([face_feat], state_a.features, metric='euclidean')[0]
                    k = min(self.knn_k, len(state_a.features))
                    d3_cross = np.sort(dists_to_a)[:k][-1] if k > 0 else float('inf')
                    if d3_cross <= merge_threshold:
                        n_fits += 1
                
                # Merge if enough faces fit
                if n_fits >= self.merge_min_faces:
                    if union(c1, c2):
                        merge_count += 1
                        logger.debug(f"  Merged {c1}+{c2}: {n_fits} faces fit, T={merge_threshold:.3f}, "
                                   f"min_exemplar_dist={min_exemplar_dist:.3f}")
        
        # Apply merges
        if merge_count > 0:
            label_mapping = {}
            for c in cluster_ids:
                root = find(c)
                if root not in label_mapping:
                    label_mapping[root] = len(label_mapping)
            
            for i, label in enumerate(merged_labels):
                if label >= 0 and label in parent:
                    merged_labels[i] = label_mapping[find(label)]
        
        logger.info(f"  Merged {merge_count} cluster pairs")
        return merged_labels, {'n_merges': merge_count}
    
    def _attach_closest_face(
        self, 
        labels: np.ndarray, 
        features: np.ndarray,
        cluster_states: Dict[int, ClusterD3State]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Attach noise points if their d3 (using cluster neighbors) fits threshold."""
        final_labels = labels.copy()
        noise_indices = np.where(labels == -1)[0]
        
        if len(noise_indices) == 0:
            return final_labels, {'n_attached': 0, 'n_remaining_noise': 0}
        
        if len(cluster_states) == 0:
            return final_labels, {'n_attached': 0, 'n_remaining_noise': len(noise_indices)}
        
        n_attached = 0
        
        for noise_idx in noise_indices:
            noise_feature = features[noise_idx:noise_idx+1]
            best_cluster = None
            best_d3 = float('inf')
            
            # Check each cluster
            for label, state in cluster_states.items():
                # Compute d3 using this cluster's faces
                dists = cdist(noise_feature, state.features, metric='euclidean')[0]
                k = min(self.knn_k, len(state.features))
                d3_cross = np.sort(dists)[:k][-1] if k > 0 else float('inf')
                
                # Check if face fits this cluster
                if d3_cross <= state.threshold and d3_cross < best_d3:
                    best_cluster = label
                    best_d3 = d3_cross
            
            if best_cluster is not None:
                final_labels[noise_idx] = best_cluster
                n_attached += 1
            # else: leave as noise (-1)
        
        n_remaining_noise = len(noise_indices) - n_attached
        logger.info(f"  Attached {n_attached} noise points, {n_remaining_noise} remain as noise")
        
        return final_labels, {'n_attached': n_attached, 'n_remaining_noise': n_remaining_noise}
    
    def _compute_stats(
        self,
        labels: np.ndarray,
        hdbscan_stats: Dict[str, Any],
        merge_info: Dict[str, Any],
        attach_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute statistics."""
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l >= 0])
        n_noise = int(np.sum(labels == -1))
        
        cluster_sizes = {}
        for label in unique_labels:
            if label >= 0:
                cluster_sizes[int(label)] = int(np.sum(labels == label))
        
        return {
            'algorithm': 'hybrid_closest_face',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'hdbscan': hdbscan_stats,
            'merges': merge_info,
            'attachments': attach_info,
            'params': {
                'min_cluster_size': self.min_cluster_size,
                'knn_k': self.knn_k,
                'iqr_multiplier': self.iqr_multiplier,
                'threshold_floor': self.threshold_floor,
                'threshold_ceiling': self.threshold_ceiling,
                'max_exemplars': self.max_exemplars,
                'merge_min_faces': self.merge_min_faces,
                'early_exit_multiplier': self.early_exit_multiplier,
                'attach_min_neighbors': self.attach_min_neighbors,
            }
        }
