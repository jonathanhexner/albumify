"""
Feature engineering for cluster merge prediction.

Separates feature computation from model training for extensibility.
Supports adding new features without modifying training code.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ClusterPairFeatures:
    """Features for a candidate cluster pair."""

    # Core distance features
    min_exemplar_dist: float
    p10_cross_dist: float
    p50_cross_dist: float
    support_fraction: float

    # Cluster size features
    diameter_ratio: float
    cluster_size_min: int
    cluster_size_ratio: float

    # Threshold features
    T_A: float
    T_B: float
    T_local: float
    T_global: float

    # Pose features (v2 - optional)
    frontal_frac_A: float = None
    frontal_frac_B: float = None
    pose_diff: float = None

    # Interaction features (v2 - optional)
    min_exemplar_dist_x_pose: float = None
    p50_cross_dist_x_pose: float = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array in specified order."""
        d = self.to_dict()
        return np.array([d.get(name, 0.0) for name in feature_names])


class FeatureComputer:
    """Compute features for cluster pairs.

    Supports versioning and extensibility for different model types.
    """

    # Feature version - increment when adding new features
    VERSION = 2

    # Feature groups for easy selection
    DISTANCE_FEATURES = [
        'min_exemplar_dist',
        'p10_cross_dist',
        'p50_cross_dist',
        'support_fraction',
    ]

    SIZE_FEATURES = [
        'diameter_ratio',
        'cluster_size_min',
        'cluster_size_ratio',
    ]

    THRESHOLD_FEATURES = [
        'T_A',
        'T_B',
        'T_local',
        'T_global',
    ]

    POSE_FEATURES = [
        'frontal_frac_A',
        'frontal_frac_B',
        'pose_diff',
    ]

    INTERACTION_FEATURES = [
        'min_exemplar_dist_x_pose',
        'p50_cross_dist_x_pose',
    ]

    # V1 features (original 12)
    V1_FEATURES = DISTANCE_FEATURES + SIZE_FEATURES + THRESHOLD_FEATURES

    # V2 features (with pose)
    V2_FEATURES = V1_FEATURES + POSE_FEATURES + INTERACTION_FEATURES

    def __init__(
        self,
        support_threshold: float = 0.35,
        frontal_threshold: float = 15.0,
        feature_version: int = 2
    ):
        """
        Initialize feature computer.

        Args:
            support_threshold: Distance threshold for support_fraction
            frontal_threshold: Max yaw/pitch for frontal faces (degrees)
            feature_version: Feature set version (1 or 2)
        """
        self.support_threshold = support_threshold
        self.frontal_threshold = frontal_threshold
        self.feature_version = feature_version

        # Select feature set based on version
        if feature_version == 1:
            self.feature_names = self.V1_FEATURES
        elif feature_version == 2:
            self.feature_names = self.V2_FEATURES
        else:
            raise ValueError(f"Unsupported feature version: {feature_version}")

    def compute_cluster_stats(
        self,
        cluster_nodes: List[int],
        exemplar_nodes: List[int],
        distance_matrix: np.ndarray,
        face_records: List[Any],
        core_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Compute statistics for a single cluster.

        Args:
            cluster_nodes: Node indices in cluster
            exemplar_nodes: Exemplar node indices
            distance_matrix: Pairwise distance matrix
            face_records: List of FaceRecord objects
            core_indices: Core face indices

        Returns:
            Dict with keys: diameter, T_A, mean_blur, face_ids, frontal_frac, mean_yaw, mean_pitch
        """
        # Diameter (max pairwise distance)
        if len(cluster_nodes) > 1:
            cluster_dists = distance_matrix[np.ix_(cluster_nodes, cluster_nodes)]
            diameter = float(cluster_dists.max())
        else:
            diameter = 0.0

        # T_A (P90 of exemplar distances)
        T_A = 0.0
        if len(exemplar_nodes) > 1:
            exemplar_dists = distance_matrix[np.ix_(exemplar_nodes, exemplar_nodes)]
            exemplar_dists_flat = exemplar_dists[np.triu_indices_from(exemplar_dists, k=1)]
            if len(exemplar_dists_flat) > 0:
                T_A = float(np.percentile(exemplar_dists_flat, 90))

        # Get face records for this cluster
        cluster_face_indices = [core_indices[n] for n in cluster_nodes]
        cluster_faces = [face_records[i] for i in cluster_face_indices]

        # Mean blur
        blur_scores = [f.blur_score for f in cluster_faces]
        mean_blur = float(np.mean(blur_scores))

        # Face IDs
        face_ids = [f.face_id for f in cluster_faces]

        # Pose statistics (v2 features)
        frontal_frac = None
        mean_yaw = None
        mean_pitch = None

        if self.feature_version >= 2:
            # Count frontal faces
            frontal_count = 0
            yaw_list = []
            pitch_list = []

            for face in cluster_faces:
                if face.pose is not None and face.pose != (0.0, 0.0, 0.0):
                    yaw, pitch, roll = face.pose
                    yaw_list.append(yaw)
                    pitch_list.append(pitch)

                    if abs(yaw) <= self.frontal_threshold and abs(pitch) <= self.frontal_threshold:
                        frontal_count += 1

            if len(yaw_list) > 0:
                frontal_frac = frontal_count / len(yaw_list)
                mean_yaw = float(np.mean(yaw_list))
                mean_pitch = float(np.mean(pitch_list))
            else:
                # No pose data available - use defaults
                frontal_frac = 1.0  # Assume frontal if no pose data
                mean_yaw = 0.0
                mean_pitch = 0.0

        return {
            'diameter': diameter,
            'T_A': T_A,
            'mean_blur': mean_blur,
            'face_ids': face_ids,
            'frontal_frac': frontal_frac,
            'mean_yaw': mean_yaw,
            'mean_pitch': mean_pitch,
        }

    def compute_pair_features(
        self,
        cluster_id_a: int,
        cluster_id_b: int,
        cluster_result: Any,
        distance_matrix: np.ndarray,
        cluster_stats: Dict[int, Dict[str, Any]],
        T_global: float
    ) -> ClusterPairFeatures:
        """
        Compute all features for a cluster pair.

        Args:
            cluster_id_a: First cluster ID
            cluster_id_b: Second cluster ID
            cluster_result: ClusterResult object
            distance_matrix: Pairwise distance matrix
            cluster_stats: Pre-computed cluster statistics
            T_global: Global threshold

        Returns:
            ClusterPairFeatures object
        """
        nodes_a = cluster_result.clusters[cluster_id_a]
        nodes_b = cluster_result.clusters[cluster_id_b]

        exemplars_a = cluster_result.exemplars.get(cluster_id_a, nodes_a)
        exemplars_b = cluster_result.exemplars.get(cluster_id_b, nodes_b)

        # 1. min_exemplar_dist
        exemplar_dists = distance_matrix[np.ix_(exemplars_a, exemplars_b)]
        min_exemplar_dist = float(exemplar_dists.min())

        # Cross-cluster distances
        cross_dists = distance_matrix[np.ix_(nodes_a, nodes_b)].flatten()

        # 2-3. p10, p50
        p10_cross_dist = float(np.percentile(cross_dists, 10))
        p50_cross_dist = float(np.percentile(cross_dists, 50))

        # 4. support_fraction
        support_fraction = float(np.mean(cross_dists < self.support_threshold))

        # 5. diameter_ratio
        dia_a = cluster_stats[cluster_id_a]['diameter']
        dia_b = cluster_stats[cluster_id_b]['diameter']
        diameter_ratio = max(dia_a, dia_b) / min(dia_a, dia_b) if dia_a > 0 and dia_b > 0 else 1.0

        # 6-7. cluster sizes
        size_a = len(nodes_a)
        size_b = len(nodes_b)
        cluster_size_min = min(size_a, size_b)
        cluster_size_ratio = max(size_a, size_b) / min(size_a, size_b)

        # 8-11. thresholds
        T_A = cluster_stats[cluster_id_a]['T_A']
        T_B = cluster_stats[cluster_id_b]['T_A']
        T_local = max(T_A, T_B)

        # V2 features: pose features
        frontal_frac_A = None
        frontal_frac_B = None
        pose_diff = None
        min_exemplar_dist_x_pose = None
        p50_cross_dist_x_pose = None

        if self.feature_version >= 2:
            frontal_frac_A = cluster_stats[cluster_id_a].get('frontal_frac', 1.0)
            frontal_frac_B = cluster_stats[cluster_id_b].get('frontal_frac', 1.0)

            # Compute pose difference (Euclidean distance of mean yaw/pitch)
            mean_yaw_a = cluster_stats[cluster_id_a].get('mean_yaw', 0.0)
            mean_pitch_a = cluster_stats[cluster_id_a].get('mean_pitch', 0.0)
            mean_yaw_b = cluster_stats[cluster_id_b].get('mean_yaw', 0.0)
            mean_pitch_b = cluster_stats[cluster_id_b].get('mean_pitch', 0.0)

            pose_diff = float(np.sqrt(
                (mean_yaw_a - mean_yaw_b) ** 2 +
                (mean_pitch_a - mean_pitch_b) ** 2
            ))

            # Interaction features
            min_exemplar_dist_x_pose = min_exemplar_dist * (1.0 + pose_diff / 90.0)  # Normalize by max pose diff
            p50_cross_dist_x_pose = p50_cross_dist * (1.0 + pose_diff / 90.0)

        return ClusterPairFeatures(
            min_exemplar_dist=min_exemplar_dist,
            p10_cross_dist=p10_cross_dist,
            p50_cross_dist=p50_cross_dist,
            support_fraction=support_fraction,
            diameter_ratio=diameter_ratio,
            cluster_size_min=cluster_size_min,
            cluster_size_ratio=cluster_size_ratio,
            T_A=T_A,
            T_B=T_B,
            T_local=T_local,
            T_global=T_global,
            frontal_frac_A=frontal_frac_A,
            frontal_frac_B=frontal_frac_B,
            pose_diff=pose_diff,
            min_exemplar_dist_x_pose=min_exemplar_dist_x_pose,
            p50_cross_dist_x_pose=p50_cross_dist_x_pose,
        )

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names for this version."""
        return self.feature_names.copy()

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get metadata about features for model saving."""
        return {
            'version': self.feature_version,
            'feature_names': self.feature_names,
            'support_threshold': self.support_threshold,
            'frontal_threshold': self.frontal_threshold,
            'n_features': len(self.feature_names),
        }
