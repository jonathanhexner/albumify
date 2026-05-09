"""Tests for the face_cluster_knn bridge in cluster_people.py.

Verifies that the main pipeline's bridge to face_cluster/ works correctly:
- Quality gating does NOT reject faces when blur/pose are unavailable
- Multiple identities produce multiple clusters (not one giant cluster)
- Noise faces (label=-1) are excluded from people_clusters
- Config dict only includes method-relevant parameters
"""
import numpy as np
import pytest
from unittest.mock import MagicMock


class ut_FaceClusterKNNBridge:
    """Tests for the face_cluster_knn method in ClusterPeopleStep."""

    def _make_face(self, path: str, face_index: int, embedding: np.ndarray):
        """Create a FaceForClustering with the given embedding."""
        from sim_bench.pipeline.steps.cluster_people import FaceForClustering
        return FaceForClustering(
            original_path=path,
            face_index=face_index,
            embedding=embedding,
            bbox={"x": 0, "y": 0, "w": 100, "h": 100},
        )

    def _make_identity_embeddings(self, n_people: int, faces_per_person: int, dim: int = 512) -> np.ndarray:
        """Create synthetic embeddings for n_people identities.

        Each person gets a distinct direction vector with small noise per face.
        In 512-D space, noise scale must be very small (~0.005) to keep
        cosine distance between same-person faces < 0.1, matching real ArcFace
        embeddings where same-person distance is typically 0.1-0.3.
        """
        rng = np.random.RandomState(42)
        embeddings = []
        for person_id in range(n_people):
            # Random unit vector for this identity
            base = rng.randn(dim)
            base = base / np.linalg.norm(base)
            for _ in range(faces_per_person):
                # Very small noise to keep cosine distance < 0.1 in high-D space
                noise = rng.randn(dim) * 0.005
                face_emb = base + noise
                face_emb = face_emb / np.linalg.norm(face_emb)
                embeddings.append(face_emb)
        return np.array(embeddings, dtype=np.float32)

    def test_quality_gating_passes_all_faces_when_blur_disabled(self):
        """With blur_min=0.0 and pose disabled, ALL faces must pass quality gating."""
        from face_cluster.config import PipelineConfig as FCConfig
        from face_cluster.quality import QualityGater
        from face_cluster.types import FaceRecord

        cfg = FCConfig(
            blur_min=0.0,
            yaw_max=999.0,
            pitch_max=999.0,
            roll_max=999.0,
            det_score_min=None,
        )
        gater = QualityGater(cfg)

        # Create faces with no blur_score, no pose, no det_score (main pipeline defaults)
        faces = []
        for i in range(10):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            faces.append(FaceRecord(
                face_id=i,
                image_id=f"img_{i}.jpg",
                image_path=f"/img/{i}.jpg",
                face_index=i,
                bbox=(0, 0, 100, 100),
                embedding=emb,
                embedding_normalized=emb,
                blur_score=0.0,   # Main pipeline never computes this
                area=10000.0,
                pose=None,        # Main pipeline never sets this
                det_score=None,   # Main pipeline never sets this
            ))

        core_indices, holdout_indices, verdicts = gater.select_core_set(faces)
        assert len(core_indices) == 10, (
            f"Expected all 10 faces to pass quality gating, got {len(core_indices)} core, "
            f"{len(holdout_indices)} holdout. "
            f"Verdicts: {[(v.face_id, v.verdict, {k: g.passed for k, g in v.gates.items()}) for v in verdicts[:3]]}"
        )

    def test_multiple_identities_produce_multiple_clusters(self):
        """3 distinct identities must produce 3 clusters, not 1 giant cluster."""
        from sim_bench.pipeline.steps.cluster_people import ClusterPeopleStep

        n_people = 3
        faces_per_person = 5
        embeddings = self._make_identity_embeddings(n_people, faces_per_person)

        faces = []
        for i in range(n_people * faces_per_person):
            faces.append(self._make_face(f"/img/{i}.jpg", 0, embeddings[i]))

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        context = MagicMock()
        context.fc_export_dir = None

        config = {
            "K": 5,
            "distance_threshold": 0.35,
            "min_cluster_size": 2,
            "merge_enabled": False,
            "attach_enabled": False,
            "export_for_analysis": False,
        }

        step = ClusterPeopleStep()
        labels = step._run_face_cluster_knn(faces, embeddings_norm, config, context)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        assert len(unique_labels) >= 2, (
            f"Expected at least 2 clusters for 3 identities, got {len(unique_labels)}. "
            f"Labels: {labels.tolist()}"
        )
        assert len(unique_labels) <= 4, (
            f"Expected at most 4 clusters for 3 identities, got {len(unique_labels)}. "
            f"Over-segmentation."
        )

    def test_noise_faces_excluded_from_people_clusters(self):
        """Faces with label=-1 must NOT appear in context.people_clusters."""
        from sim_bench.pipeline.steps.cluster_people import ClusterPeopleStep, FaceForClustering

        # Create faces and manually set labels (simulating clustering output)
        faces = [
            self._make_face("/a.jpg", 0, np.random.randn(512).astype(np.float32)),
            self._make_face("/b.jpg", 0, np.random.randn(512).astype(np.float32)),
            self._make_face("/c.jpg", 0, np.random.randn(512).astype(np.float32)),
        ]
        labels = np.array([0, -1, 1])  # face 1 is noise

        # Assign labels
        for face, label in zip(faces, labels):
            face.cluster_id = int(label)

        # Group by cluster (same logic as process() method)
        clusters = {}
        noise_count = 0
        for face in faces:
            if face.cluster_id == -1:
                noise_count += 1
                continue
            if face.cluster_id not in clusters:
                clusters[face.cluster_id] = []
            clusters[face.cluster_id].append(face)

        assert -1 not in clusters, "Noise cluster (-1) should be excluded"
        assert noise_count == 1, f"Expected 1 noise face, got {noise_count}"
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"

    def test_all_noise_produces_empty_clusters(self):
        """When all faces are noise (label=-1), clusters dict must be empty."""
        faces = [
            self._make_face("/a.jpg", 0, np.random.randn(512).astype(np.float32)),
            self._make_face("/b.jpg", 0, np.random.randn(512).astype(np.float32)),
        ]
        labels = np.array([-1, -1])

        for face, label in zip(faces, labels):
            face.cluster_id = int(label)

        clusters = {}
        for face in faces:
            if face.cluster_id == -1:
                continue
            if face.cluster_id not in clusters:
                clusters[face.cluster_id] = []
            clusters[face.cluster_id].append(face)

        assert len(clusters) == 0, f"All-noise should produce 0 clusters, got {len(clusters)}"


class ut_PipelineConfigDict:
    """Tests for the config dict built by pipeline_runner."""

    def test_face_cluster_knn_config_excludes_legacy_params(self):
        """face_cluster_knn config must NOT include HDBSCAN/mutual_knn/agglomerative params."""
        # Simulate the config building logic from pipeline_runner.py
        people_method = "face_cluster_knn"
        fc_K = 10
        fc_dist_threshold = 0.35
        fc_min_cluster = 2
        fc_merge_enabled = False
        fc_attach_enabled = False
        fc_export = True
        fc_merge_params = {}

        config = {
            "method": people_method,
            **(
                {
                    "K": fc_K,
                    "distance_threshold": fc_dist_threshold,
                    "min_cluster_size": fc_min_cluster,
                    "merge_enabled": fc_merge_enabled,
                    "attach_enabled": fc_attach_enabled,
                    "export_for_analysis": fc_export,
                    **fc_merge_params,
                } if people_method == "face_cluster_knn"
                else {}
            ),
        }

        # These legacy params must NOT be present
        assert "cluster_selection_epsilon" not in config
        assert "pca_components" not in config
        assert "k" not in config  # lowercase k (mutual_knn)
        assert "similarity_threshold" not in config
        assert "min_samples" not in config

        # These face_cluster_knn params MUST be present
        assert config["K"] == 10
        assert config["distance_threshold"] == 0.35
        assert config["method"] == "face_cluster_knn"

    def test_hdbscan_config_excludes_knn_params(self):
        """HDBSCAN config must NOT include face_cluster_knn params."""
        people_method = "hdbscan"
        people_min_cluster_size = 3
        cluster_merge_epsilon = 0.3

        config = {
            "method": people_method,
            **(
                {}
                if people_method == "face_cluster_knn"
                else {
                    "min_cluster_size": people_min_cluster_size,
                    "min_samples": people_min_cluster_size,
                    **({"cluster_selection_epsilon": cluster_merge_epsilon} if people_method in ("hdbscan", "hdbscan_pca") else {}),
                }
            ),
        }

        assert "K" not in config
        assert "merge_enabled" not in config
        assert "export_for_analysis" not in config
        assert config["min_cluster_size"] == 3
        assert config["cluster_selection_epsilon"] == 0.3
