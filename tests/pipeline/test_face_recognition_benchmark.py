"""Face Recognition Benchmark Test.

Tests face embedding extraction and clustering using CASIA WebFace test data.
This isolates whether the embedding model works correctly on valid face crops.

Test data: D:\sim-bench\test_data\casia_webface
- 2 folders (00000, 00001) representing 2 different people
- Pre-cropped face images (bypasses face detection)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import hdbscan
import numpy as np
import pytest
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from sim_bench.pipeline.face_embedding.insightface_native import InsightFaceNativeExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data location
TEST_DATA_DIR = Path(r"D:\sim-bench\test_data\casia_webface")


def load_test_faces() -> Dict[str, List[Tuple[Path, np.ndarray]]]:
    """Load face images from CASIA WebFace test folders.

    Returns:
        Dict mapping person_id to list of (path, image_array) tuples.
    """
    if not TEST_DATA_DIR.exists():
        pytest.skip(f"Test data not found: {TEST_DATA_DIR}")

    faces_by_person = {}

    for person_dir in sorted(TEST_DATA_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        faces = []

        for img_path in sorted(person_dir.glob("*.jpg")):
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            faces.append((img_path, img_array))

        if faces:
            faces_by_person[person_id] = faces
            logger.info(f"Loaded {len(faces)} faces for person {person_id}")

    return faces_by_person


def get_extractor() -> InsightFaceNativeExtractor:
    """Get InsightFace native extractor."""
    config = {
        "backend": "insightface",
        "model_name": "buffalo_l",
        "device": "cpu"
    }
    return InsightFaceNativeExtractor(config)


class TestEmbeddingExtraction:
    """Test that embedding extraction produces valid embeddings."""

    @pytest.fixture(scope="class")
    def faces_by_person(self):
        """Load test faces once for all tests in this class."""
        return load_test_faces()

    @pytest.fixture(scope="class")
    def extractor(self):
        """Get extractor once for all tests."""
        return get_extractor()

    @pytest.fixture(scope="class")
    def embeddings_by_person(self, faces_by_person, extractor) -> Dict[str, List[np.ndarray]]:
        """Extract embeddings for all faces."""
        result = {}

        for person_id, faces in faces_by_person.items():
            face_images = [img for _, img in faces]
            face_metadata = [{"path": str(p), "face_index": i} for i, (p, _) in enumerate(faces)]

            embeddings = extractor.extract_batch(face_images, face_metadata)
            result[person_id] = embeddings

            logger.info(f"Extracted {len(embeddings)} embeddings for person {person_id}")

        return result

    def test_embeddings_are_valid(self, embeddings_by_person):
        """All embeddings should be non-zero 512-dim vectors."""
        total = 0
        zero_count = 0

        for person_id, embeddings in embeddings_by_person.items():
            for i, emb in enumerate(embeddings):
                total += 1

                # Check dimension
                assert emb.shape == (512,), f"Person {person_id} face {i}: expected shape (512,), got {emb.shape}"

                # Check non-zero
                norm = np.linalg.norm(emb)
                if np.isclose(norm, 0):
                    zero_count += 1
                    logger.warning(f"Person {person_id} face {i}: ZERO VECTOR")
                else:
                    # Check normalized (should be ~1.0)
                    assert 0.99 < norm < 1.01, f"Person {person_id} face {i}: norm={norm}, expected ~1.0"

        zero_pct = 100 * zero_count / total if total > 0 else 0
        logger.info(f"Total: {total}, Zero vectors: {zero_count} ({zero_pct:.1f}%)")

        # CRITICAL: No zero vectors allowed
        assert zero_count == 0, f"Found {zero_count} zero vectors out of {total} ({zero_pct:.1f}%)"

    def test_embeddings_are_different(self, embeddings_by_person):
        """Embeddings should not all be identical."""
        all_embeddings = []
        for embeddings in embeddings_by_person.values():
            all_embeddings.extend(embeddings)

        if len(all_embeddings) < 2:
            pytest.skip("Need at least 2 embeddings")

        # Check that not all embeddings are identical
        first = all_embeddings[0]
        all_same = all(np.allclose(emb, first) for emb in all_embeddings[1:])

        assert not all_same, "All embeddings are identical - this indicates a bug"


class TestSimilarityMetrics:
    """Test similarity between embeddings."""

    @pytest.fixture(scope="class")
    def faces_by_person(self):
        return load_test_faces()

    @pytest.fixture(scope="class")
    def extractor(self):
        return get_extractor()

    @pytest.fixture(scope="class")
    def embeddings_by_person(self, faces_by_person, extractor) -> Dict[str, List[np.ndarray]]:
        result = {}
        for person_id, faces in faces_by_person.items():
            face_images = [img for _, img in faces]
            face_metadata = [{"path": str(p), "face_index": i} for i, (p, _) in enumerate(faces)]
            embeddings = extractor.extract_batch(face_images, face_metadata)
            result[person_id] = embeddings
        return result

    def test_intra_person_similarity(self, embeddings_by_person):
        """Faces of the same person should have high similarity."""
        for person_id, embeddings in embeddings_by_person.items():
            if len(embeddings) < 2:
                continue

            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)

            if similarities:
                mean_sim = np.mean(similarities)
                min_sim = np.min(similarities)
                max_sim = np.max(similarities)

                logger.info(f"Person {person_id} intra-similarity: mean={mean_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")

                # Same person should have high similarity
                assert mean_sim > 0.5, f"Person {person_id}: mean similarity {mean_sim:.3f} < 0.5 threshold"

    def test_inter_person_distance(self, embeddings_by_person):
        """Faces of different people should have low similarity."""
        person_ids = list(embeddings_by_person.keys())

        if len(person_ids) < 2:
            pytest.skip("Need at least 2 people for inter-person test")

        inter_similarities = []

        for i, pid1 in enumerate(person_ids):
            for pid2 in person_ids[i + 1:]:
                for emb1 in embeddings_by_person[pid1]:
                    for emb2 in embeddings_by_person[pid2]:
                        sim = np.dot(emb1, emb2)
                        inter_similarities.append(sim)

        if inter_similarities:
            mean_sim = np.mean(inter_similarities)
            max_sim = np.max(inter_similarities)

            logger.info(f"Inter-person similarity: mean={mean_sim:.3f}, max={max_sim:.3f}")

            # Different people should have low similarity
            assert mean_sim < 0.4, f"Inter-person mean similarity {mean_sim:.3f} > 0.4 threshold"


class TestClustering:
    """Test that clustering correctly groups faces by person."""

    @pytest.fixture(scope="class")
    def faces_by_person(self):
        return load_test_faces()

    @pytest.fixture(scope="class")
    def extractor(self):
        return get_extractor()

    @pytest.fixture(scope="class")
    def embeddings_with_labels(self, faces_by_person, extractor) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get embeddings with ground truth labels."""
        all_embeddings = []
        all_labels = []
        person_ids = []

        for person_id, faces in faces_by_person.items():
            person_ids.append(person_id)
            label = len(person_ids) - 1  # 0, 1, 2, ...

            face_images = [img for _, img in faces]
            face_metadata = [{"path": str(p), "face_index": i} for i, (p, _) in enumerate(faces)]
            embeddings = extractor.extract_batch(face_images, face_metadata)

            for emb in embeddings:
                all_embeddings.append(emb)
                all_labels.append(label)

        return np.array(all_embeddings), np.array(all_labels), person_ids

    def test_hdbscan_clustering(self, embeddings_with_labels):
        """HDBSCAN should produce correct clusters."""
        embeddings, true_labels, person_ids = embeddings_with_labels

        if len(embeddings) < 4:
            pytest.skip("Need at least 4 faces for HDBSCAN")

        # Normalize embeddings (HDBSCAN with euclidean on normalized = cosine)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_normalized = embeddings / norms

        # Same settings as cluster_people.py
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.3,
        )
        predicted_labels = clusterer.fit_predict(embeddings_normalized)

        # Count clusters (excluding noise label -1)
        unique_clusters = set(predicted_labels) - {-1}
        noise_count = np.sum(predicted_labels == -1)

        logger.info(f"HDBSCAN: {len(unique_clusters)} clusters, {noise_count} noise points")
        logger.info(f"Expected: {len(person_ids)} people")

        # We expect 2 clusters for 2 people
        assert len(unique_clusters) >= len(person_ids), \
            f"Expected at least {len(person_ids)} clusters, got {len(unique_clusters)}"

        # Calculate clustering accuracy using majority vote
        # For each true person, find which cluster has majority of their faces
        correct = 0
        total = len(true_labels)

        for true_label in set(true_labels):
            mask = true_labels == true_label
            pred_for_person = predicted_labels[mask]

            # Find most common predicted cluster (excluding noise)
            non_noise = pred_for_person[pred_for_person != -1]
            if len(non_noise) > 0:
                most_common = np.bincount(non_noise.astype(int)).argmax()
                correct += np.sum(pred_for_person == most_common)

        accuracy = correct / total if total > 0 else 0
        logger.info(f"Clustering accuracy: {accuracy:.1%} ({correct}/{total})")

        # Expect high accuracy
        assert accuracy > 0.8, f"Clustering accuracy {accuracy:.1%} < 80% threshold"

    def test_agglomerative_clustering(self, embeddings_with_labels):
        """Agglomerative clustering should produce correct clusters."""
        embeddings, true_labels, person_ids = embeddings_with_labels

        if len(embeddings) < 2:
            pytest.skip("Need at least 2 faces for clustering")

        # Cluster with known number of clusters
        clustering = AgglomerativeClustering(
            n_clusters=len(person_ids),
            metric='cosine',
            linkage='average'
        )
        predicted_labels = clustering.fit_predict(embeddings)

        # Calculate accuracy
        ari = adjusted_rand_score(true_labels, predicted_labels)

        logger.info(f"Agglomerative (n={len(person_ids)}): ARI={ari:.3f}")

        # ARI > 0.8 indicates good clustering
        assert ari > 0.8, f"Adjusted Rand Index {ari:.3f} < 0.8 threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
