"""End-to-end face pipeline test using Budapest 2025 test data.

Tests the ACTUAL pipeline steps, not reimplementations.
Uses the real InsightFace detection, embedding extraction, and clustering steps.

Test data: D:\\sim-bench\\test_data\\budapest_2025
- 2 folders (0000, 0001) representing 2 different people
- Full images (not pre-cropped)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.insightface_detect_faces import InsightFaceDetectFacesStep
from sim_bench.pipeline.steps.extract_face_embeddings import ExtractFaceEmbeddingsStep
from sim_bench.pipeline.steps.cluster_people import ClusterPeopleStep

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_DATA_DIR = Path(r"D:\sim-bench\test_data\budapest_2025")
OUTPUT_DIR = Path(r"D:\sim-bench\test_data\budapest_2025_output")


class TestFacePipelineE2E:
    """End-to-end tests using actual pipeline steps."""

    @pytest.fixture(scope="class")
    def context(self) -> PipelineContext:
        """Create pipeline context with test images."""
        if not TEST_DATA_DIR.exists():
            pytest.skip(f"Test data not found: {TEST_DATA_DIR}")

        # Collect all image paths with ground truth labels
        image_paths = []
        ground_truth = {}  # path -> person_id

        for person_dir in sorted(TEST_DATA_DIR.iterdir()):
            if not person_dir.is_dir():
                continue

            person_id = person_dir.name
            for img_path in sorted(person_dir.glob("*.jpg")):
                image_paths.append(img_path)
                ground_truth[str(img_path)] = person_id

        logger.info(f"Loaded {len(image_paths)} images from {len(set(ground_truth.values()))} people")

        ctx = PipelineContext(
            source_directory=TEST_DATA_DIR,
            image_paths=image_paths,
        )
        ctx.ground_truth = ground_truth  # Store for later verification
        return ctx

    @pytest.fixture(scope="class")
    def context_with_faces(self, context) -> PipelineContext:
        """Run face detection step."""
        step = InsightFaceDetectFacesStep()
        config = {
            "model_name": "buffalo_l",
            "det_size": 640,
            "det_thresh": 0.5,
        }

        logger.info("Running InsightFace face detection...")
        step.process(context, config)

        # Log detection results
        total_faces = 0
        for img_path, face_data in context.insightface_faces.items():
            faces = face_data.get("faces", [])
            total_faces += len(faces)
            logger.info(f"  {Path(img_path).name}: {len(faces)} faces")

        logger.info(f"Total faces detected: {total_faces}")
        return context

    @pytest.fixture(scope="class")
    def context_with_embeddings(self, context_with_faces) -> PipelineContext:
        """Run embedding extraction step."""
        step = ExtractFaceEmbeddingsStep()
        config = {
            "backend": "insightface",
            "model_name": "buffalo_l",
            "device": "cpu",
        }

        logger.info("Running face embedding extraction...")
        step.process(context_with_faces, config)

        # Log embedding results
        total = len(context_with_faces.face_embeddings)
        zero_count = sum(
            1 for emb in context_with_faces.face_embeddings.values()
            if np.allclose(emb, 0)
        )

        logger.info(f"Embeddings: {total - zero_count}/{total} valid ({100*(total-zero_count)/total:.1f}%)")

        # Log sample keys
        sample_keys = list(context_with_faces.face_embeddings.keys())[:5]
        logger.info(f"Sample embedding keys: {sample_keys}")

        return context_with_faces

    @pytest.fixture(scope="class")
    def context_with_clusters(self, context_with_embeddings) -> PipelineContext:
        """Run clustering step."""
        step = ClusterPeopleStep()
        config = {
            "method": "hdbscan",
            "min_cluster_size": 2,
            "min_samples": 2,
            "cluster_selection_epsilon": 0.3,
        }

        logger.info("Running people clustering...")
        step.process(context_with_embeddings, config)

        # Log clustering results
        num_clusters = len(context_with_embeddings.people_clusters)
        logger.info(f"Clusters created: {num_clusters}")

        for cluster_id, faces in context_with_embeddings.people_clusters.items():
            logger.info(f"  Cluster {cluster_id}: {len(faces)} faces")

        return context_with_embeddings

    def test_faces_detected(self, context_with_faces):
        """Verify faces were detected."""
        assert hasattr(context_with_faces, "insightface_faces")
        assert len(context_with_faces.insightface_faces) > 0

        total_faces = sum(
            len(data.get("faces", []))
            for data in context_with_faces.insightface_faces.values()
        )
        assert total_faces > 0, "No faces detected"

    def test_embeddings_valid(self, context_with_embeddings):
        """Verify embeddings are non-zero."""
        embeddings = context_with_embeddings.face_embeddings
        assert len(embeddings) > 0, "No embeddings extracted"

        zero_count = sum(1 for emb in embeddings.values() if np.allclose(emb, 0))
        total = len(embeddings)

        # Allow some failures but not 90%+
        assert zero_count < total * 0.5, f"Too many zero embeddings: {zero_count}/{total}"

    def test_clustering_produces_multiple_clusters(self, context_with_clusters):
        """Verify clustering produces expected number of clusters."""
        clusters = context_with_clusters.people_clusters
        assert len(clusters) >= 2, f"Expected at least 2 clusters, got {len(clusters)}"

    def test_clustering_accuracy(self, context_with_clusters):
        """Verify clustering matches ground truth."""
        clusters = context_with_clusters.people_clusters
        ground_truth = context_with_clusters.ground_truth

        # Map each cluster to the most common ground truth person
        cluster_to_person = {}
        for cluster_id, faces in clusters.items():
            if cluster_id == -1:  # Skip noise
                continue

            person_counts = {}
            for face in faces:
                img_path = str(face.original_path)
                true_person = ground_truth.get(img_path, "unknown")
                person_counts[true_person] = person_counts.get(true_person, 0) + 1

            if person_counts:
                majority_person = max(person_counts, key=person_counts.get)
                cluster_to_person[cluster_id] = majority_person
                logger.info(f"Cluster {cluster_id} -> Person {majority_person} (counts: {person_counts})")

        # Calculate accuracy
        correct = 0
        total = 0
        for cluster_id, faces in clusters.items():
            if cluster_id == -1:
                continue

            expected_person = cluster_to_person.get(cluster_id)
            for face in faces:
                total += 1
                img_path = str(face.original_path)
                true_person = ground_truth.get(img_path)
                if true_person == expected_person:
                    correct += 1

        accuracy = correct / total if total > 0 else 0
        logger.info(f"Clustering accuracy: {accuracy:.1%} ({correct}/{total})")

        assert accuracy > 0.7, f"Clustering accuracy {accuracy:.1%} < 70%"

    def test_save_cropped_faces_by_cluster(self, context_with_clusters):
        """Save cropped faces organized by cluster for visual verification."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        clusters = context_with_clusters.people_clusters
        insightface_faces = context_with_clusters.insightface_faces
        ground_truth = context_with_clusters.ground_truth

        # Build lookup: (image_path, face_index) -> bbox
        bbox_lookup = {}
        for img_path, face_data in insightface_faces.items():
            for face_info in face_data.get("faces", []):
                key = (img_path, face_info.get("face_index", 0))
                bbox_lookup[key] = face_info.get("bbox", {})

        saved_count = 0
        for cluster_id, faces in clusters.items():
            cluster_name = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
            cluster_dir = OUTPUT_DIR / cluster_name
            cluster_dir.mkdir(exist_ok=True)

            for face in faces:
                img_path = face.original_path
                face_index = face.face_index
                true_person = ground_truth.get(str(img_path), "unknown")

                # Get bbox
                key = (str(img_path).replace("\\", "/"), face_index)
                bbox = bbox_lookup.get(key, {})

                if not bbox:
                    # Try with original path format
                    key = (str(img_path), face_index)
                    bbox = bbox_lookup.get(key, {})

                x_px = bbox.get("x_px", 0)
                y_px = bbox.get("y_px", 0)
                w_px = bbox.get("w_px", 0)
                h_px = bbox.get("h_px", 0)

                if w_px <= 0 or h_px <= 0:
                    logger.warning(f"Invalid bbox for {img_path.name} face {face_index}")
                    continue

                # Load and crop image
                pil_img = Image.open(img_path)
                pil_img = ImageOps.exif_transpose(pil_img)

                # Crop with padding
                pad = int(min(w_px, h_px) * 0.2)
                left = max(0, x_px - pad)
                top = max(0, y_px - pad)
                right = min(pil_img.width, x_px + w_px + pad)
                bottom = min(pil_img.height, y_px + h_px + pad)

                if right <= left or bottom <= top:
                    logger.warning(f"Invalid crop coords for {img_path.name} face {face_index}")
                    continue

                face_crop = pil_img.crop((left, top, right, bottom))

                # Save with descriptive filename: truePerson_imageName_faceIndex.jpg
                output_name = f"true{true_person}_{img_path.stem}_face{face_index}.jpg"
                output_path = cluster_dir / output_name
                face_crop.save(output_path, quality=95)
                saved_count += 1

        logger.info(f"Saved {saved_count} cropped faces to: {OUTPUT_DIR}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
