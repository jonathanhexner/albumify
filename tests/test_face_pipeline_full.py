"""
Full pipeline test: Run complete face detection → crop → align → embed on source images.

Tests the entire pipeline from original photos to embeddings, verifying against ground truth.
"""

import pytest
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from PIL import Image

from face_cluster import InsightFaceEmbedder


# Ground truth labels (same as embedding test)
LABELS = {
    545: 1, 569: 1, 573: 1, 634: 1, 637: 1,  # Person 1
    550: 2, 551: 2, 558: 2, 580: 2,          # Person 2
    546: 3, 557: 3, 562: 3, 584: 3,          # Person 3 (557 is profile)
    587: 4, 589: 4,                          # Person 4
}

MAX_WITHIN = {
    1: 0.30, 2: 0.31, 3: 0.63, 4: 0.05
}

MIN_BETWEEN = 0.60


@pytest.fixture(scope="module")
def ground_truth_mapping():
    """Load face_id → (image, face_index) mapping."""
    mapping_file = Path("test_data/ground_truth_mapping.csv")
    mapping = {}

    with open(mapping_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                face_id = int(parts[0])
                image_name = Path(parts[1]).name
                face_index = int(parts[2])
                mapping[face_id] = (image_name, face_index)

    return mapping


@pytest.fixture(scope="module")
def ground_truth_embeddings():
    """Load ground truth embeddings from pre-extracted crops."""
    test_dir = Path("test_data/face_crops_ground_truth")
    embedder = InsightFaceEmbedder(model_name='buffalo_l')

    embeddings = {}
    for face_id in LABELS.keys():
        crop_path = test_dir / f"face_{face_id:04d}_aligned.jpg"

        if crop_path.exists():
            img = np.array(Image.open(crop_path))

            # Handle grayscale/RGBA
            if len(img.shape) == 2:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            embeddings[face_id] = embedder.get_embedding(img)

    return embeddings


@pytest.fixture(scope="module")
def pipeline_embeddings(ground_truth_mapping):
    """
    Run full pipeline on source images and extract embeddings.

    Pipeline: detect faces → align → extract embeddings
    """
    source_dir = Path("test_data/source_images_ground_truth")
    # Use detection_order (confidence) to match ground truth mapping
    embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='detection_order')

    embeddings = {}

    print("\n" + "="*70)
    print("RUNNING FULL PIPELINE")
    print("="*70)

    # Group faces by source image
    images_to_process = {}
    for face_id, (image_name, face_index) in ground_truth_mapping.items():
        if image_name not in images_to_process:
            images_to_process[image_name] = []
        images_to_process[image_name].append((face_id, face_index))

    print(f"\nProcessing {len(images_to_process)} source images...")

    for image_name, faces_in_image in sorted(images_to_process.items()):
        image_path = source_dir / image_name

        if not image_path.exists():
            print(f"\n  SKIP {image_name}: File not found")
            continue

        print(f"\n  {image_name}:")

        # Load image using PIL (supports HEIC)
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener
        register_heif_opener()

        try:
            with Image.open(image_path) as pil_img:
                # Apply EXIF transpose
                pil_img = ImageOps.exif_transpose(pil_img)

                # Convert to RGB
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Convert to numpy array
                img_rgb = np.array(pil_img)
        except Exception as e:
            print(f"    ERROR: Failed to load image - {e}")
            continue

        # Run detection + alignment (using InsightFace)
        # NOTE: Uses detection_order (confidence) to match ground truth mapping
        detected_faces = embedder.app.get(img_rgb)

        print(f"    Detected {len(detected_faces)} faces")

        # Extract embeddings for faces we care about
        for face_id, face_index in faces_in_image:
            if face_index < len(detected_faces):
                face = detected_faces[face_index]

                # Get aligned crop
                if hasattr(face, 'embedding') and face.embedding is not None:
                    embedding = face.embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings[face_id] = embedding
                    print(f"    Face {face_id} (index {face_index}): OK")
                else:
                    print(f"    Face {face_id} (index {face_index}): No embedding")
            else:
                print(f"    Face {face_id} (index {face_index}): Index out of range (only {len(detected_faces)} detected)")

    print(f"\n  Total embeddings extracted: {len(embeddings)}/{len(LABELS)}")

    return embeddings


def cosine_distance(emb1, emb2):
    """Compute cosine distance."""
    return 1.0 - np.dot(emb1, emb2)


def compute_distance_matrix(embeddings):
    """Compute pairwise distance matrix."""
    face_ids = sorted(embeddings.keys())
    n = len(face_ids)

    distances = np.zeros((n, n))
    for i, fid_a in enumerate(face_ids):
        for j, fid_b in enumerate(face_ids):
            if i != j:
                distances[i, j] = cosine_distance(embeddings[fid_a], embeddings[fid_b])

    return pd.DataFrame(distances, index=face_ids, columns=face_ids)


class TestFullPipeline:
    """Test full pipeline from source images to embeddings."""

    def test_pipeline_extracts_all_faces(self, pipeline_embeddings):
        """Verify pipeline extracted all ground truth faces."""
        missing = [fid for fid in LABELS.keys() if fid not in pipeline_embeddings]

        if missing:
            print(f"\nMissing faces: {missing}")

        # Allow some tolerance (e.g., 90% success rate)
        success_rate = len(pipeline_embeddings) / len(LABELS)
        assert success_rate >= 0.90, \
            f"Pipeline only extracted {len(pipeline_embeddings)}/{len(LABELS)} faces ({success_rate:.1%})"

    def test_pipeline_embeddings_match_ground_truth(self, pipeline_embeddings, ground_truth_embeddings):
        """Verify pipeline embeddings are similar to ground truth crops."""
        mismatches = []

        for face_id in pipeline_embeddings.keys():
            if face_id in ground_truth_embeddings:
                pipe_emb = pipeline_embeddings[face_id]
                gt_emb = ground_truth_embeddings[face_id]

                similarity = np.dot(pipe_emb, gt_emb)

                # Embeddings should be very similar (>0.95)
                if similarity < 0.95:
                    mismatches.append(f"Face {face_id}: similarity = {similarity:.3f}")

        if mismatches:
            print("\nEmbedding mismatches:")
            for msg in mismatches:
                print(f"  {msg}")

        assert not mismatches, f"{len(mismatches)} faces have embeddings that don't match ground truth"

    def test_pipeline_preserves_identity_structure(self, pipeline_embeddings):
        """Verify pipeline preserves person identity structure (within/between distances)."""

        # Only test faces that were successfully extracted
        available_faces = set(pipeline_embeddings.keys())

        if len(available_faces) < len(LABELS) * 0.9:
            pytest.skip(f"Too few faces extracted ({len(available_faces)}/{len(LABELS)})")

        dist_df = compute_distance_matrix(pipeline_embeddings)

        failures = []

        # Test within-person distances
        for person_id, max_dist in MAX_WITHIN.items():
            faces = [fid for fid, label in LABELS.items() if label == person_id and fid in available_faces]

            if len(faces) < 2:
                continue

            for i, face_a in enumerate(faces):
                for face_b in faces[i+1:]:
                    dist = dist_df.loc[face_a, face_b]
                    if dist > max_dist:
                        failures.append(
                            f"Person {person_id}: {face_a}<->{face_b} = {dist:.3f} (exceeds {max_dist:.2f})"
                        )

        # Test between-people distances
        face_list = list(available_faces)
        for i, face_a in enumerate(face_list):
            for face_b in face_list[i+1:]:
                if LABELS[face_a] != LABELS[face_b]:
                    dist = dist_df.loc[face_a, face_b]
                    if dist < MIN_BETWEEN:
                        failures.append(
                            f"Different people ({LABELS[face_a]} vs {LABELS[face_b]}): "
                            f"{face_a}<->{face_b} = {dist:.3f} (below {MIN_BETWEEN:.2f})"
                        )

        if failures:
            print("\n\nPipeline distance matrix:")
            print(dist_df.round(3))
            print("\nFailures:")
            for f in failures[:10]:  # Show first 10
                print(f"  {f}")

        assert not failures, f"{len(failures)} distance violations in pipeline output"


class TestPipelineVsGroundTruth:
    """Compare pipeline output to ground truth crops."""

    def test_distance_matrix_correlation(self, pipeline_embeddings, ground_truth_embeddings):
        """Verify pipeline distance matrix correlates highly with ground truth."""

        # Only compare faces that are in both sets
        common_faces = set(pipeline_embeddings.keys()) & set(ground_truth_embeddings.keys())

        if len(common_faces) < 10:
            pytest.skip(f"Too few common faces ({len(common_faces)})")

        pipe_dist = compute_distance_matrix({fid: pipeline_embeddings[fid] for fid in common_faces})
        gt_dist = compute_distance_matrix({fid: ground_truth_embeddings[fid] for fid in common_faces})

        # Compute correlation of distance matrices
        pipe_dists = []
        gt_dists = []

        face_list = sorted(common_faces)
        for i, fid_a in enumerate(face_list):
            for fid_b in face_list[i+1:]:
                pipe_dists.append(pipe_dist.loc[fid_a, fid_b])
                gt_dists.append(gt_dist.loc[fid_a, fid_b])

        correlation = np.corrcoef(pipe_dists, gt_dists)[0, 1]

        print(f"\nDistance matrix correlation: {correlation:.4f}")
        print(f"Common faces: {len(common_faces)}/{len(LABELS)}")

        assert correlation > 0.90, \
            f"Pipeline distance matrix has low correlation with ground truth: {correlation:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
