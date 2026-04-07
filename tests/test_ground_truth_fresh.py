"""
Test face pipeline on fresh ground truth with manual labels.

Verifies:
1. Pipeline extracts same faces as ground truth crops
2. Embeddings match between pipeline and crops
3. Same-person faces have low distance
4. Different-person faces have high distance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from face_cluster import InsightFaceEmbedder

# Register HEIC support
register_heif_opener()

# Paths
GT_DIR = Path("test_data/ground_truth_test")
CROPS_DIR = GT_DIR / "face_crops"
LABELS_FILE = CROPS_DIR / "labels_converted.csv"
MAPPING_FILE = GT_DIR / "mapping.csv"


@pytest.fixture(scope="module")
def labels():
    """Load manual labels."""
    df = pd.read_csv(LABELS_FILE)
    # Exclude invalid faces
    df = df[df['person_id'] > 0]
    return df


@pytest.fixture(scope="module")
def mapping():
    """Load crop → source mapping."""
    return pd.read_csv(MAPPING_FILE)


@pytest.fixture(scope="module")
def crop_embeddings(labels):
    """Load embeddings from metadata (saved when crops were created)."""
    import json

    metadata_file = GT_DIR / "metadata.json"
    metadata = json.load(open(metadata_file))

    embeddings = {}

    print("\n" + "="*70)
    print("LOADING GROUND TRUTH CROP EMBEDDINGS FROM METADATA")
    print("="*70)

    for detection in metadata['detections']:
        crop_file = detection['crop_filename']

        # Skip if not in labels (excluded faces)
        if crop_file not in labels['crop_filename'].values:
            continue

        # Load embedding from metadata
        emb = np.array(detection['embedding'], dtype=np.float32)
        embeddings[crop_file] = emb
        print(f"  {crop_file}: OK")

    print(f"\nTotal: {len(embeddings)}/{len(labels)}")

    return embeddings


@pytest.fixture(scope="module")
def pipeline_embeddings(mapping, labels):
    """Run pipeline on source images and extract embeddings."""
    embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='detection_order')

    source_dir = Path(r"D:\Google_Germany")

    # Group by source image
    images_to_process = {}
    for _, row in mapping.iterrows():
        crop_file = row['crop_filename']

        # Skip if not in labels (excluded faces)
        if crop_file not in labels['crop_filename'].values:
            continue

        source_image = row['source_image']
        face_index = row['face_index']

        if source_image not in images_to_process:
            images_to_process[source_image] = []

        images_to_process[source_image].append((crop_file, face_index))

    print("\n" + "="*70)
    print("RUNNING PIPELINE ON SOURCE IMAGES")
    print("="*70)

    embeddings = {}

    for source_image, crops_list in sorted(images_to_process.items()):
        image_path = source_dir / source_image

        print(f"\n{source_image}:")

        # Load image
        with Image.open(image_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_rgb = np.array(pil_img)

        # Detect faces
        faces = embedder.app.get(img_rgb)
        print(f"  Detected {len(faces)} faces")

        # Extract embeddings for labeled faces
        for crop_file, face_index in crops_list:
            if face_index < len(faces):
                face = faces[face_index]
                emb = face.embedding / np.linalg.norm(face.embedding)
                embeddings[crop_file] = emb
                print(f"    [{face_index}] {crop_file}: OK")
            else:
                print(f"    [{face_index}] {crop_file}: INDEX OUT OF RANGE")

    print(f"\nTotal: {len(embeddings)}/{len(labels)}")

    return embeddings


def cosine_distance(emb1, emb2):
    """Compute cosine distance (1 - similarity)."""
    return 1.0 - np.dot(emb1, emb2)


class TestGroundTruthFresh:
    """Test pipeline against fresh ground truth."""

    def test_all_crops_extracted(self, labels, crop_embeddings):
        """Verify all ground truth crops have embeddings."""
        assert len(crop_embeddings) == len(labels), \
            f"Only {len(crop_embeddings)}/{len(labels)} crops extracted"

    def test_all_pipeline_extracted(self, labels, pipeline_embeddings):
        """Verify pipeline extracted all labeled faces."""
        assert len(pipeline_embeddings) == len(labels), \
            f"Pipeline only extracted {len(pipeline_embeddings)}/{len(labels)} faces"

    def test_embeddings_match(self, crop_embeddings, pipeline_embeddings):
        """Verify pipeline embeddings match ground truth crops."""
        mismatches = []

        for crop_file in crop_embeddings.keys():
            if crop_file not in pipeline_embeddings:
                mismatches.append(f"{crop_file}: not in pipeline embeddings")
                continue

            crop_emb = crop_embeddings[crop_file]
            pipe_emb = pipeline_embeddings[crop_file]

            similarity = np.dot(crop_emb, pipe_emb)

            if similarity < 0.95:
                mismatches.append(f"{crop_file}: similarity={similarity:.3f}")

        if mismatches:
            print("\nEmbedding mismatches:")
            for msg in mismatches:
                print(f"  {msg}")

        assert not mismatches, f"{len(mismatches)} faces have mismatched embeddings"

    def test_same_person_distances(self, labels, pipeline_embeddings):
        """Verify same-person faces have low distances."""
        failures = []

        # For each person
        for person_id in sorted(labels['person_id'].unique()):
            person_crops = labels[labels['person_id'] == person_id]['crop_filename'].values

            if len(person_crops) < 2:
                continue

            # Compute all pairwise distances
            max_dist = 0.0
            for i, crop_a in enumerate(person_crops):
                for crop_b in person_crops[i+1:]:
                    if crop_a in pipeline_embeddings and crop_b in pipeline_embeddings:
                        dist = cosine_distance(
                            pipeline_embeddings[crop_a],
                            pipeline_embeddings[crop_b]
                        )
                        max_dist = max(max_dist, dist)

                        if dist > 0.40:  # Same person should be < 0.40
                            failures.append(
                                f"Person {person_id}: {crop_a} <-> {crop_b} = {dist:.3f} (too high)"
                            )

            print(f"  Person {person_id}: max distance = {max_dist:.3f}")

        if failures:
            print("\nSame-person distance failures:")
            for f in failures:
                print(f"  {f}")

        assert not failures, f"{len(failures)} same-person pairs have high distance"

    def test_different_person_distances(self, labels, pipeline_embeddings):
        """Verify different-person faces have high distances."""
        failures = []

        # Get all valid crop files
        all_crops = labels['crop_filename'].values

        # Compute all pairwise distances between different people
        for i, crop_a in enumerate(all_crops):
            for crop_b in all_crops[i+1:]:
                if crop_a not in pipeline_embeddings or crop_b not in pipeline_embeddings:
                    continue

                person_a = labels[labels['crop_filename'] == crop_a]['person_id'].iloc[0]
                person_b = labels[labels['crop_filename'] == crop_b]['person_id'].iloc[0]

                if person_a != person_b:
                    dist = cosine_distance(
                        pipeline_embeddings[crop_a],
                        pipeline_embeddings[crop_b]
                    )

                    if dist < 0.50:  # Different people should be >= 0.50
                        failures.append(
                            f"Person {person_a} vs {person_b}: {crop_a} <-> {crop_b} = {dist:.3f} (too low)"
                        )

        if failures:
            print("\nDifferent-person distance failures:")
            for f in failures[:10]:  # Show first 10
                print(f"  {f}")

        assert not failures, f"{len(failures)} different-person pairs have low distance"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
