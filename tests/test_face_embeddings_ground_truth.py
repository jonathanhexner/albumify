"""
Ground truth test for face embeddings using distance matrix.

Verifies:
1. Within-person distances match expected variance
2. Between-person distances maintain clear separation (> 0.60)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from face_cluster import InsightFaceEmbedder


# Ground truth labels (user-verified)
LABELS = {
    545: 1, 569: 1, 573: 1, 634: 1, 637: 1,  # Person 1
    550: 2, 551: 2, 558: 2, 580: 2,          # Person 2
    546: 3, 557: 3, 562: 3, 584: 3,          # Person 3 (557 is profile)
    587: 4, 589: 4,                          # Person 4 (new)
}

# Expected max within-person distances (from actual data with 15 faces)
MAX_WITHIN = {
    1: 0.30,  # Person 1: max = 0.299
    2: 0.31,  # Person 2: max = 0.306 (more variance with 4 faces)
    3: 0.63,  # Person 3: max = 0.628 (high variance due to profile)
    4: 0.05,  # Person 4: max = 0.032 (very tight, only 2 faces)
}

MIN_BETWEEN = 0.60  # All cross-person distances should be > 0.60


def extract_embeddings():
    """Extract embeddings for all 7 test faces (NO CACHE)."""
    test_dir = Path("test_data/face_crops_ground_truth")

    # Fresh embedder instance - no cached state
    embedder = InsightFaceEmbedder(model_name='buffalo_l')

    embeddings = {}
    for face_id in LABELS.keys():
        img = np.array(Image.open(test_dir / f"face_{face_id:04d}_aligned.jpg"))

        # Handle grayscale/RGBA
        if len(img.shape) == 2:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        embeddings[face_id] = embedder.get_embedding(img)

    return embeddings


def compute_distance_matrix(embeddings):
    """Compute pairwise cosine distance matrix."""
    face_ids = sorted(embeddings.keys())
    n = len(face_ids)

    distances = np.zeros((n, n))
    for i, fid_a in enumerate(face_ids):
        for j, fid_b in enumerate(face_ids):
            if i != j:
                distances[i, j] = 1.0 - np.dot(embeddings[fid_a], embeddings[fid_b])

    return pd.DataFrame(distances, index=face_ids, columns=face_ids)


@pytest.fixture(scope="module")
def distance_df():
    """Distance matrix as DataFrame."""
    embeddings = extract_embeddings()
    return compute_distance_matrix(embeddings)


def test_within_person_distances(distance_df):
    """Test within-person distances match expected variance."""
    failures = []

    for person_id, max_dist in MAX_WITHIN.items():
        faces = [fid for fid, label in LABELS.items() if label == person_id]

        for i, face_a in enumerate(faces):
            for face_b in faces[i+1:]:
                dist = distance_df.loc[face_a, face_b]
                if dist > max_dist:
                    failures.append(
                        f"Person {person_id}: {face_a}<->{face_b} = {dist:.3f} "
                        f"(exceeds max {max_dist:.2f})"
                    )

    if failures:
        print("\nWithin-person distance failures:")
        for failure in failures:
            print(f"  {failure}")

    assert not failures, f"{len(failures)} within-person distance(s) exceed expected max"


def test_between_people_separation(distance_df):
    """Test all cross-person distances maintain clear separation (> 0.60)."""
    face_ids = list(LABELS.keys())
    failures = []

    for i, face_a in enumerate(face_ids):
        for face_b in face_ids[i+1:]:
            if LABELS[face_a] != LABELS[face_b]:  # Different people
                dist = distance_df.loc[face_a, face_b]
                if dist <= MIN_BETWEEN:
                    failures.append(
                        f"People {LABELS[face_a]} vs {LABELS[face_b]}: "
                        f"{face_a}<->{face_b} = {dist:.3f} "
                        f"(should be > {MIN_BETWEEN:.2f})"
                    )

    if failures:
        print("\nBetween-people separation failures:")
        for failure in failures:
            print(f"  {failure}")

    assert not failures, f"{len(failures)} cross-person distance(s) below separation threshold"


def test_print_distance_matrix(distance_df):
    """Print distance matrix for inspection."""
    print("\n" + "="*70)
    print("DISTANCE MATRIX")
    print("="*70)
    print(distance_df.round(3))

    print("\nGround Truth Labels:")
    for person_id in sorted(set(LABELS.values())):
        faces = [fid for fid, label in LABELS.items() if label == person_id]
        print(f"  Person {person_id}: {faces} (max within: {MAX_WITHIN[person_id]:.2f})")

    print(f"\nSeparation threshold: {MIN_BETWEEN:.2f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
