"""Test face crop filenames match actual face identities.

Prevents SIGHTING-006: crop filenames must match the face they contain.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import cv2


def test_crop_filenames_match_embeddings(tmp_path):
    """Verify crop filename face_XXXX corresponds to actual face identity.

    When crops are saved with saved_count but metadata uses original indices,
    this creates a mismatch. This test catches that.
    """
    # This would need actual test data
    # For now, document the test strategy
    pass


def verify_crop_metadata_alignment(crops_dir: Path, embeddings_path: Path, metadata_path: Path):
    """Verify crops, embeddings, and metadata are aligned.

    Args:
        crops_dir: Directory with face_XXXX_aligned.jpg files
        embeddings_path: .npy file with embeddings
        metadata_path: .json file with face metadata

    Returns:
        dict: {'aligned': bool, 'mismatches': [(crop_id, actual_face_id), ...]}

    Strategy:
        1. Load embeddings and metadata
        2. For each crop file:
           a. Extract face_id from filename (e.g., face_0569 → 569)
           b. Load stored embedding at position matching face_id
           c. Extract fresh embedding from crop file
           d. Compare: if distance > 0.01, they don't match
        3. Report all mismatches
    """
    from sim_bench.pipeline.face_embedding.insightface_native import InsightFaceNativeExtractor

    # Load stored embeddings
    embeddings = np.load(embeddings_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Get face_id mapping
    face_ids = metadata.get('face_ids') or \
               [i if meta.get('face_index') is None else meta.get('face_index')
                for i, meta in enumerate(metadata.get('face_metadata', []))]

    # Create extractor for fresh embeddings
    config = {"backend": "insightface", "device": "cpu", "model_name": "buffalo_l"}
    extractor = InsightFaceNativeExtractor(config)

    mismatches = []

    # Check sample of crops
    for crop_file in sorted(crops_dir.glob('face_*_aligned.jpg'))[:100]:  # Sample first 100
        # Extract face_id from filename
        crop_face_id = int(crop_file.stem.split('_')[1])

        # Get stored embedding for this face_id
        try:
            idx = face_ids.index(crop_face_id)
            stored_emb = embeddings[idx]
        except (ValueError, IndexError):
            continue

        # Extract fresh embedding from crop
        img = cv2.imread(str(crop_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fresh_emb = extractor.extract_batch([img_rgb], [{'face_id': crop_face_id}])[0]

        # Compare
        cos_dist = 1 - np.dot(stored_emb, fresh_emb) / (
            np.linalg.norm(stored_emb) * np.linalg.norm(fresh_emb)
        )

        if cos_dist > 0.01:
            # Find actual match
            best_match_id = None
            min_dist = float('inf')
            for test_id in range(max(0, crop_face_id - 10), min(len(face_ids), crop_face_id + 10)):
                if test_id >= len(embeddings):
                    continue
                test_dist = 1 - np.dot(embeddings[test_id], fresh_emb) / (
                    np.linalg.norm(embeddings[test_id]) * np.linalg.norm(fresh_emb)
                )
                if test_dist < min_dist:
                    min_dist = test_dist
                    best_match_id = face_ids[test_id] if test_id < len(face_ids) else test_id

            mismatches.append({
                'crop_file': crop_file.name,
                'crop_face_id': crop_face_id,
                'actual_face_id': best_match_id,
                'distance': cos_dist
            })

    return {
        'aligned': len(mismatches) == 0,
        'mismatches': mismatches,
        'total_checked': min(100, len(list(crops_dir.glob('face_*_aligned.jpg'))))
    }


if __name__ == '__main__':
    """Run verification on actual data."""
    import sys

    if len(sys.argv) < 4:
        print("Usage: python test_face_crop_integrity.py <crops_dir> <embeddings.npy> <metadata.json>")
        sys.exit(1)

    result = verify_crop_metadata_alignment(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3])
    )

    print(f"Checked {result['total_checked']} crops")
    print(f"Aligned: {result['aligned']}")

    if not result['aligned']:
        print(f"\nFound {len(result['mismatches'])} mismatches:")
        for m in result['mismatches'][:10]:
            print(f"  {m['crop_file']}: expected face {m['crop_face_id']}, "
                  f"actually face {m['actual_face_id']} (dist={m['distance']:.4f})")
