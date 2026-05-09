"""
Direct InsightFace extraction bypassing all custom wrappers.

This script loads InsightFace directly and extracts embeddings
without using any face_cluster code, to rule out any caching/corruption
in our wrapper classes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from tqdm import tqdm

def main():
    # Initialize InsightFace directly
    print("="*70)
    print("DIRECT INSIGHTFACE EXTRACTION (NO WRAPPERS)")
    print("="*70)
    print()

    import insightface
    print("Loading InsightFace model...")
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("Model loaded\n")

    # Find face crops
    crops_dir = Path("results/Google_Germany/face_crops")
    crop_files = sorted(crops_dir.glob("face_*_aligned.jpg"))
    print(f"Found {len(crop_files)} face crops\n")

    # Extract face IDs
    face_ids = []
    for crop_file in crop_files:
        face_id = int(crop_file.stem.split('_')[1])
        face_ids.append(face_id)

    print(f"Face ID range: {min(face_ids)} to {max(face_ids)}\n")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = {}

    for crop_file in tqdm(crop_files, desc="Processing"):
        face_id = int(crop_file.stem.split('_')[1])

        # Load image
        img_bgr = cv2.imread(str(crop_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to 112x112 (ArcFace standard)
        img_resized = cv2.resize(img_rgb, (112, 112))

        # Convert back to BGR for InsightFace
        img_bgr_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        # Extract embedding using recognition model directly
        embedding = app.models['recognition'].get_feat(img_bgr_resized)

        # Ensure 1D and normalized
        embedding = np.asarray(embedding).flatten()
        embedding = embedding / np.linalg.norm(embedding)

        embeddings[face_id] = embedding

    # Convert to array
    all_face_ids = sorted(embeddings.keys())
    embeddings_array = np.array([embeddings[fid] for fid in all_face_ids], dtype=np.float32)

    print(f"\nExtracted {len(embeddings)} embeddings")
    print(f"Shape: {embeddings_array.shape}\n")

    # Save
    output_dir = Path("results/Google_Germany/embeddings_DIRECT")
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = output_dir / f"embeddings_DIRECT_{timestamp}.npy"

    np.save(output_file, embeddings_array)
    print(f"Saved to: {output_file}\n")

    # Validate
    print("="*70)
    print("VALIDATION")
    print("="*70)

    # Load what we just saved
    loaded = np.load(output_file)
    print(f"Loaded shape: {loaded.shape}")
    print(f"Face 545: {loaded[545, :5]}")

    # Compare to corrupted
    old_file = crops_dir.parent / "embeddings_FRESH_2026-03-23_01-36-50.npy"
    if old_file.exists():
        old_emb = np.load(old_file)
        print(f"\nOld (corrupted) face 545: {old_emb[545, :5]}")

        if np.allclose(loaded, old_emb):
            print("\nERROR: IDENTICAL to corrupted file!")
        else:
            print("\nOK: DIFFERENT from corrupted file!")

            # Test critical pair
            dist_545_546 = 1.0 - np.dot(loaded[545], loaded[546])
            print(f"\nFace 545 vs 546 distance: {dist_545_546:.4f}")
            if dist_545_546 > 0.60:
                print("=> CORRECT: Different people")
            else:
                print("=> WRONG: Too similar")

if __name__ == '__main__':
    main()
