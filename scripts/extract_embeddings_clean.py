"""
Clean embedding extraction using the proven isolated method.

Uses the exact same approach that worked in the isolated test:
- Direct image loading from disk
- Fresh InsightFace embedder instance
- No metadata, no cache lookups
- Validation at every step

Usage:
    python scripts/extract_embeddings_clean.py \
        --face-crops results/Google_Germany/face_crops \
        --output results/Google_Germany/embeddings_CLEAN
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm

from face_cluster import InsightFaceEmbedder

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract clean embeddings using proven method')
    parser.add_argument('--face-crops', type=Path, required=True, help='Directory with face_XXXX_aligned.jpg files')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for clean embeddings')
    args = parser.parse_args()

    crops_dir = args.face_crops
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CLEAN EMBEDDING EXTRACTION")
    print("="*70)
    print(f"Face crops: {crops_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find all face crops
    crop_files = sorted(crops_dir.glob("face_*_aligned.jpg"))
    print(f"Found {len(crop_files)} face crop files")

    if len(crop_files) == 0:
        print("ERROR: No face crop files found!")
        return

    # Extract face IDs from filenames
    face_ids = []
    for crop_file in crop_files:
        # Parse face_XXXX_aligned.jpg -> XXXX
        face_id = int(crop_file.stem.split('_')[1])
        face_ids.append(face_id)

    print(f"Face ID range: {min(face_ids)} to {max(face_ids)}")
    print()

    # Initialize embedder (fresh instance, no cached state)
    print("Initializing InsightFace embedder...")
    embedder = InsightFaceEmbedder(model_name='buffalo_l')
    print("OK Embedder ready")
    print()

    # Extract embeddings
    print("Extracting embeddings (this will take a few minutes)...")
    print("-" * 70)

    embeddings = {}
    metadata = []
    failed = []

    for crop_file in tqdm(crop_files, desc="Processing faces"):
        # Parse face ID
        face_id = int(crop_file.stem.split('_')[1])

        try:
            # Load image directly from disk
            img_pil = Image.open(crop_file)
            img_np = np.array(img_pil)

            # Ensure RGB
            if len(img_np.shape) == 2:
                import cv2
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                import cv2
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

            # Extract embedding (using the proven method)
            embedding = embedder.get_embedding(img_np)

            if embedding is None:
                failed.append((face_id, "Failed to extract embedding"))
                continue

            # Store
            embeddings[face_id] = embedding

            # Store metadata
            metadata.append({
                'face_id': face_id,
                'crop_file': crop_file.name,
                'embedding_norm': float(np.linalg.norm(embedding)),
                'extraction_method': 'InsightFaceEmbedder.get_embedding',
                'model': 'buffalo_l'
            })

        except Exception as e:
            failed.append((face_id, str(e)))
            print(f"\nERROR processing face {face_id}: {e}")

    print()
    print(f"Successfully extracted: {len(embeddings)} embeddings")
    if failed:
        print(f"Failed: {len(failed)} faces")
        for face_id, error in failed[:5]:
            print(f"  - Face {face_id}: {error}")

    # Convert to numpy array (ordered by face_id)
    all_face_ids = sorted(embeddings.keys())
    embeddings_array = np.array([embeddings[fid] for fid in all_face_ids], dtype=np.float32)

    print()
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Dtype: {embeddings_array.dtype}")

    # Normalize (L2 norm = 1.0)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms

    print(f"Normalized: all norms = 1.0")

    # Save embeddings
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    embeddings_file = output_dir / f"embeddings_CLEAN_{timestamp}.npy"
    np.save(embeddings_file, embeddings_array)
    print(f"\nOK Saved embeddings: {embeddings_file.name}")

    # Save metadata
    metadata_file = output_dir / f"embeddings_metadata_CLEAN_{timestamp}.json"
    metadata_dict = {
        'timestamp': timestamp,
        'extraction_method': 'clean_isolated_method',
        'face_crops_dir': str(crops_dir),
        'n_faces': len(embeddings),
        'n_failed': len(failed),
        'face_id_range': [min(all_face_ids), max(all_face_ids)],
        'model': 'buffalo_l',
        'faces': metadata,
        'failed': [{'face_id': fid, 'error': err} for fid, err in failed] if failed else []
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"OK Saved metadata: {metadata_file.name}")

    # Validation: Compare against old corrupted embeddings
    print()
    print("="*70)
    print("VALIDATION: Comparing against corrupted source")
    print("="*70)

    old_embeddings_file = crops_dir.parent / "embeddings_FRESH_2026-03-23_01-36-50.npy"
    if old_embeddings_file.exists():
        old_embeddings = np.load(old_embeddings_file)
        old_embeddings = old_embeddings / np.linalg.norm(old_embeddings, axis=1, keepdims=True)

        # Check if identical (BAD) or different (GOOD)
        if np.allclose(embeddings_array, old_embeddings, atol=1e-6):
            print("ERROR: Embeddings are IDENTICAL to corrupted source!")
            print("       Extraction did NOT work - still loading cached data!")
        else:
            print("OK Embeddings are DIFFERENT from corrupted source!")

            # Test specific faces we know are wrong
            test_faces = [545, 546, 550, 551, 557, 569, 573]
            print("\nTest faces (should be different):")

            for face_id in test_faces:
                if face_id < len(old_embeddings) and face_id < len(embeddings_array):
                    similarity = np.dot(old_embeddings[face_id], embeddings_array[face_id])
                    status = "ERROR" if similarity > 0.95 else "OK"
                    print(f"  Face {face_id}: similarity = {similarity:.4f} [{status}]")

            # Test the critical case: 545 vs 546
            if 545 < len(embeddings_array) and 546 < len(embeddings_array):
                dist_old = 1.0 - np.dot(old_embeddings[545], old_embeddings[546])
                dist_new = 1.0 - np.dot(embeddings_array[545], embeddings_array[546])

                print(f"\nCritical test - Face 545 vs 546:")
                print(f"  OLD (corrupted): {dist_old:.4f} (very similar - WRONG)")
                print(f"  NEW (clean):     {dist_new:.4f} (should be > 0.60)")

                if dist_new > 0.60:
                    print(f"  => OK: New embeddings show correct distance!")
                else:
                    print(f"  => WARNING: Distance still low, may still be corrupted")

    else:
        print("NOTE: Old embeddings file not found, skipping comparison")

    # Final summary
    print()
    print("="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Output files:")
    print(f"  - {embeddings_file.name}")
    print(f"  - {metadata_file.name}")
    print()
    print(f"Next steps:")
    print(f"  1. Verify validation shows 'DIFFERENT from corrupted source'")
    print(f"  2. Run clustering with clean embeddings:")
    print(f"     python scripts/export_clustering_data.py \\")
    print(f"       --embeddings {embeddings_file} \\")
    print(f"       --output results/Google_Germany/clustering_FINAL \\")
    print(f"       --k 5 --distance-threshold 0.35 --no-pose --blur-min 0")
    print()

if __name__ == '__main__':
    main()
