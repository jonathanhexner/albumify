"""
Diagnose embedding mismatch for specific faces.

Tests:
1. Load stored embeddings from export
2. Load face crops from disk
3. Compute FRESH embeddings directly from crops
4. Compare stored vs fresh distances
5. Show which embeddings are corrupted
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Use face_cluster's embedder which handles this correctly
from face_cluster import InsightFaceEmbedder

def load_stored_embeddings(export_dir: Path):
    """Load embeddings that were used for clustering."""
    # Try multiple possible embedding files
    npy_files = list(export_dir.parent.glob("embeddings*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No embeddings*.npy in {export_dir.parent}")

    # Use the most recent
    npy_file = sorted(npy_files, key=lambda p: p.stat().st_mtime)[-1]
    print(f"Loading stored embeddings from: {npy_file.name}")

    embeddings = np.load(npy_file)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def compute_fresh_embeddings(face_ids: list, crops_dir: Path):
    """Compute fresh embeddings directly from face crops on disk."""
    print("\nComputing FRESH embeddings from crops...")

    # Initialize InsightFaceEmbedder
    embedder = InsightFaceEmbedder(model_name='buffalo_l')

    fresh_embeddings = {}

    for face_id in face_ids:
        crop_path = crops_dir / f"face_{face_id:04d}_aligned.jpg"
        if not crop_path.exists():
            print(f"  ERROR Crop not found: {crop_path.name}")
            continue

        # Load crop as PIL Image
        img_pil = Image.open(crop_path)
        img_np = np.array(img_pil)

        # Ensure RGB
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        # Extract embedding from aligned crop using get_embedding
        # For 112x112 crops, it will use recognition model directly
        embedding = embedder.get_embedding(img_np)

        if embedding is None:
            print(f"  ERROR Failed to extract embedding: {crop_path.name}")
            continue

        fresh_embeddings[face_id] = embedding
        print(f"  OK face_{face_id:04d}: embedding shape {embedding.shape}")

    return fresh_embeddings

def compute_distance_matrix(embeddings_dict: dict):
    """Compute pairwise cosine distances."""
    face_ids = sorted(embeddings_dict.keys())
    n = len(face_ids)

    distances = np.zeros((n, n))

    for i, id_a in enumerate(face_ids):
        for j, id_b in enumerate(face_ids):
            if i == j:
                distances[i, j] = 0.0
            else:
                # Cosine distance = 1 - cosine_similarity
                cos_sim = np.dot(embeddings_dict[id_a], embeddings_dict[id_b])
                distances[i, j] = 1.0 - cos_sim

    return distances, face_ids

def main():
    export_dir = Path("results/Google_Germany/ground_truth_labeling")
    crops_dir = export_dir / "face_crops"

    print("="*70)
    print("EMBEDDING MISMATCH DIAGNOSTIC TEST")
    print("="*70)

    # Test faces from user report
    test_faces = [545, 546, 550, 551, 557, 569, 573]
    print(f"\nTesting faces: {test_faces}")
    print(f"User report:")
    print(f"  - Person 1: 545, 569, 573")
    print(f"  - Person 2: 550, 551, 557")
    print(f"  - Person 3: rest")
    print(f"  - Problem: face 545 shows closest to 546 (different person)")

    # Load stored embeddings
    stored_embeddings_all = load_stored_embeddings(export_dir)
    print(f"Loaded {len(stored_embeddings_all)} stored embeddings")

    # Extract stored embeddings for test faces
    stored_embeddings = {fid: stored_embeddings_all[fid] for fid in test_faces}

    # Compute fresh embeddings
    fresh_embeddings = compute_fresh_embeddings(test_faces, crops_dir)

    if len(fresh_embeddings) != len(test_faces):
        print(f"\nERROR WARNING: Only computed {len(fresh_embeddings)}/{len(test_faces)} fresh embeddings")

    # Compare stored vs fresh for same face
    print("\n" + "="*70)
    print("STORED vs FRESH EMBEDDING COMPARISON (same face)")
    print("="*70)
    print(f"{'Face ID':<10} {'Self-Distance':<15} {'Status'}")
    print("-"*70)

    for face_id in sorted(fresh_embeddings.keys()):
        stored = stored_embeddings[face_id]
        fresh = fresh_embeddings[face_id]

        # Distance between stored and fresh for SAME face (should be ~0)
        self_distance = 1.0 - np.dot(stored, fresh)

        if self_distance < 0.05:
            status = "OK MATCH"
        elif self_distance < 0.15:
            status = "WARN SUSPECT"
        else:
            status = "ERROR CORRUPTED"

        print(f"{face_id:<10} {self_distance:<15.4f} {status}")

    # Compute distance matrices
    print("\n" + "="*70)
    print("PAIRWISE DISTANCES - STORED EMBEDDINGS")
    print("="*70)

    stored_distances, face_ids = compute_distance_matrix(stored_embeddings)

    # Print distance matrix
    print(f"\n{'':>8}", end="")
    for fid in face_ids:
        print(f"{fid:>8}", end="")
    print()
    print("-" * (8 + 8 * len(face_ids)))

    for i, fid_a in enumerate(face_ids):
        print(f"{fid_a:>8}", end="")
        for j, fid_b in enumerate(face_ids):
            dist = stored_distances[i, j]
            if i == j:
                print(f"{'--':>8}", end="")
            else:
                print(f"{dist:>8.3f}", end="")
        print()

    print("\n" + "="*70)
    print("PAIRWISE DISTANCES - FRESH EMBEDDINGS")
    print("="*70)

    fresh_distances, face_ids = compute_distance_matrix(fresh_embeddings)

    # Print distance matrix
    print(f"\n{'':>8}", end="")
    for fid in face_ids:
        print(f"{fid:>8}", end="")
    print()
    print("-" * (8 + 8 * len(face_ids)))

    for i, fid_a in enumerate(face_ids):
        print(f"{fid_a:>8}", end="")
        for j, fid_b in enumerate(face_ids):
            dist = fresh_distances[i, j]
            if i == j:
                print(f"{'--':>8}", end="")
            else:
                print(f"{dist:>8.3f}", end="")
        print()

    # Analyze specific problem: 545 closest to 546
    print("\n" + "="*70)
    print("PROBLEM ANALYSIS: Face 545 distances")
    print("="*70)

    idx_545 = face_ids.index(545)

    print("\nSTORED embeddings - Face 545 distances to others:")
    for j, fid in enumerate(face_ids):
        if fid == 545:
            continue
        dist = stored_distances[idx_545, j]
        person = "Person 1" if fid in [545, 569, 573] else ("Person 2" if fid in [550, 551, 557] else "Person 3")
        print(f"  545 -> {fid}: {dist:.4f} ({person})")

    print("\nFRESH embeddings - Face 545 distances to others:")
    for j, fid in enumerate(face_ids):
        if fid == 545:
            continue
        dist = fresh_distances[idx_545, j]
        person = "Person 1" if fid in [545, 569, 573] else ("Person 2" if fid in [550, 551, 557] else "Person 3")
        print(f"  545 -> {fid}: {dist:.4f} ({person})")

    # Find closest face to 545 in each case
    print("\n" + "="*70)
    print("CLOSEST FACE TO 545")
    print("="*70)

    stored_dists_545 = [(face_ids[j], stored_distances[idx_545, j]) for j in range(len(face_ids)) if j != idx_545]
    stored_dists_545.sort(key=lambda x: x[1])

    fresh_dists_545 = [(face_ids[j], fresh_distances[idx_545, j]) for j in range(len(face_ids)) if j != idx_545]
    fresh_dists_545.sort(key=lambda x: x[1])

    print(f"\nSTORED: Closest to 545 is {stored_dists_545[0][0]} (distance: {stored_dists_545[0][1]:.4f})")
    print(f"FRESH:  Closest to 545 is {fresh_dists_545[0][0]} (distance: {fresh_dists_545[0][1]:.4f})")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    # Check if any embeddings are corrupted
    corrupted = []
    for face_id in sorted(fresh_embeddings.keys()):
        stored = stored_embeddings[face_id]
        fresh = fresh_embeddings[face_id]
        self_distance = 1.0 - np.dot(stored, fresh)
        if self_distance > 0.15:
            corrupted.append((face_id, self_distance))

    if corrupted:
        print(f"\nERROR EMBEDDINGS ARE CORRUPTED!")
        print(f"\nCorrupted faces:")
        for face_id, dist in corrupted:
            print(f"  - Face {face_id}: stored vs fresh distance = {dist:.4f}")
        print(f"\nRECOMMENDATION: Regenerate ALL embeddings from face crops")
        print(f"Command: python scripts/regenerate_embeddings_from_crops.py \\")
        print(f"           --face-crops {crops_dir} \\")
        print(f"           --output {export_dir.parent}")
    else:
        print(f"\nOK Embeddings match crops (no corruption detected)")
        print(f"\nIf clustering still wrong, issue may be:")
        print(f"  1. Face crops themselves are misaligned")
        print(f"  2. Face detection produced wrong faces")
        print(f"  3. Algorithm parameters need adjustment")

if __name__ == '__main__':
    main()
