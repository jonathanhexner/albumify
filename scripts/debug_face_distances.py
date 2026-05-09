"""Debug script to analyze embedding distances between specific faces.

Usage:
    python scripts/debug_face_distances.py --results_dir results/face_clustering_benchmark
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine


def load_embeddings(results_dir: Path) -> np.ndarray:
    """Load embeddings from benchmark results."""
    npy_files = sorted(results_dir.glob("embeddings_*.npy"), reverse=True)
    if not npy_files:
        raise FileNotFoundError(f"No embeddings NPY found in {results_dir}")
    return np.load(npy_files[0])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine_similarity)."""
    return cosine(a, b)


def main():
    parser = argparse.ArgumentParser(description="Debug face embedding distances")
    parser.add_argument("--results_dir", type=Path, default=Path("results/face_clustering_benchmark"))
    args = parser.parse_args()

    embeddings = load_embeddings(args.results_dir)
    print(f"Loaded embeddings: shape={embeddings.shape}")

    # Person 1 faces (should be same person)
    person1 = [0, 81, 79, 82, 91, 135]
    # Person 2 faces (should be same person, different from person 1)
    person2 = [2, 46, 183, 249]

    print("\n" + "=" * 60)
    print("INTRA-PERSON DISTANCES (should be LOW)")
    print("=" * 60)

    print("\n--- Person 1 (faces: 0, 81, 79, 82, 91, 135) ---")
    p1_dists = []
    for i, idx1 in enumerate(person1):
        for idx2 in person1[i+1:]:
            if idx1 < len(embeddings) and idx2 < len(embeddings):
                d = cosine_distance(embeddings[idx1], embeddings[idx2])
                p1_dists.append(d)
                print(f"  {idx1:4d} - {idx2:4d}: {d:.4f}")
    if p1_dists:
        print(f"  Summary: min={min(p1_dists):.4f}, median={np.median(p1_dists):.4f}, max={max(p1_dists):.4f}")

    print("\n--- Person 2 (faces: 2, 46, 183, 249) ---")
    p2_dists = []
    for i, idx1 in enumerate(person2):
        for idx2 in person2[i+1:]:
            if idx1 < len(embeddings) and idx2 < len(embeddings):
                d = cosine_distance(embeddings[idx1], embeddings[idx2])
                p2_dists.append(d)
                print(f"  {idx1:4d} - {idx2:4d}: {d:.4f}")
    if p2_dists:
        print(f"  Summary: min={min(p2_dists):.4f}, median={np.median(p2_dists):.4f}, max={max(p2_dists):.4f}")

    print("\n" + "=" * 60)
    print("INTER-PERSON DISTANCES (should be HIGH)")
    print("=" * 60)

    print("\n--- Person 1 vs Person 2 ---")
    cross_dists = []
    for idx1 in person1[:3]:  # Sample first 3 from each
        for idx2 in person2[:3]:
            if idx1 < len(embeddings) and idx2 < len(embeddings):
                d = cosine_distance(embeddings[idx1], embeddings[idx2])
                cross_dists.append(d)
                print(f"  {idx1:4d} - {idx2:4d}: {d:.4f}")
    if cross_dists:
        print(f"  Summary: min={min(cross_dists):.4f}, median={np.median(cross_dists):.4f}, max={max(cross_dists):.4f}")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if p1_dists and p2_dists and cross_dists:
        intra_max = max(max(p1_dists), max(p2_dists))
        inter_min = min(cross_dists)

        print(f"\nMax intra-person distance: {intra_max:.4f}")
        print(f"Min inter-person distance: {inter_min:.4f}")

        if inter_min > intra_max:
            gap = inter_min - intra_max
            print(f"\n[OK] GOOD: Clear separation exists (gap = {gap:.4f})")
            print(f"   Clustering threshold between {intra_max:.3f} and {inter_min:.3f} should work.")
        else:
            overlap = intra_max - inter_min
            print(f"\n[WARN] OVERLAP: Distributions overlap by {overlap:.4f}")
            print("   This means some same-person pairs have higher distance than different-person pairs.")
            print("   Possible causes:")
            print("   - Poor face alignment for some faces")
            print("   - Large pose variation in person 1 or 2")
            print("   - Embedding model limitations")

    # Check for zero/near-zero embeddings
    print("\n" + "=" * 60)
    print("EMBEDDING QUALITY CHECK")
    print("=" * 60)

    all_faces = person1 + person2
    for idx in all_faces:
        if idx < len(embeddings):
            emb = embeddings[idx]
            norm = np.linalg.norm(emb)
            zeros = np.sum(np.abs(emb) < 1e-6)
            print(f"  Face {idx:4d}: norm={norm:.4f}, zeros={zeros}/{len(emb)}")


if __name__ == "__main__":
    main()
