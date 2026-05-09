"""Sweep cluster_selection_epsilon values and optionally filter by pose.

Usage:
    python scripts/clustering_sweep.py --results_dir results/face_clustering_benchmark \
        --epsilons 0,0.05,0.1,0.15 \
        --max_yaw 45 --max_pitch 30
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import hdbscan
except ImportError:
    print("HDBSCAN not installed. Run: pip install hdbscan")
    exit(1)


def load_data(results_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    """Load embeddings and face metadata from benchmark results."""
    # Load embeddings
    npy_files = sorted(results_dir.glob("embeddings_*.npy"), reverse=True)
    if not npy_files:
        raise FileNotFoundError(f"No embeddings NPY found in {results_dir}")
    embeddings = np.load(npy_files[0])
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load metadata
    json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    if not json_files:
        raise FileNotFoundError(f"No benchmark JSON found in {results_dir}")

    with open(json_files[0]) as f:
        data = json.load(f)

    metadata = data.get("face_metadata", [])
    print(f"Loaded metadata for {len(metadata)} faces")

    return embeddings, metadata


def filter_by_pose(
    embeddings: np.ndarray,
    metadata: List[dict],
    max_yaw: float,
    max_pitch: float
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Filter faces by yaw and pitch angles.

    Returns:
        filtered_embeddings: Embeddings for faces that pass the filter
        kept_indices: Original indices of kept faces
        filtered_indices: Original indices of filtered-out faces
    """
    kept_indices = []
    filtered_indices = []

    for i, meta in enumerate(metadata):
        yaw = abs(meta.get("yaw_angle", 0) or 0)
        pitch = abs(meta.get("pitch_angle", 0) or 0)

        if yaw <= max_yaw and pitch <= max_pitch:
            kept_indices.append(i)
        else:
            filtered_indices.append(i)

    filtered_embeddings = embeddings[kept_indices]
    print(f"Pose filter: kept {len(kept_indices)}, filtered {len(filtered_indices)} "
          f"(max_yaw={max_yaw}, max_pitch={max_pitch})")

    return filtered_embeddings, kept_indices, filtered_indices


def run_hdbscan(
    embeddings: np.ndarray,
    epsilon: float,
    min_cluster_size: int = 2,
    min_samples: int = 2
) -> Tuple[np.ndarray, int, int]:
    """Run HDBSCAN with given epsilon.

    Uses euclidean distance on L2-normalized embeddings, which is
    equivalent to sqrt(2 * (1 - cosine_similarity)).

    The epsilon for euclidean on L2-norm relates to cosine distance as:
    d_euc = sqrt(2 * d_cos), so d_cos = d_euc^2 / 2

    Returns:
        labels, n_clusters, n_noise
    """
    # L2-normalize embeddings (should already be normalized, but ensure)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    # Convert cosine epsilon to euclidean epsilon
    # d_euc = sqrt(2 * d_cos) -> for d_cos=0.1, d_euc=0.447
    # But if epsilon is already meant for euclidean, use as-is
    # We'll use euclidean directly since that's what HDBSCAN supports best

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(normalized)

    n_clusters = len(set(labels) - {-1})
    n_noise = sum(1 for l in labels if l == -1)

    return labels, n_clusters, n_noise


def compute_cluster_stats(labels: np.ndarray) -> dict:
    """Compute cluster statistics."""
    unique_labels = set(labels) - {-1}
    cluster_sizes = [sum(1 for l in labels if l == c) for c in unique_labels]

    return {
        "n_clusters": len(unique_labels),
        "n_noise": sum(1 for l in labels if l == -1),
        "cluster_sizes": sorted(cluster_sizes, reverse=True),
        "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "singletons": sum(1 for s in cluster_sizes if s == 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep HDBSCAN epsilon values")
    parser.add_argument("--results_dir", type=Path, default=Path("results/face_clustering_benchmark"))
    parser.add_argument("--epsilons", type=str, default="0,0.2,0.3,0.4,0.5",
                        help="Comma-separated epsilon values (euclidean on L2-norm). "
                             "Conversion: d_euc = sqrt(2 * d_cos), so 0.3 euc ~ 0.045 cos")
    parser.add_argument("--max_yaw", type=float, default=None,
                        help="Max yaw angle (degrees) - filter faces above this")
    parser.add_argument("--max_pitch", type=float, default=None,
                        help="Max pitch angle (degrees) - filter faces above this")
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--min_samples", type=int, default=2)
    args = parser.parse_args()

    # Parse epsilons
    epsilons = [float(e.strip()) for e in args.epsilons.split(",")]

    # Load data
    embeddings, metadata = load_data(args.results_dir)

    # Apply pose filter if specified
    kept_indices = list(range(len(embeddings)))
    filtered_indices = []

    if args.max_yaw is not None or args.max_pitch is not None:
        max_yaw = args.max_yaw if args.max_yaw is not None else 180
        max_pitch = args.max_pitch if args.max_pitch is not None else 180
        embeddings, kept_indices, filtered_indices = filter_by_pose(
            embeddings, metadata, max_yaw, max_pitch
        )

    print("\n" + "=" * 70)
    print("HDBSCAN EPSILON SWEEP")
    print("=" * 70)
    print(f"Embeddings: {embeddings.shape[0]} faces")
    print(f"min_cluster_size: {args.min_cluster_size}")
    print(f"min_samples: {args.min_samples}")
    if filtered_indices:
        print(f"Pose-filtered: {len(filtered_indices)} faces excluded")
    print("=" * 70)
    print()

    results = []

    for eps in epsilons:
        labels, n_clusters, n_noise = run_hdbscan(
            embeddings, eps,
            args.min_cluster_size,
            args.min_samples
        )
        stats = compute_cluster_stats(labels)

        results.append({
            "epsilon": eps,
            "labels": labels,
            **stats
        })

        print(f"epsilon={eps:.3f}: {n_clusters:3d} clusters, {n_noise:3d} noise, "
              f"sizes={stats['cluster_sizes'][:5]}{'...' if len(stats['cluster_sizes']) > 5 else ''}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Epsilon':>10} | {'Clusters':>8} | {'Noise':>6} | {'Max Size':>8} | {'Avg Size':>8} | {'Singletons':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['epsilon']:>10.3f} | {r['n_clusters']:>8} | {r['n_noise']:>6} | "
              f"{r['max_cluster_size']:>8} | {r['avg_cluster_size']:>8.1f} | {r['singletons']:>10}")

    # Show pose distribution if we have the data
    if metadata and any(m.get("yaw_angle") for m in metadata):
        print("\n" + "=" * 70)
        print("POSE DISTRIBUTION (before filtering)")
        print("=" * 70)

        yaws = [abs(m.get("yaw_angle", 0) or 0) for m in metadata]
        pitches = [abs(m.get("pitch_angle", 0) or 0) for m in metadata]

        print(f"Yaw:   min={min(yaws):.1f}, median={np.median(yaws):.1f}, "
              f"max={max(yaws):.1f}, >45deg: {sum(1 for y in yaws if y > 45)}")
        print(f"Pitch: min={min(pitches):.1f}, median={np.median(pitches):.1f}, "
              f"max={max(pitches):.1f}, >30deg: {sum(1 for p in pitches if p > 30)}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find epsilon that gives reasonable cluster count
    target_clusters = len(embeddings) // 5  # Rough heuristic: ~5 faces per person
    best = min(results, key=lambda r: abs(r['n_clusters'] - target_clusters))

    print(f"Based on ~5 faces/person heuristic (target ~{target_clusters} clusters):")
    print(f"  Best epsilon: {best['epsilon']:.3f} ({best['n_clusters']} clusters)")

    # Also show current config value for comparison
    print(f"\nCurrent config uses epsilon=0.045")
    current = next((r for r in results if abs(r['epsilon'] - 0.045) < 0.001), None)
    if current:
        print(f"  -> {current['n_clusters']} clusters, {current['n_noise']} noise")


if __name__ == "__main__":
    main()
