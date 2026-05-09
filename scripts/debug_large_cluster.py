"""Debug large cluster by analyzing with UMAP and PCA.

Investigates whether the large over-merged cluster can be split.
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    print("UMAP not installed. Run: pip install umap-learn")
    exit(1)

try:
    import hdbscan
except ImportError:
    print("HDBSCAN not installed. Run: pip install hdbscan")
    exit(1)


def load_data(results_dir: Path) -> Tuple[np.ndarray, List[dict], np.ndarray]:
    """Load embeddings, metadata, and get HDBSCAN labels."""
    # Load embeddings
    npy_files = sorted(results_dir.glob("embeddings_*.npy"), reverse=True)
    if not npy_files:
        raise FileNotFoundError(f"No embeddings NPY found in {results_dir}")
    embeddings = np.load(npy_files[0])

    # Load metadata
    json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    if not json_files:
        raise FileNotFoundError(f"No benchmark JSON found in {results_dir}")

    with open(json_files[0]) as f:
        data = json.load(f)

    metadata = data.get("face_metadata", [])

    # Run HDBSCAN to get the labels
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=2,
        cluster_selection_epsilon=0.0,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(normalized)

    return embeddings, metadata, labels


def get_face_crop_base64(results_dir: Path, face_idx: int) -> str:
    """Get face crop as base64 for HTML embedding."""
    crops_dir = results_dir / "face_crops"

    for pattern in [f"face_{face_idx:04d}_aligned.jpg", f"face_{face_idx:04d}.jpg"]:
        path = crops_dir / pattern
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
    return ""


def run_analysis(
    embeddings: np.ndarray,
    name: str,
    n_clusters: int = 2
) -> dict:
    """Run UMAP and clustering on embeddings."""
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(normalized)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(normalized)

    # HDBSCAN on UMAP coords (might find natural clusters)
    hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    hdbscan_labels = hdb.fit_predict(umap_coords)

    return {
        "name": name,
        "umap_coords": umap_coords,
        "kmeans_labels": kmeans_labels,
        "hdbscan_labels": hdbscan_labels,
        "kmeans_sizes": [sum(kmeans_labels == i) for i in range(n_clusters)],
        "hdbscan_n_clusters": len(set(hdbscan_labels) - {-1}),
    }


def generate_html_report(
    results_dir: Path,
    cluster_indices: List[int],
    analyses: List[dict],
    output_path: Path
) -> None:
    """Generate HTML report with visualizations."""

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Large Cluster Debug Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 10px; }
        h2 { color: #8b949e; margin: 30px 0 15px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
        h3 { color: #c9d1d9; margin: 20px 0 10px; }
        .meta { color: #8b949e; font-size: 0.9em; margin-bottom: 20px; }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .scatter-container {
            position: relative;
            width: 100%;
            height: 400px;
            background: #1a1a2e;
            border-radius: 6px;
            overflow: hidden;
        }
        .scatter-point {
            position: absolute;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            cursor: pointer;
            transition: transform 0.1s;
        }
        .scatter-point:hover {
            transform: translate(-50%, -50%) scale(2);
            z-index: 100;
        }
        .cluster-0 { background: #4CAF50; }
        .cluster-1 { background: #2196F3; }
        .cluster-2 { background: #f44336; }
        .cluster-3 { background: #FF9800; }
        .cluster-4 { background: #9C27B0; }
        .cluster-noise { background: #666; }
        .stats {
            display: flex;
            gap: 20px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .stat-box {
            background: #21262d;
            padding: 10px 15px;
            border-radius: 4px;
        }
        .face-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 10px 0;
            max-height: 300px;
            overflow-y: auto;
        }
        .face-item {
            width: 50px;
            height: 50px;
            border-radius: 4px;
            object-fit: cover;
        }
        .face-item.cluster-0 { border: 2px solid #4CAF50; }
        .face-item.cluster-1 { border: 2px solid #2196F3; }
        .legend {
            display: flex;
            gap: 15px;
            margin: 10px 0;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Large Cluster Debug Analysis</h1>
        <div class="meta">
            Generated: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f'''<br>
            Analyzing cluster with {len(cluster_indices)} faces
        </div>
'''

    for analysis in analyses:
        name = analysis["name"]
        umap_coords = analysis["umap_coords"]
        kmeans_labels = analysis["kmeans_labels"]
        hdbscan_labels = analysis["hdbscan_labels"]
        kmeans_sizes = analysis["kmeans_sizes"]
        hdbscan_n = analysis["hdbscan_n_clusters"]

        # Normalize UMAP coords to 0-100% for positioning
        x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
        y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        html += f'''
        <h2>{name}</h2>
        <div class="card">
            <div class="stats">
                <div class="stat-box">K-Means Split: {kmeans_sizes[0]} / {kmeans_sizes[1]}</div>
                <div class="stat-box">HDBSCAN on UMAP: {hdbscan_n} clusters</div>
            </div>

            <h3>UMAP Projection (colored by K-Means 2-cluster)</h3>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot cluster-0"></div> Cluster 0 ({kmeans_sizes[0]})</div>
                <div class="legend-item"><div class="legend-dot cluster-1"></div> Cluster 1 ({kmeans_sizes[1]})</div>
            </div>
            <div class="scatter-container">
'''
        # Add scatter points
        for i, (idx, label) in enumerate(zip(cluster_indices, kmeans_labels)):
            x_pct = 5 + 90 * (umap_coords[i, 0] - x_min) / x_range
            y_pct = 5 + 90 * (umap_coords[i, 1] - y_min) / y_range
            cluster_class = f"cluster-{label}" if label >= 0 else "cluster-noise"
            html += f'<div class="scatter-point {cluster_class}" style="left:{x_pct}%;top:{y_pct}%" title="Face #{idx}"></div>\n'

        html += '''
            </div>

            <h3>Faces by K-Means Cluster</h3>
'''
        # Show faces for each cluster
        for cluster_id in range(2):
            cluster_face_indices = [cluster_indices[i] for i, l in enumerate(kmeans_labels) if l == cluster_id]
            html += f'''
            <h4>Cluster {cluster_id} ({len(cluster_face_indices)} faces)</h4>
            <div class="face-grid">
'''
            for idx in cluster_face_indices[:50]:  # Limit to 50 faces
                b64 = get_face_crop_base64(results_dir, idx)
                if b64:
                    html += f'<img class="face-item cluster-{cluster_id}" src="data:image/jpeg;base64,{b64}" title="#{idx}">\n'
                else:
                    html += f'<div class="face-item cluster-{cluster_id}" style="background:#333;display:flex;align-items:center;justify-content:center;font-size:10px">#{idx}</div>\n'

            if len(cluster_face_indices) > 50:
                html += f'<div style="padding:10px;color:#8b949e">... and {len(cluster_face_indices) - 50} more</div>'
            html += '</div>\n'

        html += '</div>\n'

    html += '''
    </div>
</body>
</html>
'''

    output_path.write_text(html, encoding='utf-8')
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug large cluster analysis")
    parser.add_argument("--results_dir", type=Path, default=Path("results/face_clustering_benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target_cluster", type=int, default=None,
                        help="Cluster ID to analyze (default: largest cluster)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "large_cluster_debug.html"

    # Load data
    print("Loading data...")
    embeddings, metadata, labels = load_data(args.results_dir)
    print(f"Loaded {len(embeddings)} embeddings")

    # Find largest cluster
    unique_labels = set(labels) - {-1}
    cluster_sizes = {l: sum(labels == l) for l in unique_labels}

    if args.target_cluster is not None:
        target_cluster = args.target_cluster
    else:
        target_cluster = max(cluster_sizes, key=cluster_sizes.get)

    print(f"\nCluster sizes: {sorted(cluster_sizes.values(), reverse=True)}")
    print(f"Analyzing cluster {target_cluster} with {cluster_sizes[target_cluster]} faces")

    # Get indices of faces in this cluster
    cluster_mask = labels == target_cluster
    cluster_indices = np.where(cluster_mask)[0].tolist()
    cluster_embeddings = embeddings[cluster_mask]

    print(f"\nRunning analyses on {len(cluster_indices)} faces...")

    analyses = []

    # 1. Raw embeddings -> UMAP
    print("  1/3: Raw embeddings -> UMAP")
    analyses.append(run_analysis(cluster_embeddings, "Raw Embeddings (512-dim) -> UMAP"))

    # 2. PCA-128 -> UMAP (or max components if fewer samples)
    n_pca_128 = min(128, len(cluster_embeddings) - 1)
    print(f"  2/3: PCA-{n_pca_128} -> UMAP")
    pca_128 = PCA(n_components=n_pca_128, random_state=42)
    emb_pca_128 = pca_128.fit_transform(cluster_embeddings)
    print(f"       PCA-{n_pca_128} explained variance: {sum(pca_128.explained_variance_ratio_):.2%}")
    analyses.append(run_analysis(emb_pca_128, f"PCA-{n_pca_128} ({sum(pca_128.explained_variance_ratio_):.1%} var) -> UMAP"))

    # 3. PCA-256 -> UMAP (or max components if fewer samples)
    n_pca_256 = min(256, len(cluster_embeddings) - 1)
    print(f"  3/3: PCA-{n_pca_256} -> UMAP")
    pca_256 = PCA(n_components=n_pca_256, random_state=42)
    emb_pca_256 = pca_256.fit_transform(cluster_embeddings)
    print(f"       PCA-{n_pca_256} explained variance: {sum(pca_256.explained_variance_ratio_):.2%}")
    analyses.append(run_analysis(emb_pca_256, f"PCA-{n_pca_256} ({sum(pca_256.explained_variance_ratio_):.1%} var) -> UMAP"))

    # Generate report
    print("\nGenerating HTML report...")
    generate_html_report(args.results_dir, cluster_indices, analyses, args.output)


if __name__ == "__main__":
    main()
