"""Benchmark HDBSCAN variants with split functionality.

Compares clustering results with different split thresholds.
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    import umap
except ImportError:
    print("UMAP not installed. Run: pip install umap-learn")
    exit(1)

from sim_bench.clustering.hybrid_hdbscan_knn import HybridHDBSCANKNN
from sim_bench.clustering.hybrid_closest_face import HybridHDBSCANClosestFace


def load_data(results_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    """Load embeddings and metadata."""
    npy_files = sorted(results_dir.glob("embeddings_*.npy"), reverse=True)
    if not npy_files:
        raise FileNotFoundError(f"No embeddings NPY found in {results_dir}")
    embeddings = np.load(npy_files[0])

    json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    if not json_files:
        raise FileNotFoundError(f"No benchmark JSON found in {results_dir}")

    with open(json_files[0]) as f:
        data = json.load(f)

    metadata = data.get("face_metadata", [])
    return embeddings, metadata


def get_face_crop_base64(results_dir: Path, face_idx: int) -> str:
    """Get face crop as base64 for HTML embedding."""
    crops_dir = results_dir / "face_crops"
    for pattern in [f"face_{face_idx:04d}_aligned.jpg", f"face_{face_idx:04d}.jpg"]:
        path = crops_dir / pattern
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
    return ""


def run_clustering(
    embeddings: np.ndarray,
    method_class,
    method_name: str,
    split_threshold: float,
    split_enabled: bool = True
) -> Dict[str, Any]:
    """Run clustering with given parameters."""
    config = {
        'params': {
            'split_enabled': split_enabled,
            'split_threshold': split_threshold,
            'split_min_cluster_size': 10,
            'split_k': 20,
        }
    }

    clusterer = method_class(config)
    labels, stats = clusterer.cluster(embeddings, collect_debug_data=True)

    return {
        'method': method_name,
        'split_threshold': split_threshold,
        'split_enabled': split_enabled,
        'labels': labels,
        'n_clusters': stats['n_clusters'],
        'n_noise': stats['n_noise'],
        'total_splits': stats.get('total_splits', 0),
        'cluster_sizes': sorted(stats['cluster_sizes'].values(), reverse=True),
        'stats': stats,
    }


def generate_html_report(
    results_dir: Path,
    embeddings: np.ndarray,
    umap_coords: np.ndarray,
    results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Generate HTML report with clustering comparison."""

    n_faces = len(embeddings)

    # Color palette
    colors = [
        "#4CAF50", "#2196F3", "#f44336", "#FF9800", "#9C27B0",
        "#00BCD4", "#FFEB3B", "#E91E63", "#8BC34A", "#3F51B5",
        "#FF5722", "#607D8B", "#795548", "#009688", "#CDDC39",
        "#673AB7", "#FFC107", "#03A9F4", "#FFEB3B", "#4CAF50"
    ]

    # Normalize UMAP for display
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HDBSCAN Variants with Split - Benchmark</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; margin-bottom: 10px; }}
        h2 {{ color: #8b949e; margin: 30px 0 15px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
        h3 {{ color: #c9d1d9; margin: 15px 0 10px; }}
        .meta {{ color: #8b949e; font-size: 0.9em; margin-bottom: 20px; }}
        .algo-box {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
        }}
        .algo-box h3 {{ color: #58a6ff; margin-bottom: 15px; }}
        .algo-box pre {{
            background: #0d1117;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.9em;
            line-height: 1.5;
        }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
        }}
        .scatter-container {{
            position: relative;
            width: 100%;
            height: 500px;
            background: #1a1a2e;
            border-radius: 6px;
            overflow: hidden;
        }}
        .scatter-point {{
            position: absolute;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            cursor: pointer;
            transition: transform 0.1s;
            border: 1px solid rgba(255,255,255,0.3);
        }}
        .scatter-point:hover {{
            transform: translate(-50%, -50%) scale(2);
            z-index: 100;
        }}
        .scatter-point.noise {{
            background: #666 !important;
            width: 6px;
            height: 6px;
        }}
        .stats {{
            display: flex;
            gap: 15px;
            margin: 10px 0;
            flex-wrap: wrap;
        }}
        .stat-box {{
            background: #21262d;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .stat-box.highlight {{
            background: #238636;
            color: white;
        }}
        .stat-box.split {{
            background: #9333ea;
            color: white;
        }}
        .clusters-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }}
        .cluster-box {{
            background: #21262d;
            border-radius: 6px;
            padding: 12px;
        }}
        .cluster-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .cluster-color {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
        }}
        .face-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            max-height: 150px;
            overflow-y: auto;
        }}
        .face-item {{
            width: 32px;
            height: 32px;
            border-radius: 3px;
            object-fit: cover;
        }}
        .tab-container {{
            margin: 20px 0;
        }}
        .tab-buttons {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .tab-btn {{
            padding: 10px 16px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .tab-btn:hover {{ background: #30363d; }}
        .tab-btn.active {{ background: #58a6ff; color: #0d1117; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }}
        .comparison-table th {{
            background: #21262d;
            color: #58a6ff;
        }}
        .comparison-table tr:hover {{
            background: #161b22;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HDBSCAN Variants with kNN Split</h1>
        <div class="meta">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Total faces: {n_faces}
        </div>

        <div class="algo-box">
            <h3>Algorithm: Post-Clustering Split</h3>
            <pre>
After HDBSCAN + merge/attach phases complete:

1. For each cluster >= split_min_cluster_size (default: 10):
   a. Build kNN graph (k = split_k neighbors per face)
   b. Compute cosine similarity between connected faces
   c. Prune edges where similarity < split_threshold
   d. Find connected components in pruned graph
   e. If multiple components exist -> split cluster

This prevents over-merging by ensuring all faces in a cluster
are connected by strong similarity paths.
            </pre>
        </div>

        <h2>Results Comparison</h2>
        <table class="comparison-table">
            <tr>
                <th>Method</th>
                <th>Split Threshold</th>
                <th>Clusters</th>
                <th>Splits</th>
                <th>Noise</th>
                <th>Max Size</th>
                <th>Top Cluster Sizes</th>
            </tr>
'''

    for r in results:
        sizes_str = str(r["cluster_sizes"][:6])
        if len(r["cluster_sizes"]) > 6:
            sizes_str = sizes_str[:-1] + ", ...]"
        split_label = f"{r['split_threshold']:.2f}" if r['split_enabled'] else "disabled"
        html += f'''
            <tr>
                <td>{r["method"]}</td>
                <td>{split_label}</td>
                <td>{r["n_clusters"]}</td>
                <td>{r["total_splits"]}</td>
                <td>{r["n_noise"]}</td>
                <td>{r["cluster_sizes"][0] if r["cluster_sizes"] else 0}</td>
                <td style="font-size:0.85em">{sizes_str}</td>
            </tr>
'''

    html += '''
        </table>

        <div class="tab-container">
            <div class="tab-buttons">
'''

    # Tab buttons
    for i, r in enumerate(results):
        active = "active" if i == 0 else ""
        split_label = f"t={r['split_threshold']:.2f}" if r['split_enabled'] else "no split"
        label = f"{r['method']} ({split_label})"
        html += f'<button class="tab-btn {active}" onclick="showTab({i})">{label}</button>\n'

    html += '''
            </div>
'''

    # Tab content for each result
    for ri, r in enumerate(results):
        labels = r["labels"]
        active = "active" if ri == 0 else ""
        split_label = f"{r['split_threshold']:.2f}" if r['split_enabled'] else "disabled"

        html += f'''
            <div id="tab-{ri}" class="tab-content {active}">
                <div class="card">
                    <h2>{r["method"]} (split_threshold={split_label})</h2>
                    <div class="stats">
                        <div class="stat-box highlight">{r["n_clusters"]} clusters</div>
                        <div class="stat-box split">{r["total_splits"]} splits</div>
                        <div class="stat-box">{r["n_noise"]} noise</div>
                        <div class="stat-box">Max: {r["cluster_sizes"][0] if r["cluster_sizes"] else 0}</div>
                    </div>

                    <h3>UMAP Projection</h3>
                    <div class="scatter-container">
'''
        # Scatter points
        for i in range(n_faces):
            x_pct = 3 + 94 * (umap_coords[i, 0] - x_min) / x_range
            y_pct = 3 + 94 * (umap_coords[i, 1] - y_min) / y_range
            label = labels[i]
            if label == -1:
                html += f'<div class="scatter-point noise" style="left:{x_pct}%;top:{y_pct}%" title="Face #{i} (noise)"></div>\n'
            else:
                color = colors[label % len(colors)]
                html += f'<div class="scatter-point" style="left:{x_pct}%;top:{y_pct}%;background:{color}" title="Face #{i}, Cluster {label}"></div>\n'

        html += '''
                    </div>

                    <h3>Clusters (largest first)</h3>
                    <div class="clusters-grid">
'''
        # Show clusters sorted by size
        unique_labels = sorted(set(labels) - {-1}, key=lambda l: -sum(labels == l))

        for cluster_id in unique_labels[:20]:  # Show top 20
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            color = colors[cluster_id % len(colors)]

            html += f'''
                        <div class="cluster-box">
                            <div class="cluster-header">
                                <div class="cluster-color" style="background:{color}"></div>
                                <strong>C{cluster_id}</strong> ({len(cluster_indices)} faces)
                            </div>
                            <div class="face-grid">
'''
            for idx in cluster_indices[:25]:  # Show up to 25 faces
                b64 = get_face_crop_base64(results_dir, idx)
                if b64:
                    html += f'<img class="face-item" src="data:image/jpeg;base64,{b64}" title="#{idx}">\n'
                else:
                    html += f'<div class="face-item" style="background:#333;display:flex;align-items:center;justify-content:center;font-size:7px">#{idx}</div>\n'

            if len(cluster_indices) > 25:
                html += f'<div style="padding:3px;color:#8b949e;font-size:10px">+{len(cluster_indices)-25}</div>'

            html += '''
                            </div>
                        </div>
'''

        if len(unique_labels) > 20:
            html += f'<div class="cluster-box" style="text-align:center;padding:20px;color:#8b949e;">+{len(unique_labels)-20} more clusters</div>'

        html += '''
                    </div>
                </div>
            </div>
'''

    html += '''
        </div>
    </div>

    <script>
        function showTab(idx) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + idx).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
'''

    output_path.write_text(html, encoding='utf-8')
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark HDBSCAN variants with split")
    parser.add_argument("--results_dir", type=Path, default=Path("results/face_clustering_benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thresholds", type=str, default="0.60,0.65,0.70,0.75",
                        help="Comma-separated split thresholds to test")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "hdbscan_split_benchmark.html"

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    # Load data
    print("Loading data...")
    embeddings, metadata = load_data(args.results_dir)
    print(f"Loaded {len(embeddings)} embeddings")

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    # Run UMAP once
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(normalized)

    # Run clustering with different configurations
    results = []

    # Baseline: no split
    print("\nRunning hybrid_hdbscan_knn (no split)...")
    results.append(run_clustering(
        embeddings, HybridHDBSCANKNN, "hybrid_hdbscan_knn", 0.0, split_enabled=False
    ))

    # Test different split thresholds
    for threshold in thresholds:
        print(f"\nRunning hybrid_hdbscan_knn (split_threshold={threshold})...")
        results.append(run_clustering(
            embeddings, HybridHDBSCANKNN, "hybrid_hdbscan_knn", threshold, split_enabled=True
        ))

    # Also test hybrid_closest_face
    print(f"\nRunning hybrid_closest_face (split_threshold=0.70)...")
    results.append(run_clustering(
        embeddings, HybridHDBSCANClosestFace, "hybrid_closest_face", 0.70, split_enabled=True
    ))

    # Generate report
    print("\nGenerating HTML report...")
    generate_html_report(args.results_dir, embeddings, umap_coords, results, args.output)


if __name__ == "__main__":
    main()
