"""UMAP visualization of all faces with K-Means clustering (k=8,9,10).

Generates HTML report showing UMAP projections and clustering results.
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


def generate_html_report(
    results_dir: Path,
    embeddings: np.ndarray,
    umap_coords: np.ndarray,
    k_values: List[int],
    output_path: Path
) -> None:
    """Generate HTML report with UMAP and K-Means results."""

    n_faces = len(embeddings)

    # Normalize embeddings for clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    # Run K-Means for each k
    kmeans_results = {}
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized)
        cluster_sizes = [sum(labels == i) for i in range(k)]
        kmeans_results[k] = {
            "labels": labels,
            "sizes": sorted(cluster_sizes, reverse=True)
        }

    # Normalize UMAP coords for display
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    # Color palette for clusters
    colors = [
        "#4CAF50", "#2196F3", "#f44336", "#FF9800", "#9C27B0",
        "#00BCD4", "#FFEB3B", "#E91E63", "#8BC34A", "#3F51B5",
        "#FF5722", "#607D8B", "#795548", "#009688", "#CDDC39"
    ]

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>All Faces UMAP Analysis</title>
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
            height: 600px;
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
        .legend {{
            display: flex;
            gap: 10px;
            margin: 10px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.85em;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .clusters-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .cluster-box {{
            background: #21262d;
            border-radius: 6px;
            padding: 15px;
        }}
        .cluster-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .cluster-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .face-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .face-item {{
            width: 40px;
            height: 40px;
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
        }}
        .tab-btn {{
            padding: 10px 20px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            cursor: pointer;
            font-size: 1em;
        }}
        .tab-btn:hover {{ background: #30363d; }}
        .tab-btn.active {{ background: #58a6ff; color: #0d1117; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>All Faces UMAP Analysis</h1>
        <div class="meta">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Total faces: {n_faces}
        </div>

        <div class="tab-container">
            <div class="tab-buttons">
'''

    # Tab buttons
    for i, k in enumerate(k_values):
        active = "active" if i == 0 else ""
        html += f'<button class="tab-btn {active}" onclick="showTab({k})">K = {k}</button>\n'

    html += '''
            </div>
'''

    # Tab content for each k
    for ki, k in enumerate(k_values):
        labels = kmeans_results[k]["labels"]
        sizes = kmeans_results[k]["sizes"]
        active = "active" if ki == 0 else ""

        html += f'''
            <div id="tab-{k}" class="tab-content {active}">
                <div class="card">
                    <h2>K-Means with K = {k}</h2>
                    <div class="stats">
                        <div class="stat-box">Cluster sizes: {sizes}</div>
                    </div>

                    <h3>UMAP Projection</h3>
                    <div class="legend">
'''
        # Legend
        for cluster_id in range(k):
            count = sum(labels == cluster_id)
            color = colors[cluster_id % len(colors)]
            html += f'<div class="legend-item"><div class="legend-dot" style="background:{color}"></div> C{cluster_id} ({count})</div>\n'

        html += f'''
                    </div>
                    <div class="scatter-container">
'''
        # Scatter points
        for i in range(n_faces):
            x_pct = 3 + 94 * (umap_coords[i, 0] - x_min) / x_range
            y_pct = 3 + 94 * (umap_coords[i, 1] - y_min) / y_range
            color = colors[labels[i] % len(colors)]
            html += f'<div class="scatter-point" style="left:{x_pct}%;top:{y_pct}%;background:{color}" title="Face #{i}, Cluster {labels[i]}"></div>\n'

        html += '''
                    </div>

                    <h3>Faces by Cluster</h3>
                    <div class="clusters-grid">
'''
        # Cluster boxes with faces
        for cluster_id in range(k):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            color = colors[cluster_id % len(colors)]

            html += f'''
                        <div class="cluster-box">
                            <div class="cluster-header">
                                <div class="cluster-color" style="background:{color}"></div>
                                <strong>Cluster {cluster_id}</strong> ({len(cluster_indices)} faces)
                            </div>
                            <div class="face-grid">
'''
            for idx in cluster_indices[:40]:  # Show up to 40 faces
                b64 = get_face_crop_base64(results_dir, idx)
                if b64:
                    html += f'<img class="face-item" src="data:image/jpeg;base64,{b64}" title="#{idx}">\n'
                else:
                    html += f'<div class="face-item" style="background:#333;display:flex;align-items:center;justify-content:center;font-size:8px">#{idx}</div>\n'

            if len(cluster_indices) > 40:
                html += f'<div style="padding:5px;color:#8b949e;font-size:11px">+{len(cluster_indices)-40} more</div>'

            html += '''
                            </div>
                        </div>
'''

        html += '''
                    </div>
                </div>
            </div>
'''

    html += '''
        </div>
    </div>

    <script>
        function showTab(k) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

            // Show selected tab
            document.getElementById('tab-' + k).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
'''

    output_path.write_text(html, encoding='utf-8')
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP analysis of all faces")
    parser.add_argument("--results_dir", type=Path, default=Path("results/face_clustering_benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k_values", type=str, default="8,9,10",
                        help="Comma-separated K values for K-Means")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "all_faces_umap.html"

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Load data
    print("Loading data...")
    embeddings, metadata = load_data(args.results_dir)
    print(f"Loaded {len(embeddings)} embeddings")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    # Run UMAP
    print("Running UMAP on all faces...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(normalized)
    print("UMAP complete")

    # Generate report
    print(f"Generating HTML report with K = {k_values}...")
    generate_html_report(args.results_dir, embeddings, umap_coords, k_values, args.output)


if __name__ == "__main__":
    main()
