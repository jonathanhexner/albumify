"""kNN Graph + Connected Components clustering.

Approach:
1. Build kNN graph (k neighbors per face)
2. Prune edges below cosine similarity threshold
3. Find connected components = clusters

No density estimation, no centroids, no hierarchy.
Just strong similarity links forming natural groups.
"""

import argparse
import base64
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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


def build_knn_graph(
    embeddings: np.ndarray,
    k: int
) -> Dict[int, List[Tuple[int, float]]]:
    """Build kNN graph based on cosine similarity.

    Returns:
        Dict mapping each face index to list of (neighbor_idx, similarity) tuples
    """
    # L2-normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    # Compute full similarity matrix
    sim_matrix = cosine_similarity(normalized)

    n_faces = len(embeddings)
    graph = {}

    for i in range(n_faces):
        # Get similarities to all other faces (exclude self)
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self

        # Get top-k neighbors
        top_k_indices = np.argsort(sims)[-k:][::-1]
        neighbors = [(int(j), float(sims[j])) for j in top_k_indices]
        graph[i] = neighbors

    return graph


def prune_graph(
    graph: Dict[int, List[Tuple[int, float]]],
    threshold: float
) -> Dict[int, Set[int]]:
    """Prune edges below threshold, return undirected adjacency.

    Returns:
        Dict mapping each face to set of strongly connected neighbors
    """
    adjacency = defaultdict(set)

    for i, neighbors in graph.items():
        for j, sim in neighbors:
            if sim >= threshold:
                # Add undirected edge
                adjacency[i].add(j)
                adjacency[j].add(i)

    return dict(adjacency)


def find_connected_components(adjacency: Dict[int, Set[int]], n_faces: int) -> np.ndarray:
    """Find connected components using BFS.

    Returns:
        Array of cluster labels (-1 for singletons/noise)
    """
    labels = np.full(n_faces, -1, dtype=int)
    visited = set()
    current_label = 0

    for start in range(n_faces):
        if start in visited:
            continue

        # BFS to find component
        component = []
        queue = [start]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)

            # Add unvisited neighbors
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Assign label (singletons get -1)
        if len(component) >= 2:
            for node in component:
                labels[node] = current_label
            current_label += 1

    return labels


def compute_stats(labels: np.ndarray) -> dict:
    """Compute clustering statistics."""
    unique_labels = set(labels) - {-1}
    cluster_sizes = [sum(labels == c) for c in unique_labels]

    return {
        "n_clusters": len(unique_labels),
        "n_noise": sum(labels == -1),
        "cluster_sizes": sorted(cluster_sizes, reverse=True),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
    }


def generate_html_report(
    results_dir: Path,
    embeddings: np.ndarray,
    umap_coords: np.ndarray,
    results: List[dict],
    output_path: Path
) -> None:
    """Generate HTML report with results for each threshold."""

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
    <title>kNN Graph + Connected Components Clustering</title>
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
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
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
            max-height: 180px;
            overflow-y: auto;
        }}
        .face-item {{
            width: 38px;
            height: 38px;
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
        <h1>kNN Graph + Connected Components Clustering</h1>
        <div class="meta">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Total faces: {n_faces}
        </div>

        <div class="algo-box">
            <h3>Algorithm</h3>
            <pre>
Step 1: Build kNN Graph (k neighbors per face)
    - For each face, find k nearest neighbors by cosine similarity
    - Creates local connectivity (no long weak chains)

Step 2: Prune Weak Edges
    - Keep only edges where cosine_similarity >= threshold
    - Creates high-precision core graph

Step 3: Connected Components
    - Find connected components in undirected graph
    - Each component = one identity cluster
    - Isolated nodes (no strong edges) = noise (-1)
            </pre>
        </div>

        <h2>Results Comparison</h2>
        <table class="comparison-table">
            <tr>
                <th>k</th>
                <th>Threshold</th>
                <th>Clusters</th>
                <th>Noise</th>
                <th>Max Size</th>
                <th>Cluster Sizes</th>
            </tr>
'''

    for r in results:
        sizes_str = str(r["stats"]["cluster_sizes"][:8])
        if len(r["stats"]["cluster_sizes"]) > 8:
            sizes_str = sizes_str[:-1] + ", ...]"
        html += f'''
            <tr>
                <td>{r["k"]}</td>
                <td>{r["threshold"]:.2f}</td>
                <td>{r["stats"]["n_clusters"]}</td>
                <td>{r["stats"]["n_noise"]}</td>
                <td>{r["stats"]["max_cluster_size"]}</td>
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
        label = f"k={r['k']}, t={r['threshold']:.2f}"
        html += f'<button class="tab-btn {active}" onclick="showTab({i})">{label}</button>\n'

    html += '''
            </div>
'''

    # Tab content for each configuration
    for ri, r in enumerate(results):
        labels = r["labels"]
        stats = r["stats"]
        active = "active" if ri == 0 else ""

        html += f'''
            <div id="tab-{ri}" class="tab-content {active}">
                <div class="card">
                    <h2>k={r["k"]}, threshold={r["threshold"]:.2f}</h2>
                    <div class="stats">
                        <div class="stat-box highlight">{stats["n_clusters"]} clusters</div>
                        <div class="stat-box">{stats["n_noise"]} noise faces</div>
                        <div class="stat-box">Max cluster: {stats["max_cluster_size"]}</div>
                        <div class="stat-box">Avg cluster: {stats["avg_cluster_size"]:.1f}</div>
                    </div>

                    <h3>UMAP Projection</h3>
                    <div class="legend">
'''
        # Legend
        unique_labels = sorted(set(labels) - {-1})
        for cluster_id in unique_labels[:15]:  # Show first 15
            count = sum(labels == cluster_id)
            color = colors[cluster_id % len(colors)]
            html += f'<div class="legend-item"><div class="legend-dot" style="background:{color}"></div> C{cluster_id} ({count})</div>\n'

        if len(unique_labels) > 15:
            html += f'<div class="legend-item">... +{len(unique_labels) - 15} more</div>\n'

        if stats["n_noise"] > 0:
            html += f'<div class="legend-item"><div class="legend-dot" style="background:#666"></div> Noise ({stats["n_noise"]})</div>\n'

        html += f'''
                    </div>
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

                    <h3>Faces by Cluster</h3>
                    <div class="clusters-grid">
'''
        # Cluster boxes with faces
        for cluster_id in unique_labels:
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
            for idx in cluster_indices[:35]:  # Show up to 35 faces
                b64 = get_face_crop_base64(results_dir, idx)
                if b64:
                    html += f'<img class="face-item" src="data:image/jpeg;base64,{b64}" title="#{idx}">\n'
                else:
                    html += f'<div class="face-item" style="background:#333;display:flex;align-items:center;justify-content:center;font-size:8px">#{idx}</div>\n'

            if len(cluster_indices) > 35:
                html += f'<div style="padding:5px;color:#8b949e;font-size:11px">+{len(cluster_indices)-35} more</div>'

            html += '''
                            </div>
                        </div>
'''

        # Show noise faces if any
        if stats["n_noise"] > 0:
            noise_indices = [i for i, l in enumerate(labels) if l == -1]
            html += f'''
                        <div class="cluster-box" style="border: 1px dashed #666;">
                            <div class="cluster-header">
                                <div class="cluster-color" style="background:#666"></div>
                                <strong>Noise</strong> ({len(noise_indices)} faces)
                            </div>
                            <div class="face-grid">
'''
            for idx in noise_indices[:35]:
                b64 = get_face_crop_base64(results_dir, idx)
                if b64:
                    html += f'<img class="face-item" src="data:image/jpeg;base64,{b64}" title="#{idx}">\n'
                else:
                    html += f'<div class="face-item" style="background:#333;display:flex;align-items:center;justify-content:center;font-size:8px">#{idx}</div>\n'

            if len(noise_indices) > 35:
                html += f'<div style="padding:5px;color:#8b949e;font-size:11px">+{len(noise_indices)-35} more</div>'

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
        function showTab(idx) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

            // Show selected tab
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
    parser = argparse.ArgumentParser(description="kNN Graph + Connected Components clustering")
    parser.add_argument("--results_dir", type=Path, default=Path("results/face_clustering_benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k_values", type=str, default="20,30,50",
                        help="Comma-separated k values for kNN graph")
    parser.add_argument("--thresholds", type=str, default="0.65,0.70,0.75",
                        help="Comma-separated cosine similarity thresholds")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "knn_components_clustering.html"

    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    # Load data
    print("Loading data...")
    embeddings, metadata = load_data(args.results_dir)
    print(f"Loaded {len(embeddings)} embeddings")

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)

    # Run UMAP once for visualization
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(normalized)

    # Run clustering for each configuration
    results = []

    for k in k_values:
        print(f"\nBuilding kNN graph (k={k})...")
        graph = build_knn_graph(embeddings, k)

        for threshold in thresholds:
            print(f"  Pruning with threshold={threshold:.2f}...")
            adjacency = prune_graph(graph, threshold)

            print(f"  Finding connected components...")
            labels = find_connected_components(adjacency, len(embeddings))
            stats = compute_stats(labels)

            print(f"    -> {stats['n_clusters']} clusters, {stats['n_noise']} noise, "
                  f"sizes={stats['cluster_sizes'][:5]}{'...' if len(stats['cluster_sizes']) > 5 else ''}")

            results.append({
                "k": k,
                "threshold": threshold,
                "labels": labels,
                "stats": stats
            })

    # Generate report
    print("\nGenerating HTML report...")
    generate_html_report(args.results_dir, embeddings, umap_coords, results, args.output)


if __name__ == "__main__":
    main()
