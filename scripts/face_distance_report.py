"""Generate HTML diagnostic report for face embedding distances between two groups.

Usage:
    python scripts/face_distance_report.py --results_dir results/face_clustering_benchmark \
        --group1 0,81,79,82,91,135 --group2 2,46,183,249 \
        --label1 "Person 1" --label2 "Person 2" \
        --output report.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import base64

import numpy as np
from scipy.spatial.distance import cosine


def load_embeddings(results_dir: Path) -> np.ndarray:
    """Load embeddings from benchmark results."""
    npy_files = sorted(results_dir.glob("embeddings_*.npy"), reverse=True)
    if not npy_files:
        raise FileNotFoundError(f"No embeddings NPY found in {results_dir}")
    return np.load(npy_files[0])


def get_face_crop_base64(results_dir: Path, face_idx: int) -> str:
    """Get face crop as base64 string for embedding in HTML."""
    crops_dir = results_dir / "face_crops"

    # Try aligned version first
    crop_path = crops_dir / f"face_{face_idx:04d}_aligned.jpg"
    if not crop_path.exists():
        crop_path = crops_dir / f"face_{face_idx:04d}.jpg"

    if crop_path.exists():
        with open(crop_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""


def compute_distances(embeddings: np.ndarray, indices: List[int]) -> List[Tuple[int, int, float]]:
    """Compute pairwise distances within a group."""
    distances = []
    for i, idx1 in enumerate(indices):
        for idx2 in indices[i+1:]:
            if idx1 < len(embeddings) and idx2 < len(embeddings):
                d = cosine(embeddings[idx1], embeddings[idx2])
                distances.append((idx1, idx2, d))
    return distances


def compute_cross_distances(embeddings: np.ndarray, group1: List[int], group2: List[int]) -> List[Tuple[int, int, float]]:
    """Compute distances between two groups."""
    distances = []
    for idx1 in group1:
        for idx2 in group2:
            if idx1 < len(embeddings) and idx2 < len(embeddings):
                d = cosine(embeddings[idx1], embeddings[idx2])
                distances.append((idx1, idx2, d))
    return distances


def generate_histogram_svg(intra1: List[float], intra2: List[float], inter: List[float], width: int = 600, height: int = 200) -> str:
    """Generate SVG histogram of distance distributions."""
    all_dists = intra1 + intra2 + inter
    if not all_dists:
        return ""

    min_d, max_d = min(all_dists), max(all_dists)
    n_bins = 20
    bin_width = (max_d - min_d) / n_bins if max_d > min_d else 0.1

    def make_histogram(dists, color):
        if not dists or bin_width == 0:
            return []
        bins = [0] * n_bins
        for d in dists:
            idx = min(int((d - min_d) / bin_width), n_bins - 1)
            bins[idx] += 1
        return bins, color

    hist1 = make_histogram(intra1, "#4CAF50")  # Green
    hist2 = make_histogram(intra2, "#2196F3")  # Blue
    hist3 = make_histogram(inter, "#f44336")   # Red

    max_count = max(max(h[0]) for h in [hist1, hist2, hist3] if h) if any([hist1, hist2, hist3]) else 1

    bar_width = (width - 80) / n_bins
    chart_height = height - 50

    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>'

    # Draw bars
    for hist_data, opacity in [(hist1, 0.7), (hist2, 0.7), (hist3, 0.7)]:
        if hist_data:
            bins, color = hist_data
            for i, count in enumerate(bins):
                if count > 0:
                    bar_height = (count / max_count) * chart_height
                    x = 50 + i * bar_width
                    y = chart_height - bar_height + 10
                    svg += f'<rect x="{x}" y="{y}" width="{bar_width-1}" height="{bar_height}" fill="{color}" opacity="{opacity}"/>'

    # X axis labels
    for i in range(0, n_bins + 1, 5):
        x = 50 + i * bar_width
        val = min_d + i * bin_width
        svg += f'<text x="{x}" y="{height-5}" fill="#888" font-size="10" text-anchor="middle">{val:.2f}</text>'

    # Legend
    svg += '<text x="60" y="25" fill="#4CAF50" font-size="11">Group 1 Intra</text>'
    svg += '<text x="160" y="25" fill="#2196F3" font-size="11">Group 2 Intra</text>'
    svg += '<text x="260" y="25" fill="#f44336" font-size="11">Inter-group</text>'

    svg += '</svg>'
    return svg


def generate_html_report(
    results_dir: Path,
    group1: List[int],
    group2: List[int],
    label1: str,
    label2: str,
    embeddings: np.ndarray
) -> str:
    """Generate complete HTML report."""

    # Compute distances
    intra1 = compute_distances(embeddings, group1)
    intra2 = compute_distances(embeddings, group2)
    inter = compute_cross_distances(embeddings, group1, group2)

    intra1_dists = [d[2] for d in intra1]
    intra2_dists = [d[2] for d in intra2]
    inter_dists = [d[2] for d in inter]

    # Statistics
    def stats(dists):
        if not dists:
            return {"min": 0, "max": 0, "median": 0, "mean": 0}
        return {
            "min": min(dists),
            "max": max(dists),
            "median": float(np.median(dists)),
            "mean": float(np.mean(dists))
        }

    stats1 = stats(intra1_dists)
    stats2 = stats(intra2_dists)
    stats_inter = stats(inter_dists)

    # Analysis
    max_intra = max(stats1["max"], stats2["max"])
    min_inter = stats_inter["min"]
    has_overlap = max_intra > min_inter
    overlap_amount = max_intra - min_inter if has_overlap else 0
    separation_gap = min_inter - max_intra if not has_overlap else 0
    gap_display = f"{overlap_amount:.3f}" if has_overlap else f"{separation_gap:.3f}"
    gap_class = "bad" if has_overlap else "good"
    gap_label = "Overlap" if has_overlap else "Separation Gap"

    # Find problematic faces (high intra-group distances)
    threshold = stats_inter["median"]  # Faces with intra-distance > inter-median are problematic
    problematic = []
    for idx1, idx2, d in intra1 + intra2:
        if d > threshold:
            problematic.append((idx1, idx2, d))
    problematic.sort(key=lambda x: -x[2])

    # Generate histogram
    histogram_svg = generate_histogram_svg(intra1_dists, intra2_dists, inter_dists)

    # Build HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Face Distance Diagnostic Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; margin-bottom: 10px; }}
        h2 {{ color: #8b949e; margin: 30px 0 15px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
        h3 {{ color: #c9d1d9; margin: 20px 0 10px; }}
        .meta {{ color: #8b949e; font-size: 0.9em; margin-bottom: 30px; }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: #21262d;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; }}
        .stat-label {{ color: #8b949e; font-size: 0.85em; }}
        .good {{ color: #3fb950; }}
        .warning {{ color: #d29922; }}
        .bad {{ color: #f85149; }}
        .face-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        .face-item {{
            text-align: center;
            background: #21262d;
            padding: 10px;
            border-radius: 6px;
        }}
        .face-item img {{
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 4px;
        }}
        .face-label {{ font-size: 0.8em; color: #8b949e; margin-top: 5px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }}
        th {{ background: #21262d; color: #8b949e; }}
        .distance {{ font-family: monospace; }}
        .high {{ background: rgba(248, 81, 73, 0.2); }}
        .verdict {{
            font-size: 1.2em;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .verdict.overlap {{ background: rgba(248, 81, 73, 0.15); border-left: 4px solid #f85149; }}
        .verdict.separated {{ background: rgba(63, 185, 80, 0.15); border-left: 4px solid #3fb950; }}
        .histogram {{ margin: 20px 0; text-align: center; }}
        .recommendation {{
            background: #1f2937;
            border-left: 4px solid #58a6ff;
            padding: 15px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Distance Diagnostic Report</h1>
        <div class="meta">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Embeddings: {len(embeddings)} faces, {embeddings.shape[1]} dimensions
        </div>

        <h2>Groups Under Analysis</h2>
        <div class="card">
            <h3>{label1} ({len(group1)} faces)</h3>
            <div class="face-grid">
'''

    # Add face images for group 1
    for idx in group1:
        img_b64 = get_face_crop_base64(results_dir, idx)
        if img_b64:
            html += f'''
                <div class="face-item">
                    <img src="data:image/jpeg;base64,{img_b64}" alt="Face {idx}">
                    <div class="face-label">#{idx}</div>
                </div>'''
        else:
            html += f'''
                <div class="face-item">
                    <div style="width:80px;height:80px;background:#30363d;border-radius:4px;display:flex;align-items:center;justify-content:center;">#{idx}</div>
                    <div class="face-label">#{idx}</div>
                </div>'''

    html += f'''
            </div>

            <h3>{label2} ({len(group2)} faces)</h3>
            <div class="face-grid">
'''

    # Add face images for group 2
    for idx in group2:
        img_b64 = get_face_crop_base64(results_dir, idx)
        if img_b64:
            html += f'''
                <div class="face-item">
                    <img src="data:image/jpeg;base64,{img_b64}" alt="Face {idx}">
                    <div class="face-label">#{idx}</div>
                </div>'''
        else:
            html += f'''
                <div class="face-item">
                    <div style="width:80px;height:80px;background:#30363d;border-radius:4px;display:flex;align-items:center;justify-content:center;">#{idx}</div>
                    <div class="face-label">#{idx}</div>
                </div>'''

    html += '''
            </div>
        </div>

        <h2>Distance Distribution</h2>
        <div class="card">
            <div class="histogram">
'''
    html += histogram_svg
    html += f'''
            </div>
            <p style="text-align:center;color:#8b949e;margin-top:10px;">
                Cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)
            </p>
        </div>

        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value" style="color:#4CAF50">{stats1['median']:.3f}</div>
                <div class="stat-label">{label1} Median Distance</div>
                <div class="stat-label">Range: {stats1['min']:.3f} - {stats1['max']:.3f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:#2196F3">{stats2['median']:.3f}</div>
                <div class="stat-label">{label2} Median Distance</div>
                <div class="stat-label">Range: {stats2['min']:.3f} - {stats2['max']:.3f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:#f44336">{stats_inter['median']:.3f}</div>
                <div class="stat-label">Inter-Group Median</div>
                <div class="stat-label">Range: {stats_inter['min']:.3f} - {stats_inter['max']:.3f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-value {gap_class}">{gap_display}</div>
                <div class="stat-label">{gap_label}</div>
            </div>
        </div>

        <h2>Diagnosis</h2>
'''

    if has_overlap:
        html += f'''
        <div class="verdict overlap">
            <strong>OVERLAP DETECTED</strong><br>
            Max intra-group distance ({max_intra:.3f}) exceeds min inter-group distance ({min_inter:.3f}).<br>
            Overlap amount: <strong>{overlap_amount:.3f}</strong><br><br>
            <em>This means some same-person pairs are MORE distant than some different-person pairs.
            No clustering threshold can perfectly separate these groups.</em>
        </div>

        <div class="recommendation">
            <strong>Recommendations:</strong>
            <ul style="margin-top:10px;padding-left:20px;">
                <li>Check alignment quality for high-distance same-person pairs</li>
                <li>Look for extreme pose variations (profile views, head tilts)</li>
                <li>Consider excluding problematic faces from clustering</li>
                <li>Embedding model may struggle with this subject's appearance variation</li>
            </ul>
        </div>
'''
    else:
        html += f'''
        <div class="verdict separated">
            <strong>GOOD SEPARATION</strong><br>
            Max intra-group distance ({max_intra:.3f}) is below min inter-group distance ({min_inter:.3f}).<br>
            Separation gap: <strong>{separation_gap:.3f}</strong><br><br>
            <em>A threshold between {max_intra:.3f} and {min_inter:.3f} should cleanly separate these groups.</em>
        </div>
'''

    # Problematic pairs
    if problematic:
        html += '''
        <h2>Problematic Pairs</h2>
        <div class="card">
            <p>Same-person pairs with unusually high distances (> inter-group median):</p>
            <table>
                <tr><th>Face A</th><th>Face B</th><th>Distance</th><th>Status</th></tr>
'''
        for idx1, idx2, d in problematic[:10]:
            html += f'''
                <tr class="high">
                    <td>#{idx1}</td>
                    <td>#{idx2}</td>
                    <td class="distance">{d:.4f}</td>
                    <td class="bad">Investigate</td>
                </tr>
'''
        html += '''
            </table>
        </div>
'''

    # Detailed distance matrix
    html += f'''
        <h2>Detailed Distances</h2>
        <div class="card">
            <h3>{label1} Intra-Group Distances</h3>
            <table>
                <tr><th>Face A</th><th>Face B</th><th>Distance</th></tr>
'''
    for idx1, idx2, d in sorted(intra1, key=lambda x: -x[2]):
        row_class = 'high' if d > threshold else ''
        html += f'<tr class="{row_class}"><td>#{idx1}</td><td>#{idx2}</td><td class="distance">{d:.4f}</td></tr>'

    html += f'''
            </table>

            <h3>{label2} Intra-Group Distances</h3>
            <table>
                <tr><th>Face A</th><th>Face B</th><th>Distance</th></tr>
'''
    for idx1, idx2, d in sorted(intra2, key=lambda x: -x[2]):
        row_class = 'high' if d > threshold else ''
        html += f'<tr class="{row_class}"><td>#{idx1}</td><td>#{idx2}</td><td class="distance">{d:.4f}</td></tr>'

    html += '''
            </table>
        </div>
    </div>
</body>
</html>
'''

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate face distance diagnostic report")
    parser.add_argument("--results_dir", type=Path, default=Path("results/face_clustering_benchmark"))
    parser.add_argument("--group1", type=str, required=True, help="Comma-separated face indices for group 1")
    parser.add_argument("--group2", type=str, required=True, help="Comma-separated face indices for group 2")
    parser.add_argument("--label1", type=str, default="Group 1", help="Label for group 1")
    parser.add_argument("--label2", type=str, default="Group 2", help="Label for group 2")
    parser.add_argument("--output", type=Path, default=Path("distance_report.html"))
    args = parser.parse_args()

    # Parse groups
    group1 = [int(x.strip()) for x in args.group1.split(",")]
    group2 = [int(x.strip()) for x in args.group2.split(",")]

    # Load embeddings
    embeddings = load_embeddings(args.results_dir)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Generate report
    html = generate_html_report(
        args.results_dir, group1, group2, args.label1, args.label2, embeddings
    )

    # Write output
    args.output.write_text(html, encoding="utf-8")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
