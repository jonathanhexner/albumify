"""
Create HTML report showing face images and distance matrix.

Visualizes:
- All face crops with labels
- Distance matrix between all pairs
- Highlights same-person vs different-person pairs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import base64
import pandas as pd
import numpy as np
from PIL import Image

# Paths
GT_DIR = Path("test_data/ground_truth_test")
CROPS_DIR = GT_DIR / "face_crops"
LABELS_FILE = CROPS_DIR / "labels_converted.csv"
METADATA_FILE = GT_DIR / "metadata.json"
OUTPUT_FILE = GT_DIR / "distance_report.html"


def cosine_distance(emb1, emb2):
    """Compute cosine distance."""
    return 1.0 - np.dot(emb1, emb2)


def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def create_html_report():
    """Generate HTML report with faces and distance matrix."""

    # Load labels
    labels_df = pd.read_csv(LABELS_FILE)
    labels_df = labels_df[labels_df['person_id'] > 0]  # Exclude invalid

    # Load metadata with embeddings
    metadata = json.load(open(METADATA_FILE))

    # Build embedding lookup
    embeddings = {}
    for det in metadata['detections']:
        crop_file = det['crop_filename']
        if crop_file in labels_df['crop_filename'].values:
            embeddings[crop_file] = np.array(det['embedding'], dtype=np.float32)

    # Compute distance matrix
    crop_files = sorted(embeddings.keys())
    n = len(crop_files)

    dist_matrix = np.zeros((n, n))
    for i, crop_a in enumerate(crop_files):
        for j, crop_b in enumerate(crop_files):
            if i != j:
                dist_matrix[i, j] = cosine_distance(embeddings[crop_a], embeddings[crop_b])

    # Get person IDs for each crop
    person_ids = [
        labels_df[labels_df['crop_filename'] == crop]['person_id'].iloc[0]
        for crop in crop_files
    ]
    person_names = [
        labels_df[labels_df['crop_filename'] == crop]['person_name'].iloc[0]
        for crop in crop_files
    ]

    # Generate HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Face Distance Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }
            h1 {
                color: #333;
            }
            .section {
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .face-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .face-card {
                text-align: center;
                border: 2px solid #ddd;
                padding: 10px;
                border-radius: 8px;
                background: #fafafa;
            }
            .face-card img {
                width: 112px;
                height: 112px;
                border: 1px solid #ccc;
            }
            .face-card .label {
                margin-top: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            .face-card .person {
                font-size: 11px;
                color: #666;
            }
            .person-1 { border-color: #e74c3c; background: #ffebee; }
            .person-2 { border-color: #3498db; background: #e3f2fd; }
            .person-3 { border-color: #2ecc71; background: #e8f5e9; }
            .person-4 { border-color: #f39c12; background: #fff8e1; }

            table.distance-matrix {
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 11px;
            }
            table.distance-matrix th,
            table.distance-matrix td {
                padding: 4px;
                border: 1px solid #ddd;
                text-align: center;
                min-width: 50px;
            }
            table.distance-matrix th {
                background: #f0f0f0;
                font-weight: bold;
                font-size: 10px;
            }
            table.distance-matrix td {
                cursor: pointer;
            }
            .dist-very-low { background: #c8e6c9; }
            .dist-low { background: #fff9c4; }
            .dist-medium { background: #ffcc80; }
            .dist-high { background: #ef9a9a; }
            .dist-same-person { font-weight: bold; border: 2px solid #333; }

            .legend {
                display: flex;
                gap: 20px;
                margin-top: 10px;
                flex-wrap: wrap;
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .legend-box {
                width: 30px;
                height: 20px;
                border: 1px solid #999;
            }

            .stats {
                margin-top: 15px;
                padding: 10px;
                background: #f9f9f9;
                border-left: 4px solid #3498db;
            }
            .stats h3 {
                margin-top: 0;
            }
        </style>
    </head>
    <body>
        <h1>Face Distance Report</h1>

        <div class="section">
            <h2>Face Gallery (n=""" + str(n) + """)</h2>
            <div class="face-grid">
    """

    # Add face images
    for i, crop_file in enumerate(crop_files):
        person_id = person_ids[i]
        person_name = person_names[i]
        img_path = CROPS_DIR / crop_file
        img_b64 = image_to_base64(img_path)

        html += f"""
                <div class="face-card person-{person_id}">
                    <img src="data:image/jpeg;base64,{img_b64}" alt="{crop_file}">
                    <div class="label">{crop_file}</div>
                    <div class="person">Person {person_id} ({person_name})</div>
                </div>
        """

    html += """
            </div>
        </div>

        <div class="section">
            <h2>Distance Matrix</h2>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box dist-very-low"></div>
                    <span>0.00-0.20 (very similar)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box dist-low"></div>
                    <span>0.20-0.40 (similar)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box dist-medium"></div>
                    <span>0.40-0.60 (different)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box dist-high"></div>
                    <span>0.60+ (very different)</span>
                </div>
                <div class="legend-item">
                    <strong>Bold border = same person</strong>
                </div>
            </div>

            <div style="overflow-x: auto;">
                <table class="distance-matrix">
                    <thead>
                        <tr>
                            <th></th>
    """

    # Column headers
    for i, crop_file in enumerate(crop_files):
        short_name = crop_file.replace('_face_', '_').replace('.jpg', '').replace('20240816_', '16_').replace('20240817_', '17_').replace('20240818_', '18_')
        html += f'<th>{short_name}</th>\n'

    html += """
                        </tr>
                    </thead>
                    <tbody>
    """

    # Matrix rows
    for i, crop_a in enumerate(crop_files):
        short_name_a = crop_a.replace('_face_', '_').replace('.jpg', '').replace('20240816_', '16_').replace('20240817_', '17_').replace('20240818_', '18_')
        html += f'<tr><th>{short_name_a}</th>\n'

        for j, crop_b in enumerate(crop_files):
            dist = dist_matrix[i, j]

            # Determine color class
            if dist < 0.01:
                color_class = 'dist-very-low'
            elif dist < 0.20:
                color_class = 'dist-very-low'
            elif dist < 0.40:
                color_class = 'dist-low'
            elif dist < 0.60:
                color_class = 'dist-medium'
            else:
                color_class = 'dist-high'

            # Check if same person
            same_person = person_ids[i] == person_ids[j] and i != j
            same_class = ' dist-same-person' if same_person else ''

            html += f'<td class="{color_class}{same_class}" title="{crop_a} vs {crop_b}: {dist:.3f}">{dist:.3f}</td>\n'

        html += '</tr>\n'

    html += """
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>Statistics</h2>
    """

    # Compute statistics
    for person_id in sorted(set(person_ids)):
        person_name = person_names[person_ids.index(person_id)]
        indices = [i for i, pid in enumerate(person_ids) if pid == person_id]

        if len(indices) < 2:
            continue

        # Same-person distances
        same_person_dists = []
        for i in indices:
            for j in indices:
                if i < j:
                    same_person_dists.append(dist_matrix[i, j])

        min_dist = min(same_person_dists) if same_person_dists else 0
        max_dist = max(same_person_dists) if same_person_dists else 0
        avg_dist = np.mean(same_person_dists) if same_person_dists else 0

        html += f"""
            <div class="stats">
                <h3>Person {person_id} ({person_name}) - {len(indices)} faces</h3>
                <p>
                    Same-person distances: min={min_dist:.3f}, max={max_dist:.3f}, avg={avg_dist:.3f}
                    {'<span style="color: red;"> ⚠️ High max distance (>0.40)</span>' if max_dist > 0.40 else ' ✓'}
                </p>
            </div>
        """

    html += """
        </div>

    </body>
    </html>
    """

    # Save HTML
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved to: {OUTPUT_FILE}")
    print(f"Total faces: {n}")
    print(f"Distance matrix: {n}x{n}")


if __name__ == '__main__':
    create_html_report()
