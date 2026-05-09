"""
Isolated embedding verification test.

Tests embedding extraction on a small set of face crops in complete isolation
to rule out any file mismatch or caching issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import json

from face_cluster import InsightFaceEmbedder

def main():
    test_dir = Path("results/embedding_verification_test")

    print("="*70)
    print("ISOLATED EMBEDDING VERIFICATION TEST")
    print("="*70)
    print(f"\nTest directory: {test_dir}")
    print("User confirmation:")
    print("  - Faces 545, 569, 573: Person 1 (SAME person)")
    print("  - Faces 550, 551: Person 2 (SAME person)")
    print("  - Face 557: Profile (different)")
    print("  - Face 546: DIFFERENT person from 545")
    print()

    # Initialize embedder
    print("Initializing InsightFace embedder...")
    embedder = InsightFaceEmbedder(model_name='buffalo_l')
    print("OK Embedder ready\n")

    # Test faces in order
    test_faces = [545, 546, 550, 551, 557, 569, 573]

    # Extract embeddings fresh
    embeddings = {}
    images = {}

    print("Extracting fresh embeddings from isolated crops:")
    print("-" * 70)

    for face_id in test_faces:
        crop_path = test_dir / f"face_{face_id:04d}_aligned.jpg"

        if not crop_path.exists():
            print(f"  ERROR: {crop_path.name} not found")
            continue

        # Load image
        img_pil = Image.open(crop_path)
        img_np = np.array(img_pil)
        images[face_id] = img_np

        # Ensure RGB
        if len(img_np.shape) == 2:
            import cv2
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            import cv2
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        # Extract embedding
        embedding = embedder.get_embedding(img_np)

        if embedding is None:
            print(f"  ERROR: Failed to extract embedding for {face_id}")
            continue

        embeddings[face_id] = embedding
        print(f"  face_{face_id:04d}: shape {embedding.shape}, norm {np.linalg.norm(embedding):.4f}")

    print(f"\nSuccessfully extracted {len(embeddings)} embeddings")

    # Compute distance matrix
    print("\n" + "="*70)
    print("PAIRWISE COSINE DISTANCES")
    print("="*70)

    face_ids = sorted(embeddings.keys())

    # Print header
    print(f"\n{'':>8}", end="")
    for fid in face_ids:
        print(f"{fid:>8}", end="")
    print()
    print("-" * (8 + 8 * len(face_ids)))

    # Print distance matrix
    distance_matrix = {}
    for i, fid_a in enumerate(face_ids):
        print(f"{fid_a:>8}", end="")
        distance_matrix[fid_a] = {}
        for j, fid_b in enumerate(face_ids):
            if fid_a == fid_b:
                print(f"{'--':>8}", end="")
                distance_matrix[fid_a][fid_b] = 0.0
            else:
                dist = 1.0 - np.dot(embeddings[fid_a], embeddings[fid_b])
                distance_matrix[fid_a][fid_b] = float(dist)
                print(f"{dist:>8.3f}", end="")
        print()

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Person 1 analysis (545, 569, 573)
    person1 = [545, 569, 573]
    print("\nPerson 1 (faces 545, 569, 573) - Should be SAME person:")
    distances_p1 = []
    for i, f1 in enumerate(person1):
        for f2 in person1[i+1:]:
            if f1 in embeddings and f2 in embeddings:
                dist = distance_matrix[f1][f2]
                distances_p1.append(dist)
                status = "OK" if dist < 0.35 else "HIGH"
                print(f"  {f1} <-> {f2}: {dist:.4f} [{status}]")

    if distances_p1:
        avg_p1 = np.mean(distances_p1)
        max_p1 = np.max(distances_p1)
        print(f"  Average: {avg_p1:.4f}, Max: {max_p1:.4f}")
        if max_p1 < 0.35:
            print("  => OK Person 1 is cohesive")
        else:
            print("  => WARNING Person 1 has high internal distances")

    # Person 2 analysis (550, 551)
    person2 = [550, 551]
    print("\nPerson 2 (faces 550, 551) - Should be SAME person:")
    if 550 in embeddings and 551 in embeddings:
        dist = distance_matrix[550][551]
        status = "OK" if dist < 0.35 else "HIGH"
        print(f"  550 <-> 551: {dist:.4f} [{status}]")

    # Critical test: 545 vs 546 (should be DIFFERENT)
    print("\nCritical Test - Face 545 vs 546 (User says DIFFERENT people):")
    if 545 in embeddings and 546 in embeddings:
        dist_545_546 = distance_matrix[545][546]
        status = "OK" if dist_545_546 > 0.50 else "ERROR"
        print(f"  545 <-> 546: {dist_545_546:.4f} [{status}]")

        if dist_545_546 < 0.50:
            print("  => ERROR: Distance too small for different people!")
            print("     Expected: > 0.50 for different people")
            print("     Got: {:.4f}".format(dist_545_546))
        else:
            print("  => OK: Distance indicates different people")

    # Find closest neighbor for 545
    print("\nFace 545 closest neighbors:")
    if 545 in embeddings:
        neighbors_545 = []
        for fid in face_ids:
            if fid != 545:
                dist = distance_matrix[545][fid]
                person = "Person 1" if fid in [545, 569, 573] else ("Person 2" if fid in [550, 551] else ("Profile" if fid == 557 else "Person 3"))
                neighbors_545.append((fid, dist, person))

        neighbors_545.sort(key=lambda x: x[1])

        for fid, dist, person in neighbors_545:
            expected = "Person 1"
            status = "CORRECT" if person == expected else "WRONG"
            print(f"  {fid}: {dist:.4f} ({person}) [{status}]")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if 545 in embeddings and 546 in embeddings:
        dist_545_546 = distance_matrix[545][546]

        if dist_545_546 < 0.15:
            print("\nERROR CRITICAL BUG DETECTED!")
            print(f"  Face 545 and 546 are VERY similar (distance {dist_545_546:.4f})")
            print(f"  But user confirms they are DIFFERENT people")
            print(f"\n  Possible causes:")
            print(f"    1. Face crop files are mislabeled (wrong faces in files)")
            print(f"    2. Face detection/alignment swapped faces during cropping")
            print(f"    3. Systematic bug in embedding extraction")

            print(f"\n  Next steps:")
            print(f"    1. Manually inspect face crops visually")
            print(f"    2. Trace back to original detection to find where swap occurred")
            print(f"    3. Re-run face detection from scratch on original images")

        elif dist_545_546 < 0.50:
            print("\nWARNING: Faces 545 and 546 are similar but distinguishable")
            print(f"  Distance: {dist_545_546:.4f}")
            print(f"  This could indicate:")
            print(f"    - Family members or similar-looking people")
            print(f"    - Sub-optimal face alignment")
            print(f"    - Need for stricter clustering thresholds")
        else:
            print("\nOK: Faces 545 and 546 are sufficiently different")
            print(f"  Distance: {dist_545_546:.4f} (> 0.50)")

    # Save results
    output_file = test_dir / "embedding_verification_results.json"
    results = {
        "test_date": "2026-03-30",
        "embeddings_extracted": len(embeddings),
        "distance_matrix": distance_matrix,
        "analysis": {
            "person1_faces": person1,
            "person2_faces": person2,
            "critical_test": {
                "face_a": 545,
                "face_b": 546,
                "distance": float(dist_545_546) if 545 in embeddings and 546 in embeddings else None,
                "expected": "different_people",
                "result": "FAIL" if dist_545_546 < 0.50 else "PASS"
            }
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Create HTML visualization
    create_html_report(test_dir, face_ids, distance_matrix, embeddings)

def create_html_report(test_dir: Path, face_ids: list, distance_matrix: dict, embeddings: dict):
    """Create HTML report with images and distances."""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Embedding Verification Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .face-grid { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }
        .face-card { border: 2px solid #333; padding: 10px; text-align: center; }
        .face-card img { width: 150px; height: 150px; }
        table { border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #333; padding: 8px; text-align: center; }
        .good { background: #90EE90; }
        .warn { background: #FFD700; }
        .bad { background: #FFB6C1; }
    </style>
</head>
<body>
    <h1>Embedding Verification Report</h1>
    <p><strong>Test</strong>: Isolated embedding extraction on 7 face crops</p>
    <p><strong>User Confirmation</strong>:</p>
    <ul>
        <li>Faces 545, 569, 573: Person 1 (SAME person)</li>
        <li>Faces 550, 551: Person 2 (SAME person)</li>
        <li>Face 546: DIFFERENT person from 545</li>
    </ul>

    <h2>Face Crops</h2>
    <div class="face-grid">
"""

    for face_id in face_ids:
        person = "Person 1" if face_id in [545, 569, 573] else ("Person 2" if face_id in [550, 551] else "Other")
        html += f"""
        <div class="face-card">
            <img src="face_{face_id:04d}_aligned.jpg">
            <div><strong>Face {face_id}</strong></div>
            <div>{person}</div>
        </div>
"""

    html += """
    </div>

    <h2>Distance Matrix</h2>
    <table>
        <tr>
            <th>Face ID</th>
"""

    for fid in face_ids:
        html += f"<th>{fid}</th>"

    html += "</tr>"

    for fid_a in face_ids:
        html += f"<tr><th>{fid_a}</th>"
        for fid_b in face_ids:
            if fid_a == fid_b:
                html += "<td>--</td>"
            else:
                dist = distance_matrix[fid_a][fid_b]
                css_class = "good" if dist > 0.50 else ("warn" if dist > 0.35 else "bad")
                html += f'<td class="{css_class}">{dist:.3f}</td>'
        html += "</tr>"

    html += """
    </table>

    <h2>Key Finding</h2>
"""

    if 545 in face_ids and 546 in face_ids:
        dist_545_546 = distance_matrix[545][546]
        if dist_545_546 < 0.15:
            html += f"""
    <div style="background: #FFB6C1; padding: 20px; border: 3px solid red;">
        <h3>ERROR: CRITICAL BUG DETECTED</h3>
        <p><strong>Face 545 and 546 distance: {dist_545_546:.4f}</strong></p>
        <p>These faces are VERY similar according to embeddings, but user confirms they are DIFFERENT people.</p>
        <p><strong>This indicates the face crop files are mislabeled or corrupted.</strong></p>
    </div>
"""
        elif dist_545_546 < 0.50:
            html += f"""
    <div style="background: #FFD700; padding: 20px; border: 3px solid orange;">
        <h3>WARNING: Borderline Case</h3>
        <p><strong>Face 545 and 546 distance: {dist_545_546:.4f}</strong></p>
        <p>Faces are similar but distinguishable. May need stricter thresholds.</p>
    </div>
"""
        else:
            html += f"""
    <div style="background: #90EE90; padding: 20px; border: 3px solid green;">
        <h3>OK: Embeddings Correct</h3>
        <p><strong>Face 545 and 546 distance: {dist_545_546:.4f}</strong></p>
        <p>Faces are sufficiently different as expected.</p>
    </div>
"""

    html += """
</body>
</html>
"""

    report_path = test_dir / "verification_report.html"
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"HTML report created: {report_path}")

if __name__ == '__main__':
    main()
