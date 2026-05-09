"""
Analyze what order InsightFace returns faces in.

Tests if face detection is deterministic and what ordering convention is used.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from face_cluster import InsightFaceEmbedder


def analyze_detection_order(image_path, runs=3):
    """Run detection multiple times and check ordering."""
    print(f"\nAnalyzing: {Path(image_path).name}")
    print("-" * 70)

    embedder = InsightFaceEmbedder(model_name='buffalo_l')

    img = cv2.imread(str(image_path))
    if img is None:
        print("  ERROR: Could not load image")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run detection multiple times
    all_runs = []
    for run_num in range(runs):
        faces = embedder.app.get(img_rgb)

        face_data = []
        for i, face in enumerate(faces):
            bbox = face.bbox
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            face_data.append({
                'index': i,
                'bbox': bbox,
                'center': (center_x, center_y),
                'area': area,
                'score': face.det_score if hasattr(face, 'det_score') else None
            })

        all_runs.append(face_data)

    # Check if order is consistent across runs
    print(f"  Detected {len(all_runs[0])} faces")

    if runs > 1:
        consistent = True
        for run_idx in range(1, runs):
            for face_idx in range(min(len(all_runs[0]), len(all_runs[run_idx]))):
                bbox0 = all_runs[0][face_idx]['bbox']
                bbox1 = all_runs[run_idx][face_idx]['bbox']

                # Check if same face (bbox should be very close)
                if not np.allclose(bbox0, bbox1, atol=1.0):
                    consistent = False
                    break

        if consistent:
            print("  Order: CONSISTENT across runs")
        else:
            print("  Order: INCONSISTENT - may vary between runs!")

    # Analyze ordering pattern
    print("\n  Face positions (in detection order):")
    for i, face in enumerate(all_runs[0]):
        cx, cy = face['center']
        score_str = f"{face['score']:.3f}" if face['score'] else 'N/A'
        print(f"    {i}: center=({cx:.0f}, {cy:.0f}), area={face['area']:.0f}, score={score_str}")

    # Check if sorted by any obvious metric
    centers_y = [f['center'][1] for f in all_runs[0]]
    centers_x = [f['center'][0] for f in all_runs[0]]
    areas = [f['area'] for f in all_runs[0]]
    scores = [f['score'] for f in all_runs[0] if f['score']]

    print("\n  Ordering analysis:")

    # Check top-to-bottom
    if centers_y == sorted(centers_y):
        print("    - Sorted TOP to BOTTOM (by Y coordinate)")
    elif centers_y == sorted(centers_y, reverse=True):
        print("    - Sorted BOTTOM to TOP (by Y coordinate)")

    # Check left-to-right
    if centers_x == sorted(centers_x):
        print("    - Sorted LEFT to RIGHT (by X coordinate)")
    elif centers_x == sorted(centers_x, reverse=True):
        print("    - Sorted RIGHT to LEFT (by X coordinate)")

    # Check by area
    if areas == sorted(areas, reverse=True):
        print("    - Sorted by AREA (largest first)")
    elif areas == sorted(areas):
        print("    - Sorted by AREA (smallest first)")

    # Check by score
    if scores and scores == sorted(scores, reverse=True):
        print("    - Sorted by CONFIDENCE SCORE (highest first)")

    if not any([
        centers_y == sorted(centers_y),
        centers_x == sorted(centers_x),
        areas == sorted(areas, reverse=True),
        scores == sorted(scores, reverse=True) if scores else False
    ]):
        print("    - NO OBVIOUS ORDERING detected!")
        print("    - May be detection order (arbitrary)")


def propose_deterministic_ordering():
    """Propose a standard ordering convention."""
    print("\n" + "="*70)
    print("PROPOSED DETERMINISTIC ORDERING")
    print("="*70)
    print("""
Convention: Sort faces by READING ORDER (left-to-right, top-to-bottom)

Algorithm:
1. Sort faces primarily by Y coordinate (top to bottom)
2. Within same row (Y within threshold), sort by X coordinate (left to right)
3. Row threshold = average face height

This matches natural reading order and is most intuitive for users.

Implementation:
    def sort_faces_reading_order(faces):
        # Calculate average face height
        heights = [(f.bbox[3] - f.bbox[1]) for f in faces]
        avg_height = np.mean(heights)
        row_threshold = avg_height * 0.5  # 50% overlap = same row

        # Sort by reading order
        def sort_key(face):
            bbox = face.bbox
            center_y = (bbox[1] + bbox[3]) / 2
            center_x = (bbox[0] + bbox[2]) / 2

            # Group into rows (round Y to row_threshold)
            row = int(center_y / row_threshold)

            return (row, center_x)  # Sort by row, then X within row

        return sorted(faces, key=sort_key)
""")


if __name__ == '__main__':
    # Test on images with multiple faces
    test_images = [
        "test_data/source_images_ground_truth/20240816_150903.jpg",  # 5 faces
        "test_data/source_images_ground_truth/20240816_150905.jpg",  # 4 faces
        "test_data/source_images_ground_truth/20240818_121446.jpg",  # 4 faces
    ]

    print("="*70)
    print("FACE DETECTION ORDER ANALYSIS")
    print("="*70)

    for img_path in test_images:
        if Path(img_path).exists():
            analyze_detection_order(img_path, runs=2)

    propose_deterministic_ordering()
