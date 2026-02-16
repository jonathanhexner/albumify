"""Quick script to debug specific faces from benchmark."""

import json
import sqlite3
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Faces to investigate
FACE_A_ID = 76
FACE_B_ID = 83

# Load benchmark results
results_dir = Path('results/face_clustering_benchmark')
benchmark_files = sorted(results_dir.glob('benchmark_*.json'), reverse=True)
benchmark_file = benchmark_files[0]

print(f"Loading: {benchmark_file.name}\n")

with open(benchmark_file, 'r') as f:
    results = json.load(f)

metadata = results['face_metadata']
embeddings_file = results_dir / results['embeddings_file']
embeddings = np.load(embeddings_file)

hdbscan_labels = results['methods']['hdbscan']['labels']
hybrid_labels = results['methods']['hybrid_knn']['labels']

# Get face metadata
face_a = metadata[FACE_A_ID]
face_b = metadata[FACE_B_ID]

print("="*70)
print(f"FACE {FACE_A_ID} (Real Face in Cluster 1)")
print("="*70)
print(f"Image: {Path(face_a['image_path']).name}")
print(f"Face index in image: {face_a['face_index']}")
print(f"Confidence: {face_a['confidence']:.3f}")
print(f"Frontal score: {face_a.get('frontal_score', 0):.3f}")
print(f"BBox: x={face_a['bbox']['x_px']}, y={face_a['bbox']['y_px']}, w={face_a['bbox']['w_px']}, h={face_a['bbox']['h_px']}")
print(f"HDBSCAN cluster: {hdbscan_labels[FACE_A_ID]}")
print(f"Hybrid cluster: {hybrid_labels[FACE_A_ID]}")

print("\n" + "="*70)
print(f"FACE {FACE_B_ID} (NOT a face)")
print("="*70)
print(f"Image: {Path(face_b['image_path']).name}")
print(f"Face index in image: {face_b['face_index']}")
print(f"Confidence: {face_b['confidence']:.3f}")
print(f"Frontal score: {face_b.get('frontal_score', 0):.3f}")
print(f"BBox: x={face_b['bbox']['x_px']}, y={face_b['bbox']['y_px']}, w={face_b['bbox']['w_px']}, h={face_b['bbox']['h_px']}")
print(f"HDBSCAN cluster: {hdbscan_labels[FACE_B_ID]}")
print(f"Hybrid cluster: {hybrid_labels[FACE_B_ID]}")

# Embedding distance
emb_a = embeddings[FACE_A_ID]
emb_b = embeddings[FACE_B_ID]
distance = cosine(emb_a, emb_b)

print("\n" + "="*70)
print(f"EMBEDDING DISTANCE: {distance:.4f}")
print("="*70)
if distance < 0.3:
    print("  [!] VERY SIMILAR (< 0.3) - Embeddings think they're the SAME person!")
elif distance < 0.5:
    print("  [!] SIMILAR (0.3-0.5) - Embeddings think they're likely same person")
else:
    print("  [OK] DIFFERENT (> 0.5) - Embeddings correctly see them as different")

print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Create visualization
fig = plt.figure(figsize=(20, 12))

# Row 1: Saved crops
ax1 = plt.subplot(3, 2, 1)
crop_a_path = results_dir / 'face_crops' / f'face_{FACE_A_ID:04d}.jpg'
if crop_a_path.exists():
    crop_a = Image.open(crop_a_path)
    ax1.imshow(crop_a)
    ax1.set_title(f"Face {FACE_A_ID}: SAVED CROP\n(Real face)", fontsize=14, fontweight='bold')
else:
    ax1.text(0.5, 0.5, 'CROP NOT FOUND', ha='center', va='center')
    ax1.set_title(f"Face {FACE_A_ID}: CROP MISSING", fontsize=14, color='red')
ax1.axis('off')

ax2 = plt.subplot(3, 2, 2)
crop_b_path = results_dir / 'face_crops' / f'face_{FACE_B_ID:04d}.jpg'
if crop_b_path.exists():
    crop_b = Image.open(crop_b_path)
    ax2.imshow(crop_b)
    ax2.set_title(f"Face {FACE_B_ID}: SAVED CROP\n(NOT a face!)", fontsize=14, fontweight='bold', color='red')
else:
    ax2.text(0.5, 0.5, 'CROP NOT FOUND', ha='center', va='center')
    ax2.set_title(f"Face {FACE_B_ID}: CROP MISSING", fontsize=14, color='red')
ax2.axis('off')

# Row 2: Original images with bboxes
def draw_bbox(img, bbox, color='red', width=5, label=None):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    x, y, w, h = bbox['x_px'], bbox['y_px'], bbox['w_px'], bbox['h_px']
    draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=width)
    if label:
        draw.text((x, y - 25), label, fill=color)
    return img_copy

ax3 = plt.subplot(3, 2, 3)
img_a = ImageOps.exif_transpose(Image.open(face_a['image_path']))
img_a_bbox = draw_bbox(img_a, face_a['bbox'], label=f"Face {FACE_A_ID}")
ax3.imshow(img_a_bbox)
ax3.set_title(f"Face {FACE_A_ID}: ORIGINAL IMAGE\n{Path(face_a['image_path']).name}", fontsize=12)
ax3.axis('off')

ax4 = plt.subplot(3, 2, 4)
img_b = ImageOps.exif_transpose(Image.open(face_b['image_path']))
img_b_bbox = draw_bbox(img_b, face_b['bbox'], label=f"Face {FACE_B_ID}")
ax4.imshow(img_b_bbox)
ax4.set_title(f"Face {FACE_B_ID}: ORIGINAL IMAGE\n{Path(face_b['image_path']).name}", fontsize=12)
ax4.axis('off')

# Row 3: Zoomed to bbox region
ax5 = plt.subplot(3, 2, 5)
bbox_a = face_a['bbox']
pad = 50
left = max(0, bbox_a['x_px'] - pad)
top = max(0, bbox_a['y_px'] - pad)
right = min(img_a.width, bbox_a['x_px'] + bbox_a['w_px'] + pad)
bottom = min(img_a.height, bbox_a['y_px'] + bbox_a['h_px'] + pad)
zoom_a = img_a.crop((left, top, right, bottom))
ax5.imshow(zoom_a)
ax5.set_title(f"Face {FACE_A_ID}: ZOOMED REGION", fontsize=12)
ax5.axis('off')

ax6 = plt.subplot(3, 2, 6)
bbox_b = face_b['bbox']
left = max(0, bbox_b['x_px'] - pad)
top = max(0, bbox_b['y_px'] - pad)
right = min(img_b.width, bbox_b['x_px'] + bbox_b['w_px'] + pad)
bottom = min(img_b.height, bbox_b['y_px'] + bbox_b['h_px'] + pad)
zoom_b = img_b.crop((left, top, right, bottom))
ax6.imshow(zoom_b)
ax6.set_title(f"Face {FACE_B_ID}: ZOOMED REGION", fontsize=12)
ax6.axis('off')

plt.tight_layout()
output_path = results_dir / f'debug_face_{FACE_A_ID}_vs_{FACE_B_ID}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")
print("\nOpen this image to see the full comparison!")

# Database check
print("\n" + "="*70)
print("DATABASE VERIFICATION")
print("="*70)

db_path = Path.home() / '.sim_bench' / 'sim_bench.db'
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

def get_db_detection(image_path, face_idx):
    cursor.execute("""
        SELECT data_blob
        FROM universal_cache
        WHERE feature_type = 'insightface_detection'
        AND image_path LIKE ?
    """, (f'%{Path(image_path).name}',))
    row = cursor.fetchone()
    if row:
        data = json.loads(row[0])
        faces = data.get('faces', [])
        if face_idx < len(faces):
            return faces[face_idx]
    return None

db_a = get_db_detection(face_a['image_path'], face_a['face_index'])
db_b = get_db_detection(face_b['image_path'], face_b['face_index'])

print(f"\nFace {FACE_A_ID} - Database check:")
if db_a:
    print(f"  [OK] Found in database")
    print(f"  Confidence: {db_a['confidence']:.3f} (metadata: {face_a['confidence']:.3f})")
    print(f"  BBox match: {db_a['bbox']['x_px'] == face_a['bbox']['x_px']}")
else:
    print(f"  [ERROR] NOT FOUND in database")

print(f"\nFace {FACE_B_ID} - Database check:")
if db_b:
    print(f"  [OK] Found in database")
    print(f"  Confidence: {db_b['confidence']:.3f} (metadata: {face_b['confidence']:.3f})")
    print(f"  BBox match: {db_b['bbox']['x_px'] == face_b['bbox']['x_px']}")
    print(f"\n  [WARNING] This confirms: InsightFace DID detect this as a 'face'")
    print(f"     -> Detector is hallucinating / has false positive")
else:
    print(f"  [ERROR] NOT FOUND in database")

conn.close()

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("Based on the visualization and data above:")
print("\n1. Check if Face 83's crop actually shows a face")
print("2. Check the confidence score of Face 83")
print("3. If detector confidence is high but it's not a face:")
print("   → Need to increase detection_threshold OR add post-filtering")
print("4. If both faces have similar embeddings:")
print("   → ArcFace is also confused by the non-face")
print("="*70)
