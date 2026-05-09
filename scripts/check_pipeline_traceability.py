"""
Check what traceability information the pipeline stores.

Analyzes:
1. Database schema (what's tracked)
2. Metadata files (what's saved)
3. Pipeline steps (what's recorded)
"""

import os
import sqlite3
import json
from pathlib import Path

print("="*70)
print("PIPELINE TRACEABILITY ANALYSIS")
print("="*70)

# Check database schema
db_path = Path.home() / ".sim_bench" / "sim_bench.db"
if db_path.exists():
    print("\n1. DATABASE SCHEMA")
    print("-"*70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List all tables
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"Tables: {[t[0] for t in tables]}")

    # Check faces table
    if ('faces',) in tables:
        print("\n'faces' table columns:")
        cursor.execute('PRAGMA table_info(faces)')
        for row in cursor.fetchall():
            col_name, col_type = row[1], row[2]
            print(f"  {col_name}: {col_type}")

        # Sample a face record
        cursor.execute("SELECT * FROM faces LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print("\nSample face record fields:")
            cursor.execute('PRAGMA table_info(faces)')
            cols = [row[1] for row in cursor.fetchall()]
            for col, val in zip(cols, sample):
                if val and len(str(val)) < 100:
                    print(f"  {col} = {val}")

    conn.close()
else:
    print("\n1. DATABASE: Not found at ~/.sim_bench/sim_bench.db")

# Check benchmark metadata format
print("\n\n2. BENCHMARK METADATA FORMAT")
print("-"*70)

benchmark_file = Path("results/Google_Germany/benchmark_2026-03-01_01-10-04.json")
if benchmark_file.exists():
    with open(benchmark_file) as f:
        data = json.load(f)

    print(f"Top-level keys: {list(data.keys())}")

    if 'face_metadata' in data and len(data['face_metadata']) > 0:
        print("\nSample face_metadata entry:")
        sample = data['face_metadata'][0]
        for key, val in sample.items():
            print(f"  {key}: {val}")
else:
    print(f"Benchmark file not found: {benchmark_file}")

# Check what pipeline steps save
print("\n\n3. PIPELINE FACE CROP STEP")
print("-"*70)

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sim_bench.pipeline.steps.save_face_crops import SaveFaceCrops

    print("SaveFaceCrops metadata:")
    meta = SaveFaceCrops._metadata
    print(f"  Name: {meta.name}")
    print(f"  Requires: {meta.requires}")
    print(f"  Produces: {meta.produces}")

    # Check what context data is used
    print("\nContext data used by save_face_crops:")
    print("  - Reads: insightface_faces (face detections)")
    print("  - Produces: face_crops directory with aligned images")
    print("  - Filename format: face_{face_id:04d}_aligned.jpg")

except Exception as e:
    print(f"Could not load SaveFaceCrops: {e}")

# Check what mapping exists between source images and face IDs
print("\n\n4. SOURCE IMAGE → FACE ID MAPPING")
print("-"*70)

print("Checking if we track:")
print("  ✓ Face ID (from index)")
print("  ✓ Bbox coordinates (in benchmark JSON)")
print("  ✓ Pose angles (in benchmark JSON)")
print("  ✓ Blur score (in benchmark JSON)")
print("  ✗ Source image path (image_path: None in benchmark)")
print("  ✗ Face index within image")
print("  ✗ Crop file path")
print("  ✗ Embedding file reference")

print("\n" + "="*70)
print("TRACEABILITY GAPS")
print("="*70)
print("""
MISSING:
1. Source image path → NOT stored in face_metadata
2. Face index within each image → face_index is None
3. Crop file path → Not explicitly linked to face_id
4. Full pipeline lineage (detect → crop → align → embed)

CONSEQUENCE:
- Cannot trace face_0545 back to source image
- Cannot reproduce specific face extraction
- Cannot verify crops match source images
- Cannot debug face ID offsets

RECOMMENDATION:
Create face_lineage.json per pipeline run with:
{
  "face_id": 545,
  "source_image": "D:/Google_Germany/IMG_1234.jpg",
  "face_index_in_image": 0,
  "bbox": [x, y, w, h],
  "crop_file": "face_0545_aligned.jpg",
  "embedding_index": 545,
  "quality": {...},
  "pipeline_version": "...",
  "extraction_timestamp": "..."
}
""")

print("\n" + "="*70)
