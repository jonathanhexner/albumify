"""
Verify we can trace all ground truth faces back to source images.

Checks if we have the data needed to create a full pipeline test.
"""

import os
import sqlite3
import json
from pathlib import Path

# Our ground truth face IDs
GROUND_TRUTH_FACES = [
    545, 546, 550, 551, 557, 569, 573,  # Original 7
    634, 637, 558, 580, 562, 584, 587, 589  # New 8
]

def get_face_lineage():
    """Get all face lineage from database."""
    db_path = Path.home() / ".sim_bench" / "sim_bench.db"

    if not db_path.exists():
        return None, "Database not found"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT face_instances FROM people WHERE face_instances IS NOT NULL")

    lineage = {}
    face_id = 0

    for row in cursor.fetchall():
        if row[0]:
            instances = json.loads(row[0])
            for instance in instances:
                lineage[face_id] = instance
                face_id += 1

    conn.close()
    return lineage, None


print("="*70)
print("GROUND TRUTH TRACEABILITY CHECK")
print("="*70)

lineage, error = get_face_lineage()

if error:
    print(f"\nERROR: {error}")
    print("\nCONCLUSION: Cannot trace faces - database not available")
    exit(1)

print(f"\nTotal faces in database: {len(lineage)}")
print(f"Ground truth faces to check: {len(GROUND_TRUTH_FACES)}")

# Check each ground truth face
found = {}
missing = []
source_images = set()

print("\n" + "-"*70)
print("CHECKING EACH GROUND TRUTH FACE")
print("-"*70)

for face_id in GROUND_TRUTH_FACES:
    if face_id in lineage:
        info = lineage[face_id]
        image_path = info.get('image_path')
        face_index = info.get('face_index')

        if image_path and face_index is not None:
            found[face_id] = info
            source_images.add(image_path)
            status = "OK"
        else:
            status = "INCOMPLETE (missing path/index)"
            missing.append(face_id)

        print(f"  Face {face_id:3d}: {status}")
        if image_path:
            print(f"             {Path(image_path).name} (face {face_index})")
    else:
        missing.append(face_id)
        print(f"  Face {face_id:3d}: NOT FOUND in database")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nTraceability status:")
print(f"  Found with full data: {len(found)}/{len(GROUND_TRUTH_FACES)}")
print(f"  Missing or incomplete: {len(missing)}/{len(GROUND_TRUTH_FACES)}")

if missing:
    print(f"\n  Missing faces: {missing}")

print(f"\nUnique source images needed: {len(source_images)}")

if source_images:
    print("\nSource images:")
    for img_path in sorted(source_images):
        path = Path(img_path)
        exists = path.exists()
        status = "EXISTS" if exists else "NOT FOUND"
        print(f"  [{status}] {path}")

# Conclusion
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if len(found) == len(GROUND_TRUTH_FACES):
    print("\nOK: All ground truth faces can be traced!")

    # Check if source images exist
    existing_images = [p for p in source_images if Path(p).exists()]

    if len(existing_images) == len(source_images):
        print("OK: All source images exist on disk!")
        print("\nNEXT STEP: Copy source images to test_data and create pipeline test")
    else:
        print(f"\nWARNING: Only {len(existing_images)}/{len(source_images)} source images exist")
        print("ACTION: Locate missing source images or update paths")
else:
    print(f"\nPROBLEM: {len(missing)} faces cannot be traced")
    print("ACTION: Need to rerun pipeline or fix database to populate lineage")
