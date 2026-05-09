"""Check what face tracking information exists in the database."""

import os
import sqlite3
import json
from pathlib import Path

db_path = Path.home() / ".sim_bench" / "sim_bench.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("="*70)
print("FACE TRACKING IN DATABASE")
print("="*70)

# Check people table
cursor.execute("SELECT face_instances FROM people WHERE face_instances IS NOT NULL LIMIT 1")
row = cursor.fetchone()

if row and row[0]:
    instances = json.loads(row[0])
    print(f"\nface_instances structure:")
    print(f"  Type: {type(instances)}")

    if isinstance(instances, list):
        print(f"  Count: {len(instances)}")
        if len(instances) > 0:
            print(f"\n  First entry:")
            for key, val in instances[0].items():
                if isinstance(val, (int, float, str)):
                    print(f"    {key}: {val}")
                elif isinstance(val, dict):
                    print(f"    {key}: {{{', '.join(val.keys())}}}")
                else:
                    print(f"    {key}: {type(val)}")
else:
    print("\nNo face_instances data found")

# Check pipeline_results
print("\n\nPIPELINE_RESULTS table:")
cursor.execute("SELECT key, value FROM pipeline_results WHERE key LIKE '%face%' OR key LIKE '%crop%' LIMIT 5")
results = cursor.fetchall()
if results:
    for key, val in results:
        if val and len(str(val)) < 200:
            print(f"  {key}: {val}")
else:
    cursor.execute("SELECT key FROM pipeline_results LIMIT 10")
    keys = [r[0] for r in cursor.fetchall()]
    print(f"  Available keys: {keys}")

conn.close()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
The 'people' table has face_instances JSON which SHOULD contain:
  - image_path
  - face_index_in_image
  - bbox
  - quality scores

This provides the traceability we need!

Next: Verify this data is populated correctly during pipeline runs.
""")
