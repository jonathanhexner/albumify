"""
Query face lineage: given a face_id or face crop filename, find the source image.

Usage:
    python scripts/query_face_lineage.py 545
    python scripts/query_face_lineage.py face_0545_aligned.jpg
"""

import os
import sqlite3
import json
import sys
from pathlib import Path


def get_face_lineage_from_db(run_id=None):
    """
    Extract face lineage from database (people.face_instances).

    Returns:
        dict: {face_id: {image_path, face_index, bbox, ...}}
    """
    db_path = Path.home() / ".sim_bench" / "sim_bench.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all face instances from people table
    if run_id:
        cursor.execute("SELECT face_instances FROM people WHERE run_id = ?", (run_id,))
    else:
        cursor.execute("SELECT face_instances FROM people WHERE face_instances IS NOT NULL")

    lineage = {}
    face_id = 0  # Global face counter across all people

    for row in cursor.fetchall():
        if row[0]:
            instances = json.loads(row[0])
            for instance in instances:
                lineage[face_id] = instance
                face_id += 1

    conn.close()
    return lineage


def query_face(face_id=None, face_filename=None):
    """Query face lineage by face_id or filename."""

    if face_filename:
        # Extract face_id from filename: face_0545_aligned.jpg -> 545
        import re
        match = re.search(r'face_(\d+)', face_filename)
        if match:
            face_id = int(match.group(1))

    if face_id is None:
        print("ERROR: Must provide face_id or face_filename")
        return

    print(f"Querying face {face_id}...")
    lineage = get_face_lineage_from_db()

    if face_id in lineage:
        info = lineage[face_id]
        print(f"\nFace {face_id} lineage:")
        print(f"  Source image: {info.get('image_path', 'N/A')}")
        print(f"  Face index in image: {info.get('face_index', 'N/A')}")
        print(f"  Bbox: {info.get('bbox', 'N/A')}")
        print(f"  Assignment: {info.get('assignment_method', 'N/A')} (confidence: {info.get('assignment_confidence', 'N/A')})")
    else:
        print(f"\nFace {face_id} NOT FOUND in database")
        print(f"Available face IDs: 0-{max(lineage.keys()) if lineage else 0}")


def export_lineage_csv(output_file="face_lineage.csv"):
    """Export all face lineage to CSV for analysis."""
    import csv

    lineage = get_face_lineage_from_db()

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['face_id', 'image_path', 'face_index', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'assignment_method', 'assignment_confidence'])

        for face_id, info in sorted(lineage.items()):
            bbox = info.get('bbox', [None, None, None, None])
            writer.writerow([
                face_id,
                info.get('image_path', ''),
                info.get('face_index', ''),
                bbox[0] if len(bbox) > 0 else '',
                bbox[1] if len(bbox) > 1 else '',
                bbox[2] if len(bbox) > 2 else '',
                bbox[3] if len(bbox) > 3 else '',
                info.get('assignment_method', ''),
                info.get('assignment_confidence', '')
            ])

    print(f"Exported {len(lineage)} faces to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--export':
            export_lineage_csv()
        elif arg.startswith('face_'):
            query_face(face_filename=arg)
        else:
            try:
                face_id = int(arg)
                query_face(face_id=face_id)
            except ValueError:
                print(f"ERROR: Invalid argument: {arg}")
                print("Usage: python scripts/query_face_lineage.py <face_id>")
                print("       python scripts/query_face_lineage.py face_0545_aligned.jpg")
                print("       python scripts/query_face_lineage.py --export")
    else:
        print("Usage:")
        print("  python scripts/query_face_lineage.py 545")
        print("  python scripts/query_face_lineage.py face_0545_aligned.jpg")
        print("  python scripts/query_face_lineage.py --export")
