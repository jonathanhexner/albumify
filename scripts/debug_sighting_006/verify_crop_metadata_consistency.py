#!/usr/bin/env python3
"""
Verify consistency between crop filenames, metadata, and embeddings.

This script checks if:
1. Crop filenames match metadata face_ids
2. Embeddings array indices match crop filenames
3. All three are properly aligned
"""

import json
import numpy as np
from pathlib import Path
import argparse
import re


def extract_face_id_from_filename(filename: str) -> int:
    """Extract numeric face_id from 'face_0123_aligned.jpg'."""
    match = re.search(r'face_(\d+)_aligned', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract face_id from filename: {filename}")


def verify_consistency(crops_dir: Path, metadata_path: Path, embeddings_path: Path):
    """Check alignment between crops, metadata, and embeddings."""

    print("=" * 80)
    print("CROP/METADATA/EMBEDDINGS CONSISTENCY CHECK")
    print("=" * 80)
    print()

    # Load data
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    embeddings = np.load(embeddings_path)

    crop_files = sorted(crops_dir.glob("face_*_aligned.jpg"))
    crop_face_ids = [extract_face_id_from_filename(f.name) for f in crop_files]

    print(f"Metadata entries: {len(metadata)}")
    print(f"Crop files: {len(crop_files)}")
    print(f"Embeddings: {embeddings.shape[0]}")
    print()

    # Check 1: Crop filenames sequential?
    print("=" * 80)
    print("CHECK 1: Are crop filenames sequential?")
    print("=" * 80)

    expected_sequence = list(range(len(crop_files)))
    if crop_face_ids == expected_sequence:
        print("✓ PASS: Crop filenames are sequential (0, 1, 2, ...)")
    else:
        print("✗ FAIL: Crop filenames have gaps or are not sequential")
        print(f"  Expected: {expected_sequence[:10]} ...")
        print(f"  Actual:   {crop_face_ids[:10]} ...")

        # Find gaps
        missing = set(expected_sequence) - set(crop_face_ids)
        if missing:
            print(f"  Missing indices: {sorted(missing)[:20]} ...")
    print()

    # Check 2: Metadata face_ids match crop filenames?
    print("=" * 80)
    print("CHECK 2: Do metadata face_ids match crop filenames?")
    print("=" * 80)

    metadata_face_ids = [entry.get('face_id') for entry in metadata]

    matches = 0
    mismatches = []

    for i in range(min(len(metadata_face_ids), len(crop_face_ids))):
        meta_id = metadata_face_ids[i]
        crop_id = crop_face_ids[i]

        if meta_id == crop_id:
            matches += 1
        else:
            mismatches.append({
                "index": i,
                "metadata_face_id": meta_id,
                "crop_face_id": crop_id,
                "offset": meta_id - crop_id if meta_id is not None and crop_id is not None else None
            })

    if matches == min(len(metadata_face_ids), len(crop_face_ids)):
        print(f"✓ PASS: All {matches} entries match perfectly")
    else:
        print(f"✗ FAIL: {len(mismatches)} mismatches out of {min(len(metadata_face_ids), len(crop_face_ids))}")
        print()
        print("First 10 mismatches:")
        for mm in mismatches[:10]:
            print(f"  Index {mm['index']}: metadata says {mm['metadata_face_id']}, "
                  f"crop filename is {mm['crop_face_id']} (offset: {mm['offset']})")

        # Check if offset is consistent
        offsets = [mm['offset'] for mm in mismatches if mm['offset'] is not None]
        if offsets and len(set(offsets)) == 1:
            print()
            print(f"  ⚠️  CONSISTENT OFFSET DETECTED: +{offsets[0]}")
            print(f"      This means metadata[i] refers to face_{i+offsets[0]:04d}_aligned.jpg")
    print()

    # Check 3: Embeddings array length matches?
    print("=" * 80)
    print("CHECK 3: Does embeddings array length match crop count?")
    print("=" * 80)

    if len(embeddings) == len(crop_files):
        print(f"✓ PASS: Both have {len(embeddings)} entries")
    else:
        print(f"✗ FAIL: Embeddings has {len(embeddings)} entries, crops has {len(crop_files)}")
        print(f"  Difference: {abs(len(embeddings) - len(crop_files))}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if matches == min(len(metadata_face_ids), len(crop_face_ids)) and \
       len(embeddings) == len(crop_files) and \
       crop_face_ids == expected_sequence:
        print("✓ ALL CHECKS PASSED: Crops, metadata, and embeddings are perfectly aligned")
    else:
        print("✗ INCONSISTENCIES DETECTED:")
        if crop_face_ids != expected_sequence:
            print("  - Crop filenames have gaps (some faces were skipped during save)")
        if matches < min(len(metadata_face_ids), len(crop_face_ids)):
            print("  - Metadata face_ids don't match crop filenames")
            if offsets and len(set(offsets)) == 1:
                print(f"    (Systematic +{offsets[0]} offset)")
        if len(embeddings) != len(crop_files):
            print("  - Embeddings array length doesn't match crop count")


def main():
    parser = argparse.ArgumentParser(description="Verify crop/metadata/embeddings consistency")
    parser.add_argument("--crops", required=True, help="Path to face_crops directory")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings NPY")

    args = parser.parse_args()

    verify_consistency(
        crops_dir=Path(args.crops),
        metadata_path=Path(args.metadata),
        embeddings_path=Path(args.embeddings)
    )


if __name__ == "__main__":
    main()
