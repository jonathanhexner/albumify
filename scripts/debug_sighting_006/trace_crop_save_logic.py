#!/usr/bin/env python3
"""
Additional debug script: Trace crop save logic to identify skipped faces.

This script simulates the save_face_crops() logic from benchmark_face_clustering.py
to identify exactly which faces were skipped and why.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse


def is_valid_bbox(w_px: int, h_px: int, min_size: int = 20) -> bool:
    """Check if bbox is valid (same logic as benchmark script)."""
    return w_px >= min_size and h_px >= min_size


class CropSaveTracer:
    """Traces the crop saving logic to identify skipped faces."""

    def __init__(self, metadata_path: Path, crops_dir: Path):
        self.metadata_path = metadata_path
        self.crops_dir = crops_dir

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Check which crop files exist
        self.existing_crops = set()
        for crop_file in crops_dir.glob("face_*_aligned.jpg"):
            # Extract index from filename
            face_id = int(crop_file.stem.split('_')[1])
            self.existing_crops.add(face_id)

    def trace_save_logic(self) -> Dict[str, Any]:
        """Simulate save_face_crops() to find skipped faces."""
        results = {
            "total_metadata": len(self.metadata),
            "total_saved_crops": len(self.existing_crops),
            "skipped_faces": [],
            "saved_mapping": [],  # (metadata_idx, saved_filename_idx)
            "failed_validations": []
        }

        saved_count = 0  # This is the counter used for filenames

        for i, face_meta in enumerate(self.metadata):
            face_id = face_meta.get('face_id')

            # Simulate validation checks
            bbox = face_meta.get('bbox', {})
            w_px = int(bbox.get('w_px', 0))
            h_px = int(bbox.get('h_px', 0))
            landmarks = face_meta.get('landmarks', [])

            # Check 1: Valid bbox?
            if not is_valid_bbox(w_px, h_px):
                results["skipped_faces"].append({
                    "metadata_index": i,
                    "face_id": face_id,
                    "reason": "invalid_bbox",
                    "details": f"w={w_px}, h={h_px}"
                })
                continue

            # Check 2: Has landmarks?
            if not landmarks or len(landmarks) < 5:
                results["skipped_faces"].append({
                    "metadata_index": i,
                    "face_id": face_id,
                    "reason": "missing_landmarks",
                    "details": f"landmarks_count={len(landmarks) if landmarks else 0}"
                })
                continue

            # Would save successfully
            expected_filename = f"face_{saved_count:04d}_aligned.jpg"
            actual_exists = saved_count in self.existing_crops

            results["saved_mapping"].append({
                "metadata_index": i,
                "metadata_face_id": face_id,
                "filename_index": saved_count,
                "expected_filename": expected_filename,
                "file_exists": actual_exists,
                "offset": face_id - saved_count if face_id is not None else None
            })

            saved_count += 1

        # Calculate offset statistics
        offsets = [m["offset"] for m in results["saved_mapping"] if m["offset"] is not None]
        if offsets:
            results["offset_stats"] = {
                "min": min(offsets),
                "max": max(offsets),
                "consistent": len(set(offsets)) == 1,
                "consistent_value": offsets[0] if len(set(offsets)) == 1 else None
            }

        return results

    def print_report(self, results: Dict[str, Any]):
        """Print detailed trace report."""
        print("=" * 80)
        print("CROP SAVE LOGIC TRACE")
        print("=" * 80)
        print()

        print(f"Total metadata entries: {results['total_metadata']}")
        print(f"Total saved crops found: {results['total_saved_crops']}")
        print(f"Skipped faces: {len(results['skipped_faces'])}")
        print()

        if results['skipped_faces']:
            print("=" * 80)
            print("SKIPPED FACES (Failed Validation)")
            print("=" * 80)
            for skip in results['skipped_faces']:
                print(f"  Metadata index {skip['metadata_index']} (face_id={skip['face_id']})")
                print(f"    Reason: {skip['reason']}")
                print(f"    Details: {skip['details']}")
            print()

        print("=" * 80)
        print("SAVED MAPPING (First 10)")
        print("=" * 80)
        print(f"{'Meta Idx':<10} {'Face ID':<10} {'File Idx':<10} {'Filename':<25} {'Offset':<10} {'Exists'}")
        print("-" * 80)
        for mapping in results['saved_mapping'][:10]:
            offset_str = str(mapping['offset']) if mapping['offset'] is not None else "N/A"
            exists_str = "✓" if mapping['file_exists'] else "✗"
            print(f"{mapping['metadata_index']:<10} "
                  f"{mapping['metadata_face_id']:<10} "
                  f"{mapping['filename_index']:<10} "
                  f"{mapping['expected_filename']:<25} "
                  f"{offset_str:<10} "
                  f"{exists_str}")
        print()

        if 'offset_stats' in results:
            print("=" * 80)
            print("OFFSET ANALYSIS")
            print("=" * 80)
            print(f"Min offset: {results['offset_stats']['min']}")
            print(f"Max offset: {results['offset_stats']['max']}")
            print(f"Consistent offset: {results['offset_stats']['consistent']}")
            if results['offset_stats']['consistent']:
                print(f"Offset value: +{results['offset_stats']['consistent_value']}")
            print()

        print("=" * 80)
        print("CONCLUSION")
        print("=" * 80)

        if results['skipped_faces']:
            print(f"✗ FOUND THE BUG: {len(results['skipped_faces'])} faces were skipped during save")
            print()
            print("ROOT CAUSE:")
            print("  benchmark_face_clustering.py line 340 uses 'saved_count' for filenames")
            print("  but metadata array uses original indices (0, 1, 2, ...)")
            print()
            print(f"IMPACT:")
            for i, skip in enumerate(results['skipped_faces']):
                print(f"  - Metadata[{skip['metadata_index']}] (face_id={skip['face_id']}) "
                      f"was skipped ({skip['reason']})")
            print()
            print("  This causes all subsequent faces to be saved with shifted filenames:")
            print(f"  - Metadata[{results['skipped_faces'][-1]['metadata_index'] + 1}] "
                  f"→ face_0000.jpg (should be face_{results['skipped_faces'][-1]['metadata_index'] + 1:04d}.jpg)")

            if 'offset_stats' in results and results['offset_stats']['consistent']:
                offset = results['offset_stats']['consistent_value']
                print()
                print(f"  Creating systematic +{offset} offset: stored[N] = fresh[N+{offset}]")
        else:
            print("✓ No faces were skipped - all metadata entries were saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Trace crop save logic to find skipped faces")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON file")
    parser.add_argument("--crops", required=True, help="Path to face_crops directory")

    args = parser.parse_args()

    tracer = CropSaveTracer(
        metadata_path=Path(args.metadata),
        crops_dir=Path(args.crops)
    )

    results = tracer.trace_save_logic()
    tracer.print_report(results)


if __name__ == "__main__":
    main()
