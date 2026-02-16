"""Trace exactly what happened with Face 83."""

import json
from pathlib import Path

# Load benchmark
results_dir = Path('results/face_clustering_benchmark')
benchmark_files = sorted(results_dir.glob('benchmark_*.json'), reverse=True)
benchmark_file = benchmark_files[0]

with open(benchmark_file, 'r') as f:
    results = json.load(f)

metadata = results['face_metadata']
total_faces = len(metadata)

print(f"Total faces in benchmark: {total_faces}")
print(f"(This is AFTER filtering to only faces with saved crops)\n")

# Face 83 in the benchmark
face_83 = metadata[83]

print("="*70)
print("FACE 83 IN BENCHMARK METADATA")
print("="*70)
print(f"Image: {face_83['image_path']}")
print(f"Face index in that image: {face_83['face_index']}")
print(f"BBox: x={face_83['bbox']['x_px']}, y={face_83['bbox']['y_px']}, "
      f"w={face_83['bbox']['w_px']}, h={face_83['bbox']['h_px']}")
print(f"Confidence: {face_83['confidence']:.3f}")

print("\n" + "="*70)
print("EXPECTED CROP FILE")
print("="*70)
print(f"Face 83 should be saved as: face_0083.jpg")
print(f"And it should contain the crop from:")
print(f"  Image: {Path(face_83['image_path']).name}")
print(f"  Region: ({face_83['bbox']['x_px']}, {face_83['bbox']['y_px']}) "
      f"to ({face_83['bbox']['x_px'] + face_83['bbox']['w_px']}, "
      f"{face_83['bbox']['y_px'] + face_83['bbox']['h_px']})")

# Check if crop exists
crop_path = results_dir / 'face_crops' / 'face_0083.jpg'
print(f"\nCrop file exists: {crop_path.exists()}")

# Now let's check Face 76 for comparison
print("\n\n" + "="*70)
print("FACE 76 IN BENCHMARK METADATA (for comparison)")
print("="*70)
face_76 = metadata[76]
print(f"Image: {face_76['image_path']}")
print(f"Face index in that image: {face_76['face_index']}")
print(f"BBox: x={face_76['bbox']['x_px']}, y={face_76['bbox']['y_px']}, "
      f"w={face_76['bbox']['w_px']}, h={face_76['bbox']['h_px']}")
print(f"Confidence: {face_76['confidence']:.3f}")

crop_76_path = results_dir / 'face_crops' / 'face_0076.jpg'
print(f"\nCrop file exists: {crop_76_path.exists()}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("If:")
print("  - face_0083.jpg EXISTS")
print("  - But shows different content than the bbox in metadata[83]")
print("Then:")
print("  -> The crop was saved from a DIFFERENT image/bbox")
print("  -> There's still a mismatch between crop filenames and metadata indices")
print("\nPossible causes:")
print("1. The filtering created metadata[83] but face_0083.jpg is from a different face")
print("2. Some faces were saved multiple times or skipped")
print("3. The saved_indices list is wrong")
