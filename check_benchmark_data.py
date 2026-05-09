"""Check what data is available in benchmark results."""

from pathlib import Path
import numpy as np

results = Path('results/face_clustering_benchmark')

# Check embeddings
npy_files = sorted(results.glob('embeddings_*.npy'), reverse=True)
print(f"Embedding files: {len(npy_files)}")
for f in npy_files[:3]:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} - {size_kb:.1f} KB")

if len(npy_files) > 0:
    # Load most recent
    embeddings = np.load(npy_files[0])
    print(f"\nMost recent embeddings: {npy_files[0].name}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  {embeddings.shape[0]} faces with {embeddings.shape[1]}-dim embeddings")

# Check face crops
crops_dir = results / 'face_crops'
print(f"\nFace crops directory: {crops_dir.exists()}")
if crops_dir.exists():
    jpg_files = list(crops_dir.glob('*.jpg'))
    png_files = list(crops_dir.glob('*.png'))
    print(f"  JPG files: {len(jpg_files)}")
    print(f"  PNG files: {len(png_files)}")

    if len(jpg_files) > 0:
        print(f"\nSample files:")
        for f in jpg_files[:5]:
            print(f"  {f.name}")

# Check metadata
metadata_files = list(results.glob('face_metadata_*.json'))
print(f"\nMetadata files: {len(metadata_files)}")
for f in metadata_files[:3]:
    print(f"  {f.name}")
