"""
Convert metadata.json to embeddings.npy format.

Extracts embeddings from metadata.json and saves as .npy file
compatible with export_clustering_data.py.
"""

import json
import numpy as np
from pathlib import Path
import sys

def convert_metadata_to_npy(metadata_path, output_dir=None):
    """Convert metadata.json to embeddings.npy."""

    metadata_path = Path(metadata_path)

    if output_dir is None:
        output_dir = metadata_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {metadata_path}")

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    detections = metadata['detections']
    n_faces = len(detections)

    print(f"Found {n_faces} faces")

    # Extract embeddings
    embeddings = []
    for det in detections:
        emb = np.array(det['embedding'], dtype=np.float32)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"Saved: {embeddings_file}")

    # Create metadata JSON (same format as benchmark script)
    metadata_output = {
        'n_faces': n_faces,
        'face_metadata': []
    }

    for det in detections:
        face_meta = {
            'face_index': det['face_index'],
            'image_path': det['source_path'],
            'bbox': {
                'x_px': det['bbox'][0],
                'y_px': det['bbox'][1],
                'w_px': det['bbox'][2] - det['bbox'][0],
                'h_px': det['bbox'][3] - det['bbox'][1]
            }
        }
        metadata_output['face_metadata'].append(face_meta)

    metadata_file = output_dir / "embeddings_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata_output, f, indent=2)
    print(f"Saved: {metadata_file}")

    return embeddings_file, metadata_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_metadata_to_npy.py <metadata.json> [output_dir]")
        sys.exit(1)

    metadata_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    embeddings_file, metadata_file = convert_metadata_to_npy(metadata_path, output_dir)

    print("\nDone! Now run:")
    print(f"python scripts/export_clustering_data.py --embeddings {embeddings_file} --output <output_dir>")
