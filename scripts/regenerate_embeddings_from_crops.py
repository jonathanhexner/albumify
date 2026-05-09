"""
Regenerate embeddings from existing face crops.

Use this when:
- Embeddings are corrupted or mismatched
- Face crops exist but embeddings are wrong
- Need to verify embedding-to-face correspondence

This script:
1. Reads existing aligned face crops (face_XXXX_aligned.jpg)
2. Extracts fresh embeddings from each crop
3. Saves embeddings with metadata mapping face_id to source

Usage:
    # Regenerate from existing crops
    python scripts/regenerate_embeddings_from_crops.py \
        --face-crops results/Google_Germany/face_crops \
        --output results/Google_Germany \
        --metadata results/Google_Germany/benchmark_*.json

    # Then re-export clustering with fresh embeddings
    python scripts/export_clustering_data.py \
        --embeddings results/Google_Germany/embeddings_FRESH_*.npy \
        --output results/Google_Germany/clustering_export_fresh
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import sys
import re

import numpy as np
from PIL import Image
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import InsightFaceEmbedder

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def find_face_crops(crops_dir: Path) -> List[tuple[int, Path]]:
    """
    Find all aligned face crops and extract face_id from filename.

    Returns:
        List of (face_id, path) tuples sorted by face_id
    """
    face_pattern = re.compile(r'face_(\d+)_aligned\.jpg')

    faces = []
    for crop_file in crops_dir.glob('face_*_aligned.jpg'):
        match = face_pattern.match(crop_file.name)
        if match:
            face_id = int(match.group(1))
            faces.append((face_id, crop_file))

    # Sort by face_id to ensure correct order
    faces.sort(key=lambda x: x[0])

    logger.info(f"Found {len(faces)} face crops")
    if faces:
        logger.info(f"Face ID range: {faces[0][0]} to {faces[-1][0]}")

    return faces


def load_existing_metadata(metadata_path: Optional[Path]) -> Dict[int, Dict]:
    """
    Load existing metadata JSON to map face_id to image_path.

    Returns:
        Dict mapping face_id to metadata dict
    """
    if metadata_path is None or not metadata_path.exists():
        logger.warning("No metadata file provided - will not have image_path mapping")
        return {}

    with open(metadata_path) as f:
        data = json.load(f)

    # Build face_id -> metadata mapping
    # Handle both old format (list with face_index) and new format
    face_metadata = data.get('face_metadata', [])

    mapping = {}
    for idx, meta in enumerate(face_metadata):
        # Try to get face_id from metadata, fallback to index
        face_id = meta.get('face_index', idx)
        mapping[face_id] = meta

    logger.info(f"Loaded metadata for {len(mapping)} faces from {metadata_path.name}")
    return mapping


def extract_embeddings_from_crops(
    face_crops: List[tuple[int, Path]],
    embedder: InsightFaceEmbedder
) -> tuple[List[int], np.ndarray]:
    """
    Extract embeddings from face crop images using InsightFaceEmbedder.

    Args:
        face_crops: List of (face_id, path) tuples
        embedder: InsightFace embedder instance (from face_cluster module)

    Returns:
        (face_ids, embeddings) where embeddings is (N, 512) normalized array
    """
    logger.info("Extracting embeddings from face crops...")
    logger.info("This will take a while for large datasets...")

    face_ids = []
    embeddings = []
    failed = []

    for i, (face_id, crop_path) in enumerate(face_crops):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing {i+1}/{len(face_crops)}...")

        try:
            # Load image
            img = cv2.imread(str(crop_path))
            if img is None:
                logger.warning(f"Failed to load {crop_path}")
                failed.append(face_id)
                continue

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract embedding using shared InsightFaceEmbedder
            # This method handles normalization internally
            embedding = embedder.get_embedding(img_rgb)

            if embedding is None:
                logger.warning(f"Failed to extract embedding for face {face_id}")
                failed.append(face_id)
                continue

            # Ensure embedding is 1D (flatten if needed)
            embedding = np.asarray(embedding).flatten()

            face_ids.append(face_id)
            embeddings.append(embedding)

        except Exception as e:
            logger.error(f"Error processing face {face_id}: {e}")
            failed.append(face_id)
            continue

    if failed:
        logger.warning(f"Failed to process {len(failed)} faces: {failed[:10]}...")

    embeddings_array = np.array(embeddings)
    logger.info(f"Successfully extracted {len(embeddings_array)} embeddings")
    logger.info(f"Embedding shape: {embeddings_array.shape}")

    return face_ids, embeddings_array


def save_embeddings_with_metadata(
    face_ids: List[int],
    embeddings: np.ndarray,
    existing_metadata: Dict[int, Dict],
    output_dir: Path
) -> tuple[Path, Path]:
    """
    Save embeddings and metadata with face_id mapping.

    Returns:
        (embeddings_path, metadata_path)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save embeddings as numpy array
    embeddings_path = output_dir / f'embeddings_FRESH_{timestamp}.npy'
    np.save(embeddings_path, embeddings)
    logger.info(f"✓ Saved embeddings: {embeddings_path}")

    # Build metadata
    face_metadata = []
    for face_id in face_ids:
        # Get existing metadata if available
        existing = existing_metadata.get(face_id, {})

        meta = {
            'face_id': face_id,
            'face_index': face_id,  # For compatibility
            'image_path': existing.get('image_path', None),
            'bbox': existing.get('bbox', None),
            'blur_score': existing.get('blur_score', None),
            'pose': existing.get('pose', None),
        }
        face_metadata.append(meta)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'regenerated_from': 'face_crops',
        'n_faces': len(face_ids),
        'face_ids': face_ids,  # Explicit face_id order
        'face_metadata': face_metadata,
    }

    metadata_path = output_dir / f'embeddings_metadata_FRESH_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata: {metadata_path}")

    return embeddings_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate embeddings from existing face crops',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--face-crops',
        type=Path,
        required=True,
        help='Directory containing face_XXXX_aligned.jpg files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for fresh embeddings and metadata'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        help='Optional: existing benchmark_*.json with face metadata (for image_path mapping)'
    )
    parser.add_argument(
        '--detector',
        type=str,
        default='buffalo_l',
        help='InsightFace detector model (default: buffalo_l)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device (default: cpu)'
    )

    args = parser.parse_args()
    setup_logging()

    logger.info("="*60)
    logger.info("Regenerate Embeddings from Face Crops")
    logger.info("="*60)
    logger.info(f"Face crops: {args.face_crops}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Metadata: {args.metadata if args.metadata else 'None (no image_path mapping)'}")

    # Validate inputs
    if not args.face_crops.exists():
        logger.error(f"Face crops directory not found: {args.face_crops}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    try:
        # Find face crops
        face_crops = find_face_crops(args.face_crops)
        if not face_crops:
            logger.error("No face crops found!")
            sys.exit(1)

        # Load existing metadata if available
        existing_metadata = load_existing_metadata(args.metadata)

        # Initialize embedder
        logger.info(f"Initializing InsightFace ({args.detector})...")
        ctx_id = 0 if args.device == 'cuda' else -1
        embedder = InsightFaceEmbedder(model_name=args.detector, ctx_id=ctx_id)

        # Extract embeddings
        face_ids, embeddings = extract_embeddings_from_crops(face_crops, embedder)

        if len(embeddings) == 0:
            logger.error("No embeddings extracted!")
            sys.exit(1)

        # Save with metadata
        embeddings_path, metadata_path = save_embeddings_with_metadata(
            face_ids,
            embeddings,
            existing_metadata,
            args.output
        )

        # Summary
        logger.info("\n" + "="*60)
        logger.info("✓ Regeneration Complete!")
        logger.info("="*60)
        logger.info(f"Face crops processed: {len(face_ids)}")
        logger.info(f"Embeddings extracted: {len(embeddings)}")
        logger.info(f"\nOutput files:")
        logger.info(f"  - {embeddings_path.name}")
        logger.info(f"  - {metadata_path.name}")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Delete old embeddings to avoid confusion:")
        logger.info(f"     rm {args.output}/embeddings_2026-*.npy")
        logger.info(f"\n  2. Re-export clustering with FRESH embeddings:")
        logger.info(f"     python scripts/export_clustering_data.py \\")
        logger.info(f"       --embeddings {embeddings_path} \\")
        logger.info(f"       --output results/clustering_export_fresh")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Regeneration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
