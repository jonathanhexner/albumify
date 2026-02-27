"""
Prepare face embeddings from image directory.

Detects faces, extracts embeddings, and saves in format compatible with
export_clustering_data.py script.

This script is meant to be run ONCE per dataset. Then use export_clustering_data.py
multiple times with different clustering parameters.

Usage:
    python scripts/prepare_embeddings.py \
        --images D:/my_photos \
        --output results/my_photos \
        --detector scrfd \
        --quality-filter
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

import numpy as np
from PIL import Image
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import InsightFaceEmbedder, FaceRecord

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    """Configure logging to both console and file."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f'prepare_embeddings_{timestamp}.log'

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")
    return log_file


def find_images(images_dir: Path) -> List[Path]:
    """Find all image files in directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []

    for ext in extensions:
        image_paths.extend(images_dir.glob(f"**/{ext}"))

    return sorted(set(image_paths))  # Remove duplicates, sort


def save_face_crops(faces: List[FaceRecord], output_dir: Path):
    """Save aligned face crops to disk."""
    crops_dir = output_dir / 'face_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for face in faces:
        if face.aligned_face is not None:
            crop_path = crops_dir / f"face_{face.face_id:04d}_aligned.jpg"
            img = Image.fromarray(face.aligned_face)
            img.save(crop_path, quality=95)
            saved_count += 1

    logger.info(f"Saved {saved_count} face crops to: {crops_dir}")
    return crops_dir


def save_embeddings_and_metadata(
    faces: List[FaceRecord],
    output_dir: Path,
    source_dir: Path
) -> tuple[Path, Path]:
    """
    Save embeddings and metadata in format compatible with export_clustering_data.py.

    Args:
        faces: List of FaceRecord objects
        output_dir: Output directory
        source_dir: Source image directory (for relative paths)

    Returns:
        (embeddings_path, metadata_path)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save embeddings as numpy array
    embeddings = np.array([f.embedding for f in faces])
    embeddings_path = output_dir / f'embeddings_{timestamp}.npy'
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved {len(embeddings)} embeddings to: {embeddings_path}")

    # Save metadata as JSON
    face_metadata = []
    for face in faces:
        # Try to compute relative path, fallback to absolute
        try:
            rel_path = Path(face.image_path).relative_to(source_dir)
            image_path_str = str(rel_path)
        except (ValueError, TypeError):
            image_path_str = face.image_path

        yaw, pitch, roll = face.pose if face.pose else (None, None, None)

        face_metadata.append({
            'face_index': face.face_index,
            'image_path': image_path_str,
            'bbox': {
                'x_px': float(face.bbox[0]),
                'y_px': float(face.bbox[1]),
                'w_px': float(face.bbox[2] - face.bbox[0]),  # Convert x2 to width
                'h_px': float(face.bbox[3] - face.bbox[1]),  # Convert y2 to height
            },
            'blur_score': float(face.blur_score) if face.blur_score else None,
            'pose': {
                'yaw': float(yaw) if yaw is not None else None,
                'pitch': float(pitch) if pitch is not None else None,
                'roll': float(roll) if roll is not None else None,
            } if face.pose else None,
        })

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source_directory': str(source_dir),
        'n_images': len(set(f.image_path for f in faces)),
        'n_faces': len(faces),
        'face_metadata': face_metadata,
    }

    metadata_path = output_dir / f'benchmark_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")

    return embeddings_path, metadata_path


def compute_blur_scores(faces: List[FaceRecord]) -> List[FaceRecord]:
    """Compute blur scores for face crops using Laplacian variance."""
    logger.info("Computing blur scores...")

    for face in faces:
        if face.aligned_face is not None:
            # Convert to grayscale
            if len(face.aligned_face.shape) == 3:
                gray = cv2.cvtColor(face.aligned_face, cv2.COLOR_RGB2GRAY)
            else:
                gray = face.aligned_face

            # Compute Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var()
            face.blur_score = float(blur_score)
        else:
            face.blur_score = 0.0

    logger.info(f"Computed blur scores (range: {min(f.blur_score for f in faces):.1f} - {max(f.blur_score for f in faces):.1f})")
    return faces


def main():
    parser = argparse.ArgumentParser(
        description='Prepare face embeddings from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--images',
        type=Path,
        required=True,
        help='Directory containing images (searches recursively)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for embeddings, metadata, and face crops'
    )

    # Detection options
    parser.add_argument(
        '--detector',
        type=str,
        default='buffalo_l',
        choices=['buffalo_l', 'buffalo_s', 'buffalo_m'],
        help='InsightFace detector model (default: buffalo_l)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for face detection (default: cpu)'
    )

    # Quality filtering
    parser.add_argument(
        '--min-face-size',
        type=int,
        default=40,
        help='Minimum face size in pixels (default: 40)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence (default: 0.5)'
    )

    args = parser.parse_args()

    # Setup logging
    args.output.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output)

    logger.info("="*60)
    logger.info("Face Embedding Preparation")
    logger.info("="*60)
    logger.info(f"Images directory: {args.images}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Detector: {args.detector}")

    # Validate input
    if not args.images.exists():
        logger.error(f"Images directory not found: {args.images}")
        sys.exit(1)

    try:
        # Find images
        logger.info("Finding images...")
        image_paths = find_images(args.images)
        logger.info(f"Found {len(image_paths)} images")

        if len(image_paths) == 0:
            logger.error(f"No images found in {args.images}")
            sys.exit(1)

        # Initialize embedder
        logger.info(f"Initializing InsightFace ({args.detector})...")
        ctx_id = 0 if args.device == 'cuda' else -1
        embedder = InsightFaceEmbedder(model_name=args.detector, ctx_id=ctx_id)

        # Detect faces and extract embeddings
        logger.info("Detecting faces and extracting embeddings...")
        logger.info("This may take a while depending on the number of images...")

        faces = embedder.detect_and_embed(
            [str(p) for p in image_paths],
            extract_pose=True
        )

        logger.info(f"Detected {len(faces)} faces from {len(image_paths)} images")

        if len(faces) == 0:
            logger.error("No faces detected! Check your images.")
            sys.exit(1)

        # Filter by size and confidence
        faces_before = len(faces)
        faces = [
            f for f in faces
            if f.area >= (args.min_face_size ** 2)
        ]
        if len(faces) < faces_before:
            logger.info(f"Filtered {faces_before - len(faces)} faces by size (min={args.min_face_size}px)")

        # Compute blur scores
        faces = compute_blur_scores(faces)

        # Save face crops
        crops_dir = save_face_crops(faces, args.output)

        # Save embeddings and metadata
        embeddings_path, metadata_path = save_embeddings_and_metadata(
            faces,
            args.output,
            args.images
        )

        # Summary
        logger.info("\n" + "="*60)
        logger.info("Preparation Complete!")
        logger.info("="*60)
        logger.info(f"Images processed: {len(image_paths)}")
        logger.info(f"Faces detected: {len(faces)}")
        logger.info(f"\nOutput files:")
        logger.info(f"  - {embeddings_path.name}")
        logger.info(f"  - {metadata_path.name}")
        logger.info(f"  - face_crops/ ({len(faces)} images)")
        logger.info(f"\nNext step:")
        logger.info(f"  python scripts/export_clustering_data.py \\")
        logger.info(f"    --embeddings {embeddings_path} \\")
        logger.info(f"    --output results/face_clustering_training/my_dataset")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
