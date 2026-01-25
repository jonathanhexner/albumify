"""
Extract landmarks from AffectNet dataset using MediaPipe.

Pre-extracts and caches landmarks for efficient training.

Usage:
    python -m sim_bench.training.phase2_pretraining.extract_landmarks \
        --data_dir D:/DataSets/AffectNet/train \
        --output_cache landmarks_train.json \
        --num_landmarks 5
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from sim_bench.training.phase2_pretraining.landmark_extractor import LandmarkExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_all_images(data_dir: Path) -> list:
    """Find all images in AffectNet directory structure."""
    images = []
    for expr_dir in sorted(data_dir.iterdir()):
        if not expr_dir.is_dir():
            continue
        for img_file in expr_dir.glob("*.jpg"):
            images.append(img_file)
    return images


def main():
    parser = argparse.ArgumentParser(description='Extract landmarks from AffectNet')
    parser.add_argument('--data_dir', type=str, required=True, help='AffectNet data directory')
    parser.add_argument('--output_cache', type=str, required=True, help='Output cache JSON path')
    parser.add_argument('--num_landmarks', type=int, default=5, choices=[5, 10], help='Number of landmarks')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    logger.info(f"Scanning {data_dir} for images...")
    image_paths = find_all_images(data_dir)
    logger.info(f"Found {len(image_paths)} images")

    extractor = LandmarkExtractor(num_landmarks=args.num_landmarks)
    
    logger.info("Extracting landmarks...")
    results = extractor.extract_batch(image_paths, cache_dir=None)

    valid_count = sum(1 for v in results.values() if v is not None)
    logger.info(f"Extracted landmarks for {valid_count}/{len(image_paths)} images ({100*valid_count/len(image_paths):.1f}%)")

    output_path = Path(args.output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable = {}
    for path, landmarks in results.items():
        if landmarks is not None:
            serializable[str(path)] = landmarks.tolist()
        else:
            serializable[str(path)] = None

    import json
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    logger.info(f"Saved landmarks cache to {output_path}")


if __name__ == '__main__':
    main()
