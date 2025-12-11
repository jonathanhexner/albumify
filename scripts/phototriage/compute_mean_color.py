"""
Compute mean RGB color from PhotoTriage training set.

This computes the mean pixel color across all training images,
which is used for padding in the aspect-ratio preserving preprocessing
described in the PhotoTriage paper.

The PhotoTriage paper states:
"we resize each image so that its larger dimension is the required size,
while maintaining the original aspect ratio and padding with the mean
pixel color in the training set."

This script generates data/phototriage/training_mean_color.json which
contains the computed mean color values. This file is required for
paper-accurate preprocessing.

Usage:
    python scripts/phototriage/compute_mean_color.py

Output:
    - Prints mean RGB color in [0,1] normalized range (for config)
    - Prints mean RGB color in [0,255] range (for reference)
    - Saves to data/phototriage/training_mean_color.json
    - Compares with ImageNet mean

Typical runtime: 5-10 minutes (processes ~12K images)

Example output usage:
    python train_multifeature_ranker.py \\
        --use_paper_preprocessing true \\
        --padding_mean_color 0.460 0.450 0.430 \\
        --cnn_freeze_mode none
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_image_paths(pairs_file: str = "data/phototriage/pairs_train.jsonl") -> List[str]:
    """
    Load all unique image paths from training pairs file.

    Args:
        pairs_file: Path to training pairs JSONL file

    Returns:
        List of unique image paths in training set
    """
    image_paths = set()

    with open(pairs_file, 'r') as f:
        for line in f:
            pair = json.loads(line.strip())
            image_paths.add(pair['image_a_path'])
            image_paths.add(pair['image_b_path'])

    return sorted(list(image_paths))


def compute_mean_color(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean RGB color across all images.

    Args:
        image_paths: List of image file paths

    Returns:
        Tuple of:
        - Mean RGB in [0,1] range (for PyTorch normalization)
        - Mean RGB in [0,255] range (for reference)
    """
    logger.info(f"Computing mean color from {len(image_paths)} training images...")

    # Accumulate pixel sums and counts
    rgb_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Convert to numpy array
            img_array = np.array(img, dtype=np.float64)  # (H, W, 3)

            # Accumulate RGB sums
            rgb_sum += img_array.sum(axis=(0, 1))  # Sum across H and W dimensions
            pixel_count += img_array.shape[0] * img_array.shape[1]

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            continue

    # Compute mean
    mean_rgb_255 = rgb_sum / pixel_count  # [0, 255] range
    mean_rgb_01 = mean_rgb_255 / 255.0    # [0, 1] range

    return mean_rgb_01, mean_rgb_255


def main():
    """Compute and display mean RGB color from PhotoTriage training set."""

    # Load training image paths
    logger.info("Loading training image paths...")
    image_paths = load_training_image_paths()
    logger.info(f"Found {len(image_paths)} unique images in training set")

    # Compute mean color
    mean_01, mean_255 = compute_mean_color(image_paths)

    # Display results
    logger.info("\n" + "="*60)
    logger.info("PhotoTriage Training Set Mean Color")
    logger.info("="*60)
    logger.info(f"\nMean RGB [0,1] range (for config):")
    logger.info(f"  R: {mean_01[0]:.6f}")
    logger.info(f"  G: {mean_01[1]:.6f}")
    logger.info(f"  B: {mean_01[2]:.6f}")
    logger.info(f"\nAs list for --padding_mean_color:")
    logger.info(f"  {mean_01[0]:.6f} {mean_01[1]:.6f} {mean_01[2]:.6f}")

    logger.info(f"\nMean RGB [0,255] range (for reference):")
    logger.info(f"  R: {mean_255[0]:.1f}")
    logger.info(f"  G: {mean_255[1]:.1f}")
    logger.info(f"  B: {mean_255[2]:.1f}")

    logger.info(f"\nComparison with ImageNet mean [0,1]:")
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    logger.info(f"  ImageNet: R={imagenet_mean[0]:.3f}, G={imagenet_mean[1]:.3f}, B={imagenet_mean[2]:.3f}")
    logger.info(f"  PhotoTriage: R={mean_01[0]:.3f}, G={mean_01[1]:.3f}, B={mean_01[2]:.3f}")
    diff = mean_01 - imagenet_mean
    logger.info(f"  Difference: R={diff[0]:+.3f}, G={diff[1]:+.3f}, B={diff[2]:+.3f}")

    logger.info("="*60 + "\n")

    # Save to file for future reference
    output_file = "data/phototriage/training_mean_color.json"
    output_data = {
        "mean_rgb_normalized": mean_01.tolist(),
        "mean_rgb_255": mean_255.tolist(),
        "num_images": len(image_paths),
        "description": "Mean RGB color computed from PhotoTriage training set for aspect-ratio preserving preprocessing"
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved mean color to: {output_file}")


if __name__ == "__main__":
    main()
