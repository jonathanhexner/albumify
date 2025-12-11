"""
Quick script to pre-compute CLIP embeddings for PhotoTriage dataset.

This creates the embeddings cache that can be reused by all training scripts.

Usage:
    python precompute_embeddings.py

This will:
1. Load all images from PhotoTriage
2. Compute CLIP embeddings (takes ~30-60 min on CPU)
3. Save to outputs/phototriage_binary/embeddings_cache.pkl
4. Exit (no training)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
from sim_bench.quality_assessment.trained_models.phototriage_binary import BinaryClassifierConfig
from sim_bench.quality_assessment.trained_models.train_binary_classifier import precompute_embeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Pre-computing CLIP Embeddings for PhotoTriage")
    logger.info("=" * 60)

    # Create config
    config = BinaryClassifierConfig(
        csv_path=r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv",
        image_dir=r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs",
        output_dir="outputs/phototriage_binary",
        use_cache=True
    )

    cache_path = Path(config.output_dir) / "embeddings_cache.pkl"

    logger.info(f"Image directory: {config.image_dir}")
    logger.info(f"Cache output: {cache_path}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 60)

    # Check if cache already exists
    if cache_path.exists():
        logger.warning(f"\nCache already exists at {cache_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting without overwriting cache")
            return

    # Pre-compute embeddings
    logger.info("\nStarting embedding computation...")
    logger.info("This will take approximately 30-60 minutes on CPU")
    logger.info("Progress bar will show status\n")

    embeddings = precompute_embeddings(
        config.image_dir,
        config,
        str(cache_path)
    )

    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDINGS SAVED!")
    logger.info("=" * 60)
    logger.info(f"Total images: {len(embeddings)}")
    logger.info(f"Cache location: {cache_path}")
    logger.info(f"Cache size: {cache_path.stat().st_size / (1024**2):.1f} MB")
    logger.info("\nYou can now run any training script:")
    logger.info("  - train_binary_classifier.py (pairwise)")
    logger.info("  - train_series_classifier.py (series-softmax)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
