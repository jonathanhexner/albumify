"""
Quick test script to verify training pipeline works.

This runs on a very small subset to check:
1. Data loading works
2. Embeddings computation works
3. Model creation works
4. Training loop runs without errors
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sim_bench.quality_assessment.trained_models.phototriage_binary import (
    BinaryClassifierConfig,
    PhotoTriageBinaryClassifier,
    BinaryClassifierTrainer
)
from sim_bench.quality_assessment.trained_models.train_binary_classifier import (
    load_and_filter_data,
    split_by_series,
    precompute_embeddings,
    PhotoTriagePairDataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("TESTING TRAINING PIPELINE")
    logger.info("="*60)

    # Config for quick test
    config = BinaryClassifierConfig(
        csv_path=r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv",
        image_dir=r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs",
        output_dir="outputs/test_pipeline",
        batch_size=16,
        max_epochs=2,  # Just 2 epochs for testing
        learning_rate=1e-3,
        use_cache=True,
        log_interval=5
    )

    # Step 1: Load data
    logger.info("\n[1/7] Loading data...")
    df = load_and_filter_data(config.csv_path, config.min_agreement, config.min_reviewers)

    # Use only first 100 pairs for testing
    logger.info("Sampling 100 pairs for quick test...")
    df_sample = df.head(100).copy()

    # Step 2: Split
    logger.info("\n[2/7] Splitting data...")
    train_df, val_df, test_df = split_by_series(
        df_sample,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed
    )

    # Step 3: Get unique images needed
    logger.info("\n[3/7] Finding unique images...")
    unique_files = set()
    for df_split in [train_df, val_df, test_df]:
        unique_files.update(df_split['compareFile1'].unique())
        unique_files.update(df_split['compareFile2'].unique())
    logger.info(f"Unique images needed: {len(unique_files)}")

    # Step 4: Pre-compute embeddings (just for these images)
    logger.info("\n[4/7] Pre-computing embeddings...")
    logger.info("NOTE: This may take 2-5 minutes on CPU...")

    cache_path = Path(config.output_dir) / "test_embeddings_cache.pkl"

    # If cache exists, load it
    if cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        import pickle
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        # Compute embeddings for all images (we'll cache them)
        embeddings = precompute_embeddings(config.image_dir, config, str(cache_path))

    logger.info(f"Embeddings loaded: {len(embeddings)}")

    # Step 5: Create datasets
    logger.info("\n[5/7] Creating datasets...")
    train_dataset = PhotoTriagePairDataset(train_df, embeddings)
    val_dataset = PhotoTriagePairDataset(val_df, embeddings)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Step 6: Create model
    logger.info("\n[6/7] Creating model...")
    model = PhotoTriageBinaryClassifier(config)

    # Step 7: Train for 2 epochs
    logger.info("\n[7/7] Training for 2 epochs...")
    trainer = BinaryClassifierTrainer(model, train_loader, val_loader, config)
    trainer.train()

    logger.info("\n" + "="*60)
    logger.info("PIPELINE TEST COMPLETE!")
    logger.info("="*60)
    logger.info("✓ Data loading works")
    logger.info("✓ Embedding computation works")
    logger.info("✓ Model creation works")
    logger.info("✓ Training loop works")
    logger.info("\nYou can now run full training with:")
    logger.info("python sim_bench/quality_assessment/trained_models/train_binary_classifier.py")

if __name__ == "__main__":
    main()
