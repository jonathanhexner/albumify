"""
Simple test to isolate the training error.
"""

import sys
from pathlib import Path
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
    PhotoTriagePairDataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Simple training test")

    # Config
    config = BinaryClassifierConfig(
        csv_path=r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv",
        image_dir=r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs",
        output_dir="outputs/test_simple",
        batch_size=8,  # Very small batch
        max_epochs=1,  # Just 1 epoch
        use_cache=True,
        log_interval=1
    )

    # Load data
    logger.info("Loading data...")
    df = load_and_filter_data(config.csv_path, config.min_agreement, config.min_reviewers)
    df_sample = df.head(50).copy()  # Just 50 pairs

    # Split
    logger.info("Splitting...")
    train_df, val_df, test_df = split_by_series(
        df_sample, config.train_ratio, config.val_ratio, config.test_ratio, config.random_seed
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load embeddings from cache
    logger.info("Loading embeddings from cache...")
    import pickle
    cache_path = Path("outputs/test_pipeline/test_embeddings_cache.pkl")
    with open(cache_path, 'rb') as f:
        embeddings = pickle.load(f)
    logger.info(f"Loaded {len(embeddings)} embeddings")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = PhotoTriagePairDataset(train_df, embeddings)
    val_dataset = PhotoTriagePairDataset(val_df, embeddings)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Test one batch
    logger.info("\nTesting one batch from train_loader...")
    for i, batch in enumerate(train_loader):
        logger.info(f"Batch {i}:")
        logger.info(f"  emb1 shape: {batch['emb1'].shape}")
        logger.info(f"  emb2 shape: {batch['emb2'].shape}")
        logger.info(f"  label shape: {batch['label'].shape}")
        logger.info(f"  label dtype: {batch['label'].dtype}")
        logger.info(f"  label values: {batch['label']}")
        if i >= 0:  # Just first batch
            break

    # Create model
    logger.info("\nCreating model...")
    model = PhotoTriageBinaryClassifier(config)

    # Test forward pass
    logger.info("\nTesting forward pass...")
    with torch.no_grad():
        logits = model(emb1=batch['emb1'], emb2=batch['emb2'])
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Logits: {logits}")

    # Test training step
    logger.info("\nTesting one training step...")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    emb1 = batch['emb1'].to(config.device)
    emb2 = batch['emb2'].to(config.device)
    labels = batch['label'].to(config.device)

    logger.info(f"Input shapes - emb1: {emb1.shape}, emb2: {emb2.shape}, labels: {labels.shape}")
    logger.info(f"Label dtype: {labels.dtype}, values: {labels}")

    logits = model(emb1=emb1, emb2=emb2)
    logger.info(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")

    loss = criterion(logits, labels)
    logger.info(f"Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info("✓ Training step successful!")

    logger.info("\nAll tests passed!")

if __name__ == "__main__":
    main()
