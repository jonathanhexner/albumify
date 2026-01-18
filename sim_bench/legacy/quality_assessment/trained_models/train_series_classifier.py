"""
Training script for PhotoTriage series classifier.

This trains on full series instead of pairs, using series-softmax loss.

Usage:
    # Independent scoring (MLP on each image)
    python sim_bench/quality_assessment/trained_models/train_series_classifier.py

    # With Transformer (images attend to each other)
    python sim_bench/quality_assessment/trained_models/train_series_classifier.py --use_transformer

    # Linear head baseline
    python sim_bench/quality_assessment/trained_models/train_series_classifier.py --mlp_hidden_dims []
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sim_bench.quality_assessment.trained_models.phototriage_series import (
    SeriesClassifierConfig,
    PhotoTriageSeriesClassifier,
    PhotoTriageSeriesDataset,
    SeriesClassifierTrainer,
    series_collate_fn
)
from sim_bench.quality_assessment.trained_models.train_binary_classifier import (
    load_and_filter_data,
    normalize_filename
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_series_data_from_csv(
    df: pd.DataFrame,
    split_name: str = "full"
) -> List[Dict]:
    """
    Build series-level data from pairwise CSV.

    For each series, we need to:
    1. Find all unique images in that series
    2. Determine which image is "best" based on pairwise votes

    Args:
        df: DataFrame with pairwise comparisons (filtered by agreement)
        split_name: Name of this split (for logging)

    Returns:
        List of series dicts with keys:
            - 'series_id': str
            - 'images': List[str] (filenames)
            - 'best_idx': int (index of best image, 0-indexed)
    """
    logger.info(f"\nBuilding series data for {split_name} split...")

    series_dict = defaultdict(lambda: {
        'images': set(),
        'wins': defaultdict(int)  # image -> win count
    })

    # Collect all images and win counts per series
    for _, row in df.iterrows():
        series_id = str(row['series_id'])
        img1 = row['compareFile1']
        img2 = row['compareFile2']
        winner_is_2 = row['label']  # 1 if img2 wins, 0 if img1 wins

        # Add images to series
        series_dict[series_id]['images'].add(img1)
        series_dict[series_id]['images'].add(img2)

        # Count wins
        if winner_is_2:
            series_dict[series_id]['wins'][img2] += 1
        else:
            series_dict[series_id]['wins'][img1] += 1

    # Convert to list format
    series_data = []

    for series_id, data in series_dict.items():
        images = sorted(list(data['images']))  # Sort for consistency
        wins = data['wins']

        # Find image with most wins
        best_img = max(images, key=lambda img: wins.get(img, 0))
        best_idx = images.index(best_img)

        series_data.append({
            'series_id': series_id,
            'images': images,
            'best_idx': best_idx
        })

    logger.info(f"{split_name} split:")
    logger.info(f"  Total series: {len(series_data)}")

    # Stats
    series_lengths = [len(s['images']) for s in series_data]
    logger.info(f"  Images per series: min={min(series_lengths)}, max={max(series_lengths)}, avg={np.mean(series_lengths):.2f}")

    return series_data


def split_by_series(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by series_id to prevent leakage.

    Args:
        df: DataFrame with series_id column
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        random_seed: Random seed

    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("Splitting data by series_id...")

    # Get unique series
    unique_series = df['series_id'].unique()
    n_series = len(unique_series)
    logger.info(f"Total series: {n_series}")

    # Shuffle series
    rng = np.random.RandomState(random_seed)
    rng.shuffle(unique_series)

    # Split series
    n_train = int(train_ratio * n_series)
    n_val = int(val_ratio * n_series)

    train_series = unique_series[:n_train]
    val_series = unique_series[n_train:n_train + n_val]
    test_series = unique_series[n_train + n_val:]

    logger.info(f"Train series: {len(train_series)} ({len(train_series)/n_series:.1%})")
    logger.info(f"Val series: {len(val_series)} ({len(val_series)/n_series:.1%})")
    logger.info(f"Test series: {len(test_series)} ({len(test_series)/n_series:.1%})")

    # Filter pairs by series
    train_df = df[df['series_id'].isin(train_series)].copy()
    val_df = df[df['series_id'].isin(val_series)].copy()
    test_df = df[df['series_id'].isin(test_series)].copy()

    logger.info(f"Train pairs: {len(train_df)} ({len(train_df)/len(df):.1%})")
    logger.info(f"Val pairs: {len(val_df)} ({len(val_df)/len(df):.1%})")
    logger.info(f"Test pairs: {len(test_df)} ({len(test_df)/len(df):.1%})")

    return train_df, val_df, test_df


def load_embeddings(cache_path: str, image_dir: str, config: 'SeriesClassifierConfig') -> Dict[str, torch.Tensor]:
    """
    Load pre-computed embeddings from cache, or compute them if cache doesn't exist.

    Args:
        cache_path: Path to embeddings pickle file
        image_dir: Directory containing images (for computing if needed)
        config: Model configuration (for computing if needed)

    Returns:
        Dict mapping filename -> embedding tensor
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings for {len(embeddings)} images")
    else:
        logger.warning(f"Embeddings cache not found at {cache_path}")
        logger.info("Computing embeddings now (this will take ~30-60 min on CPU)...")

        # Import here to avoid circular dependency
        from sim_bench.quality_assessment.trained_models.train_binary_classifier import precompute_embeddings

        embeddings = precompute_embeddings(image_dir, config, str(cache_path))

    return embeddings


def evaluate_series_model(
    model: PhotoTriageSeriesClassifier,
    test_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate series model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dict with metrics
    """
    logger.info("Evaluating on test set...")

    model.eval()

    all_preds = []
    all_labels = []
    all_series_ids = []
    all_ranks = []  # Rank of true best image (1 = top rank)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            embeddings = batch['embeddings'].to(device)
            best_idx = batch['best_idx'].to(device)
            masks = batch['masks'].to(device)

            # Predict
            preds, probs = model.predict(embeddings, masks)

            all_preds.append(preds.cpu())
            all_labels.append(best_idx.cpu())
            all_series_ids.extend(batch['series_ids'])

            # Calculate rank of true best image
            # Higher score = better, so argsort descending
            for i in range(embeddings.shape[0]):
                # Get scores for this series
                scores = model(embeddings[i:i+1], masks[i:i+1])
                scores = scores[0]  # (num_images,)

                # Mask out padding
                mask = masks[i]
                scores_valid = scores[mask]

                # Argsort descending (best first)
                sorted_indices = torch.argsort(scores_valid, descending=True)

                # Find rank of true best image
                true_best = best_idx[i].item()
                rank = (sorted_indices == true_best).nonzero(as_tuple=True)[0].item() + 1
                all_ranks.append(rank)

    # Concatenate
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_ranks = np.array(all_ranks)

    # Compute metrics
    top1_accuracy = (all_preds == all_labels).float().mean().item()
    top2_accuracy = (all_ranks <= 2).mean()
    top3_accuracy = (all_ranks <= 3).mean()
    mean_rank = all_ranks.mean()
    mrr = (1.0 / all_ranks).mean()  # Mean Reciprocal Rank

    # Calculate random baseline
    # For each series, random guess would have 1/num_images chance
    # We approximate with average series length
    series_lengths = []
    for batch in test_loader:
        for mask in batch['masks']:
            length = mask.sum().item()
            series_lengths.append(length)
    avg_series_length = np.mean(series_lengths)
    random_baseline = 1.0 / avg_series_length

    metrics = {
        'top1_accuracy': top1_accuracy,
        'top2_accuracy': top2_accuracy,
        'top3_accuracy': top3_accuracy,
        'mean_rank': mean_rank,
        'mrr': mrr,
        'random_baseline': random_baseline,
        'avg_series_length': avg_series_length,
        'num_series': len(all_preds)
    }

    # Log metrics
    logger.info("\n" + "="*60)
    logger.info("TEST SET RESULTS (Series Ranking)")
    logger.info("="*60)
    logger.info(f"Number of series: {metrics['num_series']}")
    logger.info(f"Average series length: {avg_series_length:.2f} images")
    logger.info(f"\nRandom Baseline: {100*random_baseline:.2f}%")
    logger.info(f"Top-1 Accuracy: {100*top1_accuracy:.2f}%")
    logger.info(f"  Improvement over random: {100*(top1_accuracy - random_baseline):.2f} pp")
    logger.info(f"\nTop-2 Accuracy: {100*top2_accuracy:.2f}%")
    logger.info(f"Top-3 Accuracy: {100*top3_accuracy:.2f}%")
    logger.info(f"Mean Rank: {mean_rank:.2f}")
    logger.info(f"Mean Reciprocal Rank: {mrr:.4f}")
    logger.info("="*60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PhotoTriage series classifier")
    parser.add_argument(
        '--csv_path',
        type=str,
        default=r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv",
        help="Path to CSV file with pairs"
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default=r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs",
        help="Directory containing images"
    )
    parser.add_argument(
        '--embeddings_cache',
        type=str,
        default="outputs/phototriage_binary/embeddings_cache.pkl",
        help="Path to pre-computed embeddings cache"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="outputs/phototriage_series",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=30,
        help="Maximum training epochs"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        '--mlp_hidden_dims',
        type=int,
        nargs='*',
        default=[256, 128],
        help="MLP hidden dimensions. Empty list [] = linear only"
    )
    parser.add_argument(
        '--use_transformer',
        action='store_true',
        help="Use Transformer instead of independent MLP"
    )

    args = parser.parse_args()

    # Create config
    config = SeriesClassifierConfig(
        csv_path=args.csv_path,
        image_dir=args.images_dir,
        embedding_cache_path=args.embeddings_cache,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        mlp_hidden_dims=args.mlp_hidden_dims,
        use_transformer=args.use_transformer
    )

    logger.info("="*60)
    logger.info("PhotoTriage Series Classifier Training")
    logger.info("="*60)
    logger.info(f"CSV: {config.csv_path}")
    logger.info(f"Embeddings Cache: {config.embedding_cache_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Architecture: {'Transformer' if config.use_transformer else 'Independent MLP'}")
    logger.info(f"MLP Hidden Dims: {config.mlp_hidden_dims}")
    logger.info("="*60)

    # Step 1: Load and filter data
    df = load_and_filter_data(config.csv_path, config.min_agreement, config.min_reviewers)

    # Step 2: Split by series
    train_df, val_df, test_df = split_by_series(
        df,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed
    )

    # Step 3: Build series-level data
    train_series = build_series_data_from_csv(train_df, "train")
    val_series = build_series_data_from_csv(val_df, "val")
    test_series = build_series_data_from_csv(test_df, "test")

    # Save series data
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_series.json", 'w') as f:
        json.dump(train_series, f, indent=2)
    with open(output_dir / "val_series.json", 'w') as f:
        json.dump(val_series, f, indent=2)
    with open(output_dir / "test_series.json", 'w') as f:
        json.dump(test_series, f, indent=2)
    logger.info(f"Series data saved to {output_dir}")

    # Step 4: Load embeddings (will compute if cache doesn't exist)
    embeddings = load_embeddings(config.embedding_cache_path, config.image_dir, config)

    # Step 5: Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = PhotoTriageSeriesDataset(train_series, embeddings)
    val_dataset = PhotoTriageSeriesDataset(val_series, embeddings)
    test_dataset = PhotoTriageSeriesDataset(test_series, embeddings)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=series_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=series_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=series_collate_fn
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Step 6: Create model
    logger.info("\nCreating model...")
    model = PhotoTriageSeriesClassifier(config)

    # Step 7: Train
    logger.info("\nStarting training...")
    trainer = SeriesClassifierTrainer(model, train_loader, val_loader, config)
    trainer.train()

    # Step 8: Load best model and evaluate on test set
    logger.info("\nLoading best model for final evaluation...")
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        model = PhotoTriageSeriesClassifier.load(str(best_model_path), device=config.device)
    else:
        logger.warning("Best model not found, using final model")

    test_metrics = evaluate_series_model(model, test_loader, config.device)

    # Save test results
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test results saved to {results_path}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
