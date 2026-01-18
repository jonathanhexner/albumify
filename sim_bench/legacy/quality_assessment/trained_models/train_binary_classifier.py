"""
Training script for PhotoTriage binary classifier.

Usage:
    python sim_bench/quality_assessment/trained_models/train_binary_classifier.py

This script:
1. Loads the PhotoTriage CSV dataset
2. Filters pairs by agreement and reviewer count
3. Pre-computes CLIP embeddings for all images
4. Splits data by series_id to prevent leakage
5. Trains MLP classifier on embeddings
6. Evaluates on test set
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sim_bench.quality_assessment.trained_models.phototriage_binary import (
    BinaryClassifierConfig,
    PhotoTriageBinaryClassifier,
    BinaryClassifierTrainer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhotoTriagePairDataset(Dataset):
    """
    PyTorch dataset for PhotoTriage pairwise comparisons.

    Uses pre-computed embeddings for efficiency.
    """

    def __init__(self, pairs_df: pd.DataFrame, embeddings_dict: Dict[str, torch.Tensor]):
        """
        Args:
            pairs_df: DataFrame with columns [compareFile1, compareFile2, label]
            embeddings_dict: Dict mapping filename -> embedding tensor
        """
        self.pairs = pairs_df.reset_index(drop=True)
        self.embeddings = embeddings_dict

        # Verify all files have embeddings
        missing = []
        for idx, row in self.pairs.iterrows():
            if row['compareFile1'] not in embeddings_dict:
                missing.append(row['compareFile1'])
            if row['compareFile2'] not in embeddings_dict:
                missing.append(row['compareFile2'])

        if missing:
            logger.warning(f"Missing embeddings for {len(set(missing))} files")
            logger.warning(f"Examples: {missing[:5]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        emb1 = self.embeddings[row['compareFile1']]  # (embed_dim,)
        emb2 = self.embeddings[row['compareFile2']]  # (embed_dim,)
        label = row['label']  # 0 or 1

        return {
            'emb1': emb1,
            'emb2': emb2,
            'label': torch.tensor(label, dtype=torch.long)
        }


def normalize_filename(filename: str) -> str:
    """
    Convert CSV filename format to actual image filename format.

    CSV has: "1-1.JPG", "12-3.JPG"
    Images are: "000001-01.JPG", "000012-03.JPG"

    Args:
        filename: Filename from CSV (e.g., "1-1.JPG")

    Returns:
        Normalized filename (e.g., "000001-01.JPG")
    """
    # Split on '-' and '.'
    parts = filename.replace('.JPG', '').split('-')
    if len(parts) == 2:
        series_num, image_num = parts
        # Pad to 6 digits for series, 2 digits for image number
        normalized = f"{int(series_num):06d}-{int(image_num):02d}.JPG"
        return normalized
    else:
        # Return as-is if format doesn't match expected pattern
        return filename


def load_and_filter_data(csv_path: str, min_agreement: float, min_reviewers: int) -> pd.DataFrame:
    """
    Load CSV and filter by agreement and reviewer count.

    Args:
        csv_path: Path to CSV file
        min_agreement: Minimum agreement threshold
        min_reviewers: Minimum number of reviewers

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Total pairs: {len(df)}")

    # Filter
    filtered = df[
        (df['Agreement'] > min_agreement) &
        (df['num_reviewers'] >= min_reviewers)
    ].copy()

    logger.info(f"Filtered pairs (Agreement>{min_agreement}, num_reviewers>={min_reviewers}): {len(filtered)}")

    # Normalize filenames to match actual image files
    filtered['compareFile1'] = filtered['compareFile1'].apply(normalize_filename)
    filtered['compareFile2'] = filtered['compareFile2'].apply(normalize_filename)
    logger.info(f"Normalized filenames (e.g., '1-1.JPG' → '{filtered.iloc[0]['compareFile1']}')")

    # Create label: 1 if MaxVote == compareID2, else 0
    filtered['label'] = (filtered['MaxVote'] == filtered['compareID2']).astype(int)

    # Check label distribution
    label_counts = filtered['label'].value_counts()
    logger.info(f"Label distribution: {dict(label_counts)}")
    logger.info(f"Class balance: {label_counts[0]/len(filtered):.1%} / {label_counts[1]/len(filtered):.1%}")

    return filtered


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


def precompute_embeddings(
    image_dir: str,
    config: BinaryClassifierConfig,
    cache_path: str = None
) -> Dict[str, torch.Tensor]:
    """
    Pre-compute CLIP embeddings for all images.

    Args:
        image_dir: Directory containing images
        config: Model configuration
        cache_path: Path to save/load embeddings cache

    Returns:
        Dict mapping filename -> embedding tensor
    """
    # Check cache
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings for {len(embeddings)} images")
        return embeddings

    logger.info(f"Pre-computing CLIP embeddings from {image_dir}")

    # Load CLIP model
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model,
        pretrained=config.clip_checkpoint
    )
    model.eval()
    model.to(config.device)

    # Get all image files
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.JPG")) + sorted(image_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images")

    # Compute embeddings
    embeddings = {}

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Computing embeddings"):
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(config.device)

                # Encode
                embedding = model.encode_image(img_tensor)
                embedding = embedding.squeeze(0).cpu()

                # L2 normalize (CLIP is trained with normalized embeddings)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                # Store with filename as key
                filename = img_path.name
                embeddings[filename] = embedding

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

    logger.info(f"Computed embeddings for {len(embeddings)} images")

    # Save cache
    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Embeddings saved to cache: {cache_path}")

    return embeddings


def evaluate_test_set(
    model: PhotoTriageBinaryClassifier,
    test_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on test set.

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
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            emb1 = batch['emb1'].to(device)
            emb2 = batch['emb2'].to(device)
            labels = batch['label'].to(device)

            # Predict
            preds, probs = model.predict(emb1=emb1, emb2=emb2)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # Compute metrics
    accuracy = (all_preds == all_labels).float().mean().item()

    # Per-class accuracy
    class_0_mask = all_labels == 0
    class_1_mask = all_labels == 1

    class_0_acc = (all_preds[class_0_mask] == all_labels[class_0_mask]).float().mean().item() if class_0_mask.sum() > 0 else 0
    class_1_acc = (all_preds[class_1_mask] == all_labels[class_1_mask]).float().mean().item() if class_1_mask.sum() > 0 else 0

    # Confusion matrix
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    metrics = {
        'accuracy': accuracy,
        'class_0_accuracy': class_0_acc,
        'class_1_accuracy': class_1_acc,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
    }

    # F1 score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0

    # Log metrics
    logger.info("\n" + "="*60)
    logger.info("TEST SET RESULTS (Pairwise Comparison)")
    logger.info("="*60)
    logger.info(f"Random Baseline (pairwise): 50.0%")
    logger.info(f"Accuracy: {100*metrics['accuracy']:.2f}%")
    logger.info(f"  Improvement over random: {100*(metrics['accuracy'] - 0.5):.2f} pp")
    logger.info(f"\nClass 0 Accuracy: {100*metrics['class_0_accuracy']:.2f}%")
    logger.info(f"Class 1 Accuracy: {100*metrics['class_1_accuracy']:.2f}%")
    logger.info(f"Precision: {100*metrics['precision']:.2f}%")
    logger.info(f"Recall: {100*metrics['recall']:.2f}%")
    logger.info(f"F1 Score: {100*metrics['f1']:.2f}%")
    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Positives: {tp}")
    logger.info(f"  False Positives: {fp}")
    logger.info(f"  True Negatives: {tn}")
    logger.info(f"  False Negatives: {fn}")
    logger.info("="*60)
    logger.info("NOTE: For series ranking task, use series_train.py instead")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PhotoTriage binary classifier")
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
        '--output_dir',
        type=str,
        default="outputs/phototriage_binary",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help="Maximum training epochs"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help="Don't use embedding cache"
    )

    args = parser.parse_args()

    # Create config
    config = BinaryClassifierConfig(
        csv_path=args.csv_path,
        image_dir=args.images_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_cache=not args.no_cache
    )

    logger.info("="*60)
    logger.info("PhotoTriage Binary Classifier Training")
    logger.info("="*60)
    logger.info(f"CSV: {config.csv_path}")
    logger.info(f"Images: {config.image_dir}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Device: {config.device}")
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

    # Save splits
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train_pairs.csv", index=False)
    val_df.to_csv(output_dir / "val_pairs.csv", index=False)
    test_df.to_csv(output_dir / "test_pairs.csv", index=False)
    logger.info(f"Splits saved to {output_dir}")

    # Step 3: Pre-compute embeddings
    cache_path = output_dir / "embeddings_cache.pkl" if config.use_cache else None
    embeddings = precompute_embeddings(config.image_dir, config, cache_path)

    # Step 4: Create datasets
    logger.info("Creating datasets...")
    train_dataset = PhotoTriagePairDataset(train_df, embeddings)
    val_dataset = PhotoTriagePairDataset(val_df, embeddings)
    test_dataset = PhotoTriagePairDataset(test_df, embeddings)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Step 5: Create model
    logger.info("\nCreating model...")
    model = PhotoTriageBinaryClassifier(config)

    # Step 6: Train
    logger.info("\nStarting training...")
    trainer = BinaryClassifierTrainer(model, train_loader, val_loader, config)
    trainer.train()

    # Step 7: Load best model and evaluate on test set
    logger.info("\nLoading best model for final evaluation...")
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        model = PhotoTriageBinaryClassifier.load(str(best_model_path), device=config.device)
    else:
        logger.warning("Best model not found, using final model")

    test_metrics = evaluate_test_set(model, test_loader, config.device)

    # Save test results
    import json
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test results saved to {results_path}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
