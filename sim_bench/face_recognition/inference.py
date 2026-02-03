"""Apply trained ArcFace model and produce per-image predictions.

Generates CSV files with per-image predictions for train/val sets:
- sample_index, true_label, predicted_label, confidence, correct

Usage:
    python -m sim_bench.face_recognition.inference \\
        --checkpoint outputs/face/20260125_224553/best_model.pt \\
        --config configs/face/resnet50_arcface.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sim_bench.models.face_resnet import FaceResNet, create_transform
from sim_bench.datasets.face_dataset import MXNetRecordDataset
from sim_bench.face_recognition.utils import load_config, setup_logging
from sim_bench.face_recognition.train import create_identity_split, IdentitySubset

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str):
    """
    Load trained FaceResNet model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint
        device: Device to load model on

    Returns:
        model, config from checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    model_config = config['model']

    model = FaceResNet(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    lfw_acc = checkpoint.get('lfw_acc', 'unknown')
    logger.info(f"Loaded model from epoch {epoch}, LFW acc: {lfw_acc}")

    return model, config


def predict_dataset(model, loader, device, desc="Predicting"):
    """
    Run predictions on entire dataset.

    Args:
        model: Trained model
        loader: DataLoader
        device: Device
        desc: Progress bar description

    Returns:
        List of dicts with predictions per sample
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            indices = batch['index']

            # Forward pass (no labels = inference mode)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            # Get predictions
            confidence, predicted = probs.max(dim=1)

            # Store results
            for i in range(len(indices)):
                predictions.append({
                    'sample_index': int(indices[i]),
                    'true_label': int(labels[i].item()),
                    'predicted_label': int(predicted[i].item()),
                    'confidence': float(confidence[i].item()),
                    'correct': bool(predicted[i].item() == labels[i].item())
                })

    return predictions


def run_inference(checkpoint_path: str, config_path: str, output_dir: Path):
    """
    Run inference on train and val sets, save predictions to CSV.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to config YAML (for dataset setup)
        output_dir: Where to save prediction CSVs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config (for dataset params)
    config = load_config(config_path)
    device = config.get('device', 'cpu')

    # Load model (gets num_classes from checkpoint)
    model, ckpt_config = load_model(checkpoint_path, device)

    # Create transforms
    transform_config = config.get('transform', {})
    val_transform = create_transform(transform_config, is_train=False)

    # Load full dataset
    logger.info("Loading dataset...")
    full_dataset = MXNetRecordDataset(
        rec_path=config['data']['rec_path'],
        transform=val_transform
    )

    # Split by identity
    val_ratio = config['data'].get('val_ratio', 0.1)
    seed = config.get('seed', 42)
    max_train_ids = config['data'].get('max_train_identities')

    train_idx, val_idx, train_remap, val_remap = create_identity_split(
        full_dataset, val_ratio, seed, max_train_identities=max_train_ids
    )

    # Create subset datasets
    train_dataset = IdentitySubset(full_dataset, train_idx, train_remap)
    val_dataset = IdentitySubset(full_dataset, val_idx, val_remap)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    batch_size = config['training'].get('batch_size', 128)
    num_workers = config['data'].get('num_workers', 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Run predictions on train set
    logger.info("\nRunning predictions on train set...")
    train_preds = predict_dataset(model, train_loader, device, desc="Train")
    train_df = pd.DataFrame(train_preds)

    train_acc = 100.0 * train_df['correct'].sum() / len(train_df)
    logger.info(f"Train accuracy: {train_acc:.2f}% ({train_df['correct'].sum()}/{len(train_df)})")

    train_csv = output_dir / 'train_predictions.csv'
    train_df.to_csv(train_csv, index=False)
    logger.info(f"Saved train predictions to {train_csv}")

    # Run predictions on val set
    logger.info("\nRunning predictions on val set...")
    val_preds = predict_dataset(model, val_loader, device, desc="Val")
    val_df = pd.DataFrame(val_preds)

    val_acc = 100.0 * val_df['correct'].sum() / len(val_df)
    logger.info(f"Val accuracy: {val_acc:.2f}% ({val_df['correct'].sum()}/{len(val_df)})")

    val_csv = output_dir / 'val_predictions.csv'
    val_df.to_csv(val_csv, index=False)
    logger.info(f"Saved val predictions to {val_csv}")

    # Save summary statistics
    summary = {
        'checkpoint': str(checkpoint_path),
        'config': str(config_path),
        'train': {
            'samples': len(train_df),
            'correct': int(train_df['correct'].sum()),
            'accuracy': train_acc,
            'num_classes': len(train_remap)
        },
        'val': {
            'samples': len(val_df),
            'correct': int(val_df['correct'].sum()),
            'accuracy': val_acc,
            'num_classes': len(val_remap)
        }
    }

    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # Print confusion matrix stats (top misclassified classes)
    logger.info("\n=== Train Set Analysis ===")
    analyze_predictions(train_df, "Train")

    logger.info("\n=== Val Set Analysis ===")
    analyze_predictions(val_df, "Val")

    return train_df, val_df


def analyze_predictions(df: pd.DataFrame, split_name: str, top_n: int = 10):
    """Analyze prediction results."""
    # Overall accuracy
    acc = 100.0 * df['correct'].sum() / len(df)
    logger.info(f"{split_name} Accuracy: {acc:.2f}%")

    # Confidence distribution
    logger.info(f"Confidence - mean: {df['confidence'].mean():.4f}, "
                f"min: {df['confidence'].min():.4f}, "
                f"max: {df['confidence'].max():.4f}")

    # Correct vs incorrect confidence
    correct_conf = df[df['correct']]['confidence'].mean()
    incorrect_conf = df[~df['correct']]['confidence'].mean() if (~df['correct']).any() else 0
    logger.info(f"Avg confidence - correct: {correct_conf:.4f}, incorrect: {incorrect_conf:.4f}")

    # Per-class accuracy
    class_acc = df.groupby('true_label').agg({
        'correct': ['sum', 'count']
    })
    class_acc.columns = ['correct', 'total']
    class_acc['accuracy'] = 100.0 * class_acc['correct'] / class_acc['total']

    # Worst performing classes
    worst_classes = class_acc.nsmallest(top_n, 'accuracy')
    logger.info(f"\nWorst {top_n} classes by accuracy:")
    for label, row in worst_classes.iterrows():
        logger.info(f"  Class {label}: {row['accuracy']:.1f}% ({row['correct']}/{row['total']})")


def main():
    parser = argparse.ArgumentParser(description='ArcFace Model Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output-dir', default=None, help='Output directory for predictions')
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_dir = Path(args.checkpoint).parent
        output_dir = ckpt_dir / 'inference'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    # Run inference
    run_inference(args.checkpoint, args.config, output_dir)

    logger.info("\nInference complete!")


if __name__ == '__main__':
    main()
