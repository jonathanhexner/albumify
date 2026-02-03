"""Verify train/val identity split has no overlap.

Produces a DataFrame showing:
- All sample indices with their original and remapped labels
- Which split each sample belongs to (train/val)
- Verifies zero identity overlap between splits

Usage:
    python -m sim_bench.face_recognition.verify_split --config configs/face/resnet50_arcface.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from sim_bench.datasets.face_dataset import MXNetRecordDataset
from sim_bench.face_recognition.utils import load_config, setup_logging
from sim_bench.face_recognition.train import create_identity_split

logger = logging.getLogger(__name__)


def verify_identity_split(config: dict, output_dir: Path = None):
    """
    Verify train/val split has no identity overlap.

    Args:
        config: Configuration dictionary
        output_dir: Where to save verification results

    Returns:
        DataFrame with sample info and split assignments
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = MXNetRecordDataset(
        rec_path=config['data']['rec_path'],
        num_classes=config['model'].get('num_classes')
    )

    # Run identity split with same seed as training
    val_ratio = config['data'].get('val_ratio', 0.1)
    seed = config.get('seed', 42)
    max_train_ids = config['data'].get('max_train_identities')

    logger.info(f"Running identity split (val_ratio={val_ratio}, seed={seed})...")
    train_idx, val_idx, train_remap, val_remap = create_identity_split(
        dataset, val_ratio, seed, max_train_identities=max_train_ids
    )

    # Get identity sets
    train_identities = set(train_remap.keys())
    val_identities = set(val_remap.keys())

    # Check for overlap
    overlap = train_identities & val_identities
    logger.info(f"\n=== Identity Split Verification ===")
    logger.info(f"Train identities: {len(train_identities)}")
    logger.info(f"Val identities:   {len(val_identities)}")
    logger.info(f"Overlap:          {len(overlap)}")

    if overlap:
        logger.error(f"OVERLAP DETECTED! Overlapping identities: {sorted(list(overlap))[:20]}...")
    else:
        logger.info("SUCCESS: No identity overlap between train and val splits")

    # Build DataFrame with all samples
    logger.info("\nBuilding sample DataFrame...")
    records = []

    # Train samples
    train_idx_set = set(train_idx)
    for idx in train_idx:
        offset, orig_label = dataset.samples[idx]
        records.append({
            'sample_index': idx,
            'original_label': orig_label,
            'remapped_label': train_remap[orig_label],
            'split': 'train'
        })

    # Val samples
    for idx in val_idx:
        offset, orig_label = dataset.samples[idx]
        records.append({
            'sample_index': idx,
            'original_label': orig_label,
            'remapped_label': val_remap[orig_label],
            'split': 'val'
        })

    df = pd.DataFrame(records)

    # Sort by sample index
    df = df.sort_values('sample_index').reset_index(drop=True)

    # Save to CSV
    csv_path = output_dir / 'split_verification.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved split verification to {csv_path}")

    # Print summary statistics
    logger.info(f"\n=== Sample Statistics ===")
    logger.info(f"Total samples:  {len(df)}")
    logger.info(f"Train samples:  {len(df[df['split'] == 'train'])}")
    logger.info(f"Val samples:    {len(df[df['split'] == 'val'])}")

    # Verify label remapping is contiguous
    train_remapped = df[df['split'] == 'train']['remapped_label'].unique()
    val_remapped = df[df['split'] == 'val']['remapped_label'].unique()
    logger.info(f"\nTrain remapped labels: {min(train_remapped)} to {max(train_remapped)} ({len(train_remapped)} unique)")
    logger.info(f"Val remapped labels:   {min(val_remapped)} to {max(val_remapped)} ({len(val_remapped)} unique)")

    # Save identity mapping
    identity_df = pd.DataFrame([
        {'original_label': k, 'remapped_label': v, 'split': 'train'}
        for k, v in train_remap.items()
    ] + [
        {'original_label': k, 'remapped_label': v, 'split': 'val'}
        for k, v in val_remap.items()
    ])
    identity_path = output_dir / 'identity_mapping.csv'
    identity_df.to_csv(identity_path, index=False)
    logger.info(f"Saved identity mapping to {identity_path}")

    return df, len(overlap) == 0


def main():
    parser = argparse.ArgumentParser(description='Verify Train/Val Identity Split')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output-dir', default=None, help='Output directory for verification files')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs/face_recognition/split_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    # Run verification
    df, is_valid = verify_identity_split(config, output_dir)

    if is_valid:
        logger.info("\nVerification PASSED: No identity overlap")
    else:
        logger.error("\nVerification FAILED: Identity overlap detected!")
        exit(1)


if __name__ == '__main__':
    main()
