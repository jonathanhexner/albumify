"""
Evaluate a saved model checkpoint and save complete validation results.

This script loads a trained model checkpoint (best_model.pt) and evaluates it
on the validation set, saving ALL predictions (not limited to 200 samples).

Usage:
    python evaluate_best_model.py --checkpoint outputs/your_run/best_model.pt --output outputs/your_run/complete_validation
    python evaluate_best_model.py --checkpoint outputs/your_run/best_model.pt  # Auto-generates output dir
"""
import argparse
import yaml
import logging
import torch
import numpy as np
from pathlib import Path

from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.datasets.transform_factory import create_transform
from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.siamese_dataloaders import get_dataset_from_loader
from sim_bench.training.train_siamese_e2e import set_random_seeds, create_model, compute_batch_metrics

logger = logging.getLogger(__name__)


def save_complete_metrics(all_preds, all_winners, all_logprobs, all_image1, all_image2,
                          dataset, avg_loss, metrics_dir):
    """
    Save COMPLETE validation metrics (all samples, not limited to 200).

    This is a modified version of save_epoch_metrics() that saves all predictions.
    """
    import json
    import pandas as pd
    from sim_bench.training.diagnostics import compute_confusion_matrix, extract_series_id

    # Check if dataset has metadata
    has_metadata = hasattr(dataset, 'get_dataframe')
    pairs_df = dataset.get_dataframe() if has_metadata else None

    # Confusion matrix
    cm = compute_confusion_matrix(all_winners, all_preds)
    recall_0 = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() > 0 else 0.0
    recall_1 = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() > 0 else 0.0

    summary = {
        'loss': float(avg_loss),
        'acc': float((all_preds == all_winners).mean()),
        'target_counts': {'0': int((all_winners == 0).sum()), '1': int((all_winners == 1).sum())},
        'pred_counts': {'0': int((all_preds == 0).sum()), '1': int((all_preds == 1).sum())},
        'confusion': cm.tolist(),
        'recall_class0': recall_0,
        'recall_class1': recall_1,
        'total_samples': len(all_preds)
    }

    with open(metrics_dir / 'eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save ALL predictions (no limit)
    n_dump = len(all_preds)
    logger.info(f"Saving {n_dump} predictions (complete validation set)")

    # Convert logits to probabilities using softmax
    all_probs = np.exp(all_logprobs) / np.exp(all_logprobs).sum(axis=1, keepdims=True)
    dump_data = {
        'image1': all_image1[:n_dump],
        'image2': all_image2[:n_dump],
        'winner': all_winners[:n_dump].tolist(),
        'pred': all_preds[:n_dump].tolist(),
        'logit0': [float(logit[0]) for logit in all_logprobs[:n_dump]],
        'logit1': [float(logit[1]) for logit in all_logprobs[:n_dump]],
        'p0': [float(prob[0]) for prob in all_probs[:n_dump]],
        'p1': [float(prob[1]) for prob in all_probs[:n_dump]]
    }

    # Add metadata - PhotoTriage has agreement/num_reviewers, both have series_id
    if pairs_df is not None:
        # PhotoTriage - get all metadata from pairs_df
        agreements = []
        num_reviewers_list = []
        series_ids = []

        for img1, img2 in zip(all_image1[:n_dump], all_image2[:n_dump]):
            match = pairs_df[(pairs_df['image1'] == img1) & (pairs_df['image2'] == img2)]
            if len(match) > 0:
                row = match.iloc[0]
                agreements.append(float(row['agreement']))
                num_reviewers_list.append(int(row['num_reviewers']))
                series_ids.append(str(row['series_id']))
            else:
                agreements.append(0.0)
                num_reviewers_list.append(0)
                series_ids.append('unknown')

        dump_data['agreement'] = agreements
        dump_data['num_reviewers'] = num_reviewers_list
        dump_data['series_id'] = series_ids
    else:
        # External - extract series_id from filename (000001-01.JPG -> 000001)
        dump_data['series_id'] = [extract_series_id(img1) for img1 in all_image1[:n_dump]]

    pd.DataFrame(dump_data).to_csv(metrics_dir / 'eval_dump.csv', index=False)
    logger.info(f"Saved complete results to {metrics_dir / 'eval_dump.csv'}")


def evaluate_complete(model, loader, device, output_dir, log_interval=10):
    """
    Evaluate model and save COMPLETE validation results.

    Similar to train_siamese_e2e.evaluate() but saves all predictions.
    """
    # Extract dataset from loader
    dataset = get_dataset_from_loader(loader)

    model.eval()

    # Create directories
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Collect predictions
    all_preds, all_winners, all_logprobs = [], [], []
    all_image1, all_image2 = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            winners = batch['winner'].to(device)

            # Use common metrics function
            logits = model(img1, img2)
            loss, batch_acc, _, _ = compute_batch_metrics(logits, winners)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_winners.extend(winners.cpu().numpy())
            all_logprobs.extend(logits.detach().cpu().numpy())
            all_image1.extend(batch['image1'])
            all_image2.extend(batch['image2'])
            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                logger.info(f"  Eval Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}, acc={batch_acc:.3f}")

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_winners = np.array(all_winners)
    all_logprobs = np.array(all_logprobs)
    avg_loss = total_loss / len(loader)
    avg_acc = (all_preds == all_winners).mean()

    # Save complete diagnostics
    save_complete_metrics(all_preds, all_winners, all_logprobs, all_image1, all_image2,
                         dataset, avg_loss, metrics_dir)

    logger.info(f"  Validation: loss={avg_loss:.4f}, acc={avg_acc:.3f}, samples={len(all_preds)}")
    return avg_loss, avg_acc


def create_validation_loader(config, transform, batch_size):
    """
    Create validation dataloader from config.

    Supports both PhotoTriage and external dataloaders.
    """
    seed = config.get('seed')
    set_random_seeds(seed)

    use_external = config.get('use_external_dataloader', False)
    factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=seed)

    if use_external:
        # External dataloader
        import sys
        external_path = config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection')
        if external_path not in sys.path:
            logger.info(f"Adding external path to sys.path: {external_path}")
            sys.path.insert(0, external_path)

        from data.dataloader import MyDataset

        # Get image_root
        image_root = config['data'].get('image_root') or config['data'].get('root_dir')
        if image_root is None:
            raise ValueError("Must specify 'image_root' or 'root_dir' in config['data']")

        logger.info(f"Using external dataloader with image_root: {image_root}")
        val_data = MyDataset(train=False, image_root=image_root, seed=seed)

        _, val_loader, _ = factory.create_from_external(None, val_data, None)
        return val_loader
    else:
        # PhotoTriage
        data = PhotoTriageData(
            config['data']['root_dir'],
            config['data']['min_agreement'],
            config['data']['min_reviewers']
        )

        train_df, val_df, test_df = data.get_series_based_splits(
            0.8, 0.1, 0.1,
            seed,
            config['data'].get('quick_experiment')
        )

        logger.info(f"PhotoTriage: {len(val_df)} validation samples")
        _, val_loader, _ = factory.create_from_phototriage(data, train_df, val_df, test_df, transform)
        return val_loader


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved model checkpoint with complete results')
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pt checkpoint')
    parser.add_argument('--output', default=None, help='Output directory (default: checkpoint_dir/complete_validation)')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N batches')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Auto-generate output directory if not specified
    if args.output is None:
        output_dir = checkpoint_path.parent / 'complete_validation'
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / 'evaluation.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    epoch = checkpoint['epoch']
    saved_val_acc = checkpoint['val_acc']

    logger.info(f"Loaded checkpoint from epoch {epoch} with validation accuracy: {saved_val_acc:.3f}")

    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    logger.info(f"Using device: {device}")

    # Set random seeds
    set_random_seeds(config.get('seed'))

    # Create model and load weights
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")

    # Create transform and validation loader
    transform = create_transform(config)
    val_loader = create_validation_loader(
        config,
        transform,
        config['training']['batch_size']
    )
    logger.info(f"Validation loader created with {len(val_loader)} batches")

    # Evaluate with complete results
    logger.info("\nEvaluating model on validation set...")
    val_loss, val_acc = evaluate_complete(
        model, val_loader, device, output_dir,
        log_interval=args.log_interval
    )

    # Verify accuracy matches
    acc_diff = abs(val_acc - saved_val_acc)
    if acc_diff < 0.001:
        logger.info(f"✓ Accuracy matches checkpoint: {val_acc:.3f}")
    else:
        logger.warning(f"⚠ Accuracy differs from checkpoint: {val_acc:.3f} vs {saved_val_acc:.3f} (diff: {acc_diff:.4f})")

    # Save config for reference
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    logger.info(f"\n✓ Complete evaluation results saved to: {output_dir}")
    logger.info(f"  - Metrics: {output_dir / 'metrics' / 'eval_dump.csv'}")
    logger.info(f"  - Summary: {output_dir / 'metrics' / 'eval_summary.json'}")


if __name__ == '__main__':
    main()
