"""
Training script for eye open/closed classification.

Trains a ResNet model on the MRL Eye Dataset for binary classification.

Usage:
    python -m sim_bench.training.train_eye_state --config configs/eye_state/resnet18_cpu.yaml
    python -m sim_bench.training.train_eye_state --config configs/eye_state/resnet18_gpu.yaml
"""
import argparse
import yaml
import logging
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sim_bench.models.eye_state_classifier import EyeStateClassifier, create_eye_transform
from sim_bench.datasets.mrl_eye_dataset import MRLEyeDataset, load_mrl_labels

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: dict) -> EyeStateClassifier:
    """Create EyeStateClassifier model from config."""
    model = EyeStateClassifier(config['model'])
    device = config['device']
    return model.to(device)


def create_optimizer(model: EyeStateClassifier, config: dict):
    """Create optimizer with optional differential learning rates."""
    opt_name = config['training']['optimizer'].lower()
    base_lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']
    use_diff_lr = config['training'].get('differential_lr', True)

    if use_diff_lr:
        param_groups = [
            {'params': list(model.get_1x_lr_params()), 'lr': base_lr},
            {'params': list(model.get_10x_lr_params()), 'lr': base_lr * 10}
        ]
        logger.info(f"Using differential LR: backbone={base_lr}, head={base_lr * 10}")
    else:
        param_groups = model.parameters()
        logger.info(f"Using single LR: {base_lr}")

    if opt_name == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=wd)

    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)


def create_dataloaders(config: dict) -> tuple:
    """Create train/val/test dataloaders from pre-split MRL dataset."""
    data_dir = config['data']['data_dir']
    cached_parquet = config['data'].get('cached_parquet')
    max_samples = config['data'].get('max_samples')

    # Load each split separately (MRL has pre-defined train/val/test folders)
    train_df = load_mrl_labels(data_dir, split='train', cached_parquet=cached_parquet, max_samples=max_samples)
    val_df = load_mrl_labels(data_dir, split='val', cached_parquet=cached_parquet, max_samples=max_samples)
    test_df = load_mrl_labels(data_dir, split='test', cached_parquet=cached_parquet, max_samples=max_samples)

    logger.info(f"Splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    logger.info(f"Train class balance: {train_df['label'].mean():.2%} open (awake)")

    # Create transforms
    transform_config = config.get('transform', {})
    train_transform = create_eye_transform(transform_config, is_train=True)
    val_transform = create_eye_transform(transform_config, is_train=False)

    # Create datasets
    train_dataset = MRLEyeDataset(train_df, train_transform)
    val_dataset = MRLEyeDataset(val_df, val_transform)
    test_dataset = MRLEyeDataset(test_df, val_transform)

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 0)
    pin_memory = config['device'] == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute binary cross entropy loss.

    Args:
        output: Model logits (batch_size, 1)
        target: Labels (batch_size,) with values 0 or 1
        pos_weight: Weight for positive class (for class imbalance)
    """
    output = output.squeeze(-1)
    target = target.float()
    return F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute classification metrics.

    Args:
        predictions: Binary predictions (0 or 1)
        labels: Ground truth labels

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
    }


def train_epoch(
    model: EyeStateClassifier,
    loader: DataLoader,
    optimizer,
    device: str,
    config: dict,
    epoch: int = None,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        (avg_loss, all_preds, all_labels)
    """
    model.train()
    log_interval = config.get('log_interval', 50)
    max_batches = config.get('max_batches')

    # Class weighting for imbalance
    pos_weight = config['training'].get('pos_weight')
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight], device=device)

    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch_idx, batch in enumerate(loader, 1):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = compute_loss(output, labels, pos_weight)
        loss.backward()
        optimizer.step()

        # Collect telemetry
        from sim_bench.telemetry.eye_state_telemetry import collect_telemetry
        collect_telemetry(model, optimizer, batch_idx, epoch, device, config)

        total_loss += loss.item()
        num_batches += 1

        preds = (torch.sigmoid(output.squeeze(-1)) > 0.5).cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % log_interval == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}")

        if max_batches and batch_idx >= max_batches:
            logger.info(f"  Stopping after {max_batches} batches (max_batches limit)")
            break

    avg_loss = total_loss / num_batches
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate(
    model: EyeStateClassifier,
    loader: DataLoader,
    device: str,
    config: dict,
    output_dir: Path = None,
    epoch: int = None,
) -> tuple:
    """
    Evaluate model.

    Returns:
        (avg_loss, metrics_dict, all_preds, all_labels, all_filenames)
    """
    model.eval()
    max_batches = config.get('max_batches')

    pos_weight = config['training'].get('pos_weight')
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight], device=device)

    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    all_filenames = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            output = model(images)
            loss = compute_loss(output, labels, pos_weight)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(output.squeeze(-1)).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(batch['filename'])

            if max_batches and batch_idx >= max_batches:
                break

    if num_batches == 0:
        return 0.0, {}, np.array([]), np.array([]), []

    avg_loss = total_loss / num_batches
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_preds, all_labels)

    # Save predictions if requested
    if output_dir is not None and epoch is not None and config.get('save_val_predictions', False):
        save_predictions(
            all_filenames, all_probs, all_preds, all_labels, output_dir, epoch
        )

    return avg_loss, metrics, all_preds, all_labels, all_filenames


def save_predictions(
    filenames: list,
    probs: list,
    preds: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    epoch: int,
):
    """Save predictions to parquet file."""
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        'filename': filenames,
        'prob_open': probs,
        'pred': preds,
        'label': labels,
        'correct': preds == labels,
    })
    df.to_parquet(pred_dir / f'val_epoch_{epoch:03d}.parquet', index=False)
    logger.info(f"Saved predictions to {pred_dir / f'val_epoch_{epoch:03d}.parquet'}")


def train_model(
    model: EyeStateClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    config: dict,
    output_dir: Path,
) -> tuple:
    """
    Main training loop with early stopping.

    Returns:
        (best_f1, history)
    """
    device = config['device']
    max_epochs = config['training']['max_epochs']
    patience = config['training'].get('early_stop_patience', 10)

    best_f1 = 0.0
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1': [],
        'val_f1': [],
    }

    # Check if val_loader is empty (for overfit sanity check)
    use_train_for_eval = len(val_loader.dataset) == 0

    # Initialize telemetry
    from sim_bench.telemetry.eye_state_telemetry import init_telemetry
    init_telemetry(config)

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")

        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, device, config, epoch
        )
        train_metrics = compute_metrics(train_preds, train_labels)
        logger.info(
            f"  Train: loss={train_loss:.4f}, acc={train_metrics['accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}"
        )

        # Validate
        if use_train_for_eval:
            val_loss, val_metrics, _, _, _ = evaluate(
                model, train_loader, device, config, output_dir, epoch
            )
            logger.info(
                f"  Eval (on train): loss={val_loss:.4f}, acc={val_metrics['accuracy']:.4f}, "
                f"f1={val_metrics['f1']:.4f}"
            )
        else:
            val_loss, val_metrics, _, _, _ = evaluate(
                model, val_loader, device, config, output_dir, epoch
            )
            logger.info(
                f"  Val: loss={val_loss:.4f}, acc={val_metrics['accuracy']:.4f}, "
                f"f1={val_metrics['f1']:.4f}"
            )

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])

        # Check for improvement (using F1 as primary metric)
        val_f1 = val_metrics['f1']
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_metrics['accuracy'],
                'config': config
            }, output_dir / 'best_model.pt')
            logger.info(f"  New best model saved (f1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_f1, history


def plot_training_curves(history: dict, output_dir: Path):
    """Plot and save training curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # F1 plot
    axes[2].plot(epochs, history['train_f1'], 'b-', label='Train')
    axes[2].plot(epochs, history['val_f1'], 'r-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Training and Validation F1')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Eye State Classifier')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup output directory
    output_dir_str = config.get('output_dir')
    if output_dir_str:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path(f"outputs/eye_state/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Output directory: {output_dir}")

    # Update config with resolved output_dir
    config['output_dir'] = str(output_dir)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Set seeds
    set_random_seeds(config.get('seed', 42))

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model backbone: {config['model'].get('backbone', 'resnet18')}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Train
    logger.info("Starting training...")
    best_f1, history = train_model(
        model, train_loader, val_loader, optimizer, config, output_dir
    )

    # Plot curves
    plot_training_curves(history, output_dir)

    # Test evaluation
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    if len(test_loader.dataset) > 0:
        logger.info("Evaluating on test set...")
        test_loss, test_metrics, _, _, _ = evaluate(
            model, test_loader, config['device'], config
        )
        logger.info(
            f"Test: loss={test_loss:.4f}, acc={test_metrics['accuracy']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, precision={test_metrics['precision']:.4f}, "
            f"recall={test_metrics['recall']:.4f}"
        )
    else:
        logger.info("No test set (skipping test evaluation)")
        test_metrics = None

    # Save final results
    results = {
        'best_val_f1': best_f1,
        'test_metrics': test_metrics,
        'final_epoch': len(history['train_loss'])
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete. Best val F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
