"""
Training script for AVA aesthetic score prediction.

Trains a ResNet model to predict aesthetic score distributions or mean scores.

Usage:
    python -m sim_bench.training.train_ava_resnet --config configs/ava/resnet50_cpu.yaml
    python -m sim_bench.training.train_ava_resnet --config configs/ava/resnet50_gpu.yaml
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
from scipy.stats import spearmanr

from sim_bench.models.ava_resnet import AVAResNet, create_transform
from sim_bench.datasets.ava_dataset import AVADataset, load_ava_labels, create_splits

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


def create_model(config: dict) -> AVAResNet:
    """Create AVAResNet model from config."""
    model = AVAResNet(config['model'])
    device = config['device']
    return model.to(device)


def create_optimizer(model: AVAResNet, config: dict):
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
    else:  # adamw
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)


def create_dataloaders(config: dict) -> tuple:
    """Create train/val/test dataloaders."""
    # Load labels
    labels_df = load_ava_labels(config['data']['ava_txt'])
    logger.info(f"Loaded {len(labels_df)} images from AVA.txt")

    # Create splits
    train_ratio = config['data'].get('train_ratio', 0.8)
    val_ratio = config['data'].get('val_ratio', 0.1)
    seed = config.get('seed', 42)

    train_idx, val_idx, test_idx = create_splits(labels_df, train_ratio, val_ratio, seed)
    logger.info(f"Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Create transforms
    transform_config = config.get('transform', {})
    train_transform = create_transform(transform_config, is_train=True)
    val_transform = create_transform(transform_config, is_train=False)

    # Create datasets
    image_dir = config['data']['image_dir']
    output_mode = config['model'].get('output_mode', 'distribution')

    train_dataset = AVADataset(labels_df, image_dir, train_transform, train_idx, output_mode)
    val_dataset = AVADataset(labels_df, image_dir, val_transform, val_idx, output_mode)
    test_dataset = AVADataset(labels_df, image_dir, val_transform, test_idx, output_mode)

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(config['device'] == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(config['device'] == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(config['device'] == 'cuda'))

    return train_loader, val_loader, test_loader


def compute_loss(output: torch.Tensor, target: torch.Tensor,
                 output_mode: str, loss_type: str) -> torch.Tensor:
    """
    Compute loss based on output mode and loss type.

    Args:
        output: Model output (B, 10) for distribution or (B, 1) for regression
        target: Target (B, 10) for distribution or (B,) for regression
        output_mode: 'distribution' or 'regression'
        loss_type: 'kl_div', 'cross_entropy', 'mse', or 'l1'
    """
    if output_mode == 'distribution':
        if loss_type == 'kl_div':
            log_probs = F.log_softmax(output, dim=1)
            return F.kl_div(log_probs, target, reduction='batchmean')
        else:  # cross_entropy
            return F.cross_entropy(output, target)
    else:  # regression
        output = output.squeeze(-1)
        if loss_type == 'mse':
            return F.mse_loss(output, target)
        else:  # l1
            return F.l1_loss(output, target)


def compute_mean_score(output: torch.Tensor, output_mode: str) -> torch.Tensor:
    """
    Convert model output to mean score (1-10).

    Args:
        output: Model output (B, 10) for distribution or (B, 1) for regression

    Returns:
        Mean scores (B,)
    """
    if output_mode == 'distribution':
        probs = F.softmax(output, dim=1)
        scores = torch.arange(1, 11, dtype=torch.float32, device=output.device)
        return (probs * scores).sum(dim=1)
    else:  # regression
        return output.squeeze(-1).clamp(1, 10)


def train_epoch(model: AVAResNet, loader: DataLoader, optimizer,
                device: str, config: dict) -> tuple:
    """
    Train for one epoch.

    Returns:
        (avg_loss, all_pred_means, all_gt_means)
    """
    model.train()
    output_mode = config['model'].get('output_mode', 'distribution')
    loss_type = config['training'].get('loss_type', 'kl_div')
    log_interval = config.get('log_interval', 50)

    total_loss = 0.0
    all_pred_means = []
    all_gt_means = []

    for batch_idx, batch in enumerate(loader, 1):
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        gt_means = batch['mean_score'].numpy()

        optimizer.zero_grad()
        output = model(images)
        loss = compute_loss(output, targets, output_mode, loss_type)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_means = compute_mean_score(output, output_mode).detach().cpu().numpy()
        all_pred_means.extend(pred_means)
        all_gt_means.extend(gt_means)

        if batch_idx % log_interval == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    return avg_loss, np.array(all_pred_means), np.array(all_gt_means)


def evaluate(model: AVAResNet, loader: DataLoader, device: str, config: dict,
             output_dir: Path = None, epoch: int = None) -> tuple:
    """
    Evaluate model and optionally save predictions.

    Returns:
        (avg_loss, spearman_corr, pred_means, gt_means, image_ids, pred_dists, gt_dists)
    """
    model.eval()
    output_mode = config['model'].get('output_mode', 'distribution')
    loss_type = config['training'].get('loss_type', 'kl_div')

    total_loss = 0.0
    all_pred_means = []
    all_gt_means = []
    all_image_ids = []
    all_pred_dists = []
    all_gt_dists = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            gt_means = batch['mean_score'].numpy()
            image_ids = batch['image_id']
            gt_dists = batch['distribution'].numpy()

            output = model(images)
            loss = compute_loss(output, targets, output_mode, loss_type)

            total_loss += loss.item()
            pred_means = compute_mean_score(output, output_mode).cpu().numpy()

            if output_mode == 'distribution':
                pred_dists = F.softmax(output, dim=1).cpu().numpy()
            else:
                pred_dists = np.zeros((len(output), 10))

            all_pred_means.extend(pred_means)
            all_gt_means.extend(gt_means)
            all_image_ids.extend(image_ids)
            all_pred_dists.extend(pred_dists)
            all_gt_dists.extend(gt_dists)

    avg_loss = total_loss / len(loader)
    pred_means = np.array(all_pred_means)
    gt_means = np.array(all_gt_means)

    # Compute Spearman correlation
    spearman_corr, _ = spearmanr(pred_means, gt_means)

    # Save predictions if requested
    if output_dir is not None and epoch is not None and config.get('save_val_predictions', False):
        save_predictions(all_image_ids, pred_means, gt_means,
                        np.array(all_pred_dists), np.array(all_gt_dists),
                        output_dir, epoch)

    return avg_loss, spearman_corr, pred_means, gt_means, all_image_ids, all_pred_dists, all_gt_dists


def save_predictions(image_ids: list, pred_means: np.ndarray, gt_means: np.ndarray,
                     pred_dists: np.ndarray, gt_dists: np.ndarray,
                     output_dir: Path, epoch: int):
    """Save predictions to parquet file."""
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    data = {
        'image_id': image_ids,
        'pred_mean': pred_means,
        'gt_mean': gt_means,
    }

    # Add distribution columns
    for i in range(10):
        data[f'pred_dist_{i+1}'] = pred_dists[:, i]
        data[f'gt_dist_{i+1}'] = gt_dists[:, i]

    df = pd.DataFrame(data)
    df.to_parquet(pred_dir / f'val_epoch_{epoch:03d}.parquet', index=False)
    logger.info(f"Saved predictions to {pred_dir / f'val_epoch_{epoch:03d}.parquet'}")


def train_model(model: AVAResNet, train_loader: DataLoader, val_loader: DataLoader,
                optimizer, config: dict, output_dir: Path) -> tuple:
    """
    Main training loop with early stopping.

    Returns:
        (best_spearman, history)
    """
    device = config['device']
    max_epochs = config['training']['max_epochs']
    patience = config['training'].get('early_stop_patience', 5)

    best_spearman = -1.0
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_spearman': [],
        'val_spearman': []
    }

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")

        # Train
        train_loss, train_preds, train_gts = train_epoch(model, train_loader, optimizer, device, config)
        train_spearman, _ = spearmanr(train_preds, train_gts)
        logger.info(f"  Train: loss={train_loss:.4f}, spearman={train_spearman:.4f}")

        # Validate
        val_loss, val_spearman, _, _, _, _, _ = evaluate(model, val_loader, device, config, output_dir, epoch)
        logger.info(f"  Val: loss={val_loss:.4f}, spearman={val_spearman:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_spearman'].append(train_spearman)
        history['val_spearman'].append(val_spearman)

        # Check for improvement
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_spearman': val_spearman,
                'config': config
            }, output_dir / 'best_model.pt')
            logger.info(f"  New best model saved (spearman={best_spearman:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_spearman, history


def plot_training_curves(history: dict, output_dir: Path):
    """Plot and save training curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Spearman plot
    ax2.plot(epochs, history['train_spearman'], 'b-', label='Train')
    ax2.plot(epochs, history['val_spearman'], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Training and Validation Spearman')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train AVA ResNet')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output-dir', default=None, help='Override output directory')
    parser.add_argument('--device', default=None, help='Override device (cpu/cuda)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.device:
        config['device'] = args.device

    # Setup output directory
    output_dir = Path(
        config.get('output_dir') or
        f"outputs/ava/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
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

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Set seeds
    set_random_seeds(config.get('seed', 42))

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model output mode: {config['model'].get('output_mode', 'distribution')}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Train
    logger.info("Starting training...")
    best_spearman, history = train_model(model, train_loader, val_loader, optimizer, config, output_dir)

    # Plot curves
    plot_training_curves(history, output_dir)

    # Test evaluation
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_spearman, _, _, _, _, _ = evaluate(model, test_loader, config['device'], config)
    logger.info(f"Test: loss={test_loss:.4f}, spearman={test_spearman:.4f}")

    # Save final results
    results = {
        'best_val_spearman': best_spearman,
        'test_loss': test_loss,
        'test_spearman': test_spearman,
        'final_epoch': len(history['train_loss'])
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete. Best val spearman: {best_spearman:.4f}, Test spearman: {test_spearman:.4f}")


if __name__ == '__main__':
    main()
