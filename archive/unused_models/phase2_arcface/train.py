"""
Training script for multitask face model with ArcFace backbone.

Trains on AffectNet with two tasks:
- Expression classification (8 classes)
- Landmark regression (68 key points)

Uses pretrained ArcFace backbone (frozen or fine-tuned).

Usage:
    python -m sim_bench.training.phase2_arcface.train --config configs/face/multitask_arcface.yaml
"""

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from sim_bench.models.multitask_face_arcface import MultitaskFaceModelArcFace
from sim_bench.training.phase2_pretraining.affectnet_dataset import (
    AffectNetDataset,
    create_affectnet_transform,
    create_val_transform,
    get_horizontal_flip_prob
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, log_level: int = logging.INFO):
    """Setup logging to both console and file."""
    log_file = output_dir / 'training.log'

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


class MetricsLogger:
    """Log metrics to CSV and TensorBoard."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.csv_path = output_dir / 'metrics.csv'
        self.tb_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
        self._csv_initialized = False

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict,
                  lr: float, uncertainty_weights: dict = None):
        """Log metrics for an epoch."""
        row = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_expr_acc': train_metrics['expression_accuracy'],
            'train_lm_error': train_metrics['landmark_error'],
            'val_loss': val_metrics['loss'],
            'val_expr_acc': val_metrics['expression_accuracy'],
            'val_lm_error': val_metrics['landmark_error'],
            'learning_rate': lr,
        }

        if uncertainty_weights:
            row.update(uncertainty_weights)

        self._write_csv(row)

        self.tb_writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        self.tb_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        self.tb_writer.add_scalar('Expression_Accuracy/train', train_metrics['expression_accuracy'], epoch)
        self.tb_writer.add_scalar('Expression_Accuracy/val', val_metrics['expression_accuracy'], epoch)
        self.tb_writer.add_scalar('Landmark_Error/train', train_metrics['landmark_error'], epoch)
        self.tb_writer.add_scalar('Landmark_Error/val', val_metrics['landmark_error'], epoch)
        self.tb_writer.add_scalar('Learning_Rate', lr, epoch)

        if uncertainty_weights:
            self.tb_writer.add_scalar('Uncertainty/expression_weight', uncertainty_weights.get('expression_weight', 1), epoch)
            self.tb_writer.add_scalar('Uncertainty/landmark_weight', uncertainty_weights.get('landmark_weight', 1), epoch)

        self.tb_writer.flush()

    def _write_csv(self, row: dict):
        """Write a row to CSV file."""
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
        else:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

    def close(self):
        """Close the TensorBoard writer."""
        self.tb_writer.close()


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


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    data_cfg = config['data']
    train_cfg = config['training']

    # Get landmark cache path
    landmarks_cache = data_cfg.get('landmarks_cache')
    if landmarks_cache:
        landmarks_cache = Path(landmarks_cache)

    # Number of landmarks
    num_landmarks = data_cfg.get('num_landmarks', 68)

    # Create transforms (using 112x112 input to match ArcFace)
    train_transform = create_affectnet_transform(config)
    val_transform = create_val_transform(config)

    # Horizontal flip probability
    flip_prob = get_horizontal_flip_prob(config)

    # Create datasets
    train_dataset = AffectNetDataset(
        data_dir=Path(data_cfg['root_dir']) / 'train',
        landmarks_cache=landmarks_cache,
        transform=train_transform,
        num_landmarks=num_landmarks,
        horizontal_flip_prob=flip_prob
    )

    val_dataset = AffectNetDataset(
        data_dir=Path(data_cfg['root_dir']) / 'val',
        landmarks_cache=landmarks_cache,
        transform=val_transform,
        num_landmarks=num_landmarks,
        horizontal_flip_prob=0.0  # No flip for validation
    )

    # Create dataloaders
    batch_size = train_cfg['batch_size']
    num_workers = data_cfg.get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer with optional differential learning rates."""
    train_cfg = config['training']
    lr = train_cfg['learning_rate']
    optimizer_name = train_cfg.get('optimizer', 'adam').lower()
    weight_decay = train_cfg.get('weight_decay', 0.0)

    # Differential learning rate for backbone
    use_differential_lr = train_cfg.get('differential_lr', True)
    backbone_lr_mult = train_cfg.get('backbone_lr_multiplier', 0.1)

    if use_differential_lr and not model.is_backbone_frozen:
        # Lower LR for backbone, higher for heads
        param_groups = [
            {'params': model.get_backbone_params(), 'lr': lr * backbone_lr_mult},
            {'params': model.get_head_params(), 'lr': lr}
        ]
        logger.info(f"Using differential LR: backbone={lr * backbone_lr_mult}, heads={lr}")
    else:
        param_groups = [{'params': model.parameters(), 'lr': lr}]

    if optimizer_name == 'adam':
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = train_cfg.get('momentum', 0.9)
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, config: dict, num_batches_per_epoch: int):
    """Create learning rate scheduler."""
    train_cfg = config['training']
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'none')

    if scheduler_type == 'none':
        return None

    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get('step_size', 10),
            gamma=scheduler_cfg.get('gamma', 0.1)
        )

    if scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_cfg.get('milestones', [10, 20]),
            gamma=scheduler_cfg.get('gamma', 0.1)
        )

    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg['max_epochs'],
            eta_min=scheduler_cfg.get('min_lr', 1e-6)
        )

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_epoch(model, loader, optimizer, device, config, epoch):
    """Train for one epoch."""
    model.train()
    use_uncertainty = config['training'].get('use_uncertainty_weighting', True)
    log_interval = config.get('log_interval', 100)

    total_loss = 0.0
    total_lm_loss = 0.0
    total_expr_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_lm_error = 0.0
    total_lm_samples = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader, 1):
        images = batch['image'].to(device)
        expressions = batch['expression'].to(device)
        landmarks = batch['landmarks'].to(device)
        has_landmarks = batch['has_landmarks'].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Compute loss
        losses = model.compute_loss(
            outputs,
            landmarks,
            expressions,
            use_uncertainty_weighting=use_uncertainty
        )

        losses['total'].backward()
        optimizer.step()

        # Track metrics
        total_loss += losses['total'].item()
        total_lm_loss += losses['landmark'].item()
        total_expr_loss += losses['expression'].item()
        num_batches += 1

        # Expression accuracy
        preds = outputs['expression'].argmax(dim=1)
        total_correct += (preds == expressions).sum().item()
        total_samples += len(expressions)

        # Landmark error (only for samples with landmarks)
        if has_landmarks.any():
            lm_pred = outputs['landmarks'][has_landmarks]
            lm_target = landmarks[has_landmarks]
            lm_error = torch.mean(torch.abs(lm_pred - lm_target)).item()
            total_lm_error += lm_error * has_landmarks.sum().item()
            total_lm_samples += has_landmarks.sum().item()

        if batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Batch {batch_idx}/{len(loader)}: "
                       f"loss={losses['total'].item():.4f}, "
                       f"expr_acc={100.0 * total_correct / total_samples:.1f}%, "
                       f"lr={lr:.6f}")

    return {
        'loss': total_loss / num_batches,
        'expression_accuracy': 100.0 * total_correct / total_samples,
        'landmark_error': total_lm_error / total_lm_samples if total_lm_samples > 0 else 0.0,
        'expression_loss': total_expr_loss / num_batches,
        'landmark_loss': total_lm_loss / num_batches
    }


def evaluate(model, loader, device, config):
    """Evaluate on validation set."""
    model.eval()
    use_uncertainty = config['training'].get('use_uncertainty_weighting', True)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_lm_error = 0.0
    total_lm_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            expressions = batch['expression'].to(device)
            landmarks = batch['landmarks'].to(device)
            has_landmarks = batch['has_landmarks'].to(device)

            outputs = model(images)

            losses = model.compute_loss(
                outputs,
                landmarks,
                expressions,
                use_uncertainty_weighting=use_uncertainty
            )

            total_loss += losses['total'].item()
            num_batches += 1

            preds = outputs['expression'].argmax(dim=1)
            total_correct += (preds == expressions).sum().item()
            total_samples += len(expressions)

            if has_landmarks.any():
                lm_pred = outputs['landmarks'][has_landmarks]
                lm_target = landmarks[has_landmarks]
                lm_error = torch.mean(torch.abs(lm_pred - lm_target)).item()
                total_lm_error += lm_error * has_landmarks.sum().item()
                total_lm_samples += has_landmarks.sum().item()

    return {
        'loss': total_loss / num_batches,
        'expression_accuracy': 100.0 * total_correct / total_samples,
        'landmark_error': total_lm_error / total_lm_samples if total_lm_samples > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='Train Multitask Face Model with ArcFace Backbone')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup output directory
    output_dir = Path(f"outputs/phase2_arcface/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Set seeds
    set_random_seeds(config.get('seed', 42))

    # Device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Create model from ArcFace checkpoint
    logger.info("Creating model...")
    model_cfg = config['model']
    arcface_checkpoint = model_cfg['arcface_checkpoint']

    model = MultitaskFaceModelArcFace.from_arcface_checkpoint(arcface_checkpoint, model_cfg)
    model = model.to(device)

    # Log model info
    param_counts = model.get_num_params()
    logger.info(f"Model params: total={param_counts['total']:,}, "
               f"trainable={param_counts['trainable']:,}, "
               f"backbone={param_counts['backbone']:,}, "
               f"heads={param_counts['heads']:,}")
    logger.info(f"Backbone frozen: {model.is_backbone_frozen}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    # Metrics logger
    metrics_logger = MetricsLogger(output_dir)

    # Training loop
    train_cfg = config['training']
    max_epochs = train_cfg['max_epochs']
    unfreeze_epoch = train_cfg.get('unfreeze_backbone_epoch')
    best_val_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(1, max_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"\nEpoch {epoch}/{max_epochs} (lr={lr:.6f})")

        # Unfreeze backbone at specified epoch
        if unfreeze_epoch and epoch == unfreeze_epoch and model.is_backbone_frozen:
            logger.info(f"Unfreezing backbone at epoch {epoch}")
            model.unfreeze_backbone()
            # Recreate optimizer with differential LR
            optimizer = create_optimizer(model, config)
            scheduler = create_scheduler(optimizer, config, len(train_loader))

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch)
        logger.info(f"  Train: loss={train_metrics['loss']:.4f}, "
                   f"expr_acc={train_metrics['expression_accuracy']:.2f}%, "
                   f"lm_error={train_metrics['landmark_error']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, config)
        logger.info(f"  Val:   loss={val_metrics['loss']:.4f}, "
                   f"expr_acc={val_metrics['expression_accuracy']:.2f}%, "
                   f"lm_error={val_metrics['landmark_error']:.4f}")

        # Get uncertainty weights
        uncertainty_weights = {
            'expression_weight': torch.exp(-model.log_var_expression).item(),
            'landmark_weight': torch.exp(-model.log_var_landmark).item()
        }
        logger.info(f"  Weights: expr={uncertainty_weights['expression_weight']:.4f}, "
                   f"lm={uncertainty_weights['landmark_weight']:.4f}")

        # Log metrics
        metrics_logger.log_epoch(epoch, train_metrics, val_metrics, lr, uncertainty_weights)

        # Step scheduler
        if scheduler:
            scheduler.step()

        # Save best model
        if val_metrics['expression_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['expression_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'best_model.pt')
            logger.info(f"  New best model saved (val_acc={best_val_acc:.2f}%)")

        # Periodic checkpoint
        checkpoint_interval = config.get('checkpoint_interval', 10)
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    torch.save({
        'epoch': max_epochs,
        'model_state_dict': model.state_dict(),
        'config': config
    }, output_dir / 'final_model.pt')

    # Save final results
    results = {
        'best_val_accuracy': best_val_acc,
        'final_epoch': max_epochs,
        'output_dir': str(output_dir)
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    metrics_logger.close()
    logger.info(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
