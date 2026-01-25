"""
Training script for multitask face pretraining (landmarks + expression).

Trains ResNet backbone on AffectNet with two tasks:
- Expression classification (8 classes)
- Landmark regression (5-10 key points)

Uses uncertainty weighting for automatic loss balancing.

Usage:
    python -m sim_bench.training.phase2_pretraining.train_multitask --config configs/face/multitask_affectnet.yaml
"""

import argparse
import yaml
import logging
import json
import random
import csv
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from sim_bench.training.phase2_pretraining.multitask_model import MultitaskFaceModel
from PIL import Image
import torchvision.transforms.functional as TF

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

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")


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
        # Prepare row
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

        # Write to CSV
        self._write_csv(row)

        # Write to TensorBoard
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


def create_model(config: dict) -> MultitaskFaceModel:
    """Create multitask model from config."""
    model = MultitaskFaceModel(config)
    device = config['device']
    return model.to(device)


def create_optimizer(model: MultitaskFaceModel, config: dict):
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
        logger.info(f"Using differential LR: backbone={base_lr}, heads={base_lr * 10}")
    else:
        param_groups = model.parameters()
        logger.info(f"Using single LR: {base_lr}")

    if opt_name == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=wd)
    else:
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)


def create_scheduler(optimizer, config: dict, num_batches_per_epoch: int = None):
    """Create learning rate scheduler with optional warmup."""
    scheduler_cfg = config['training'].get('scheduler')
    if scheduler_cfg is None:
        return None, None

    scheduler_type = scheduler_cfg.get('type', 'step')
    warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)

    if scheduler_type == 'step':
        step_size = scheduler_cfg.get('step_size', 10)
        gamma = scheduler_cfg.get('gamma', 0.1)
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = scheduler_cfg.get('T_max', config['training']['max_epochs'])
        eta_min = scheduler_cfg.get('eta_min', 0)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return None, None

    warmup_scheduler = None
    if warmup_epochs > 0 and num_batches_per_epoch is not None:
        warmup_steps = warmup_epochs * num_batches_per_epoch
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        logger.info(f"Using {warmup_epochs} epoch warmup ({warmup_steps} steps)")

    return main_scheduler, warmup_scheduler


def create_dataloaders(config: dict) -> tuple:
    """Create train/val dataloaders."""
    import sys
    
    data_cfg = config['data']
    batch_size = config['training']['batch_size']
    num_workers = data_cfg.get('num_workers', 4)
    
    # Windows multiprocessing fix: use 0 workers to avoid hangs
    if sys.platform == 'win32' and num_workers > 0:
        logger.warning(f"Windows detected: setting num_workers=0 to avoid multiprocessing issues")
        num_workers = 0
    
    # CPU training: reduce batch size if too large to avoid crashes
    if config['device'] == 'cpu' and batch_size > 32:
        logger.warning(f"CPU training with batch_size={batch_size} may cause crashes. Consider reducing to 16-32.")
    
    pin_memory = config['device'] == 'cuda' and num_workers > 0

    train_dir = Path(data_cfg['train_dir'])
    val_dir = Path(data_cfg.get('val_dir')) if data_cfg.get('val_dir') else None
    landmarks_cache = Path(data_cfg['landmarks_cache']) if data_cfg.get('landmarks_cache') else None
    val_split = data_cfg.get('val_split', 0.1)  # Default 10% for validation

    train_transform = create_affectnet_transform(config)
    val_transform = create_val_transform(config)

    # Check if we need to split train data
    use_split = val_dir is None or val_dir == train_dir

    if use_split:
        # Load all data with train transform first, then split
        full_dataset = AffectNetDataset(
            train_dir,
            landmarks_cache=landmarks_cache,
            transform=None,  # Apply transforms later
            num_landmarks=config['model'].get('num_landmarks', 5),
            horizontal_flip_prob=0.0  # Handle separately
        )

        # Split indices
        n_total = len(full_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        indices = list(range(n_total))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        logger.info(f"Split dataset: {n_train} train, {n_val} val (from {n_total} total)")

        # Create subset datasets with proper transforms
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform,
                                        get_horizontal_flip_prob(config))
        val_dataset = TransformSubset(full_dataset, val_indices, val_transform, 0.0)
    else:
        train_dataset = AffectNetDataset(
            train_dir,
            landmarks_cache=landmarks_cache,
            transform=train_transform,
            num_landmarks=config['model'].get('num_landmarks', 5),
            horizontal_flip_prob=get_horizontal_flip_prob(config)
        )

        val_dataset = AffectNetDataset(
            val_dir,
            landmarks_cache=landmarks_cache,
            transform=val_transform,
            num_landmarks=config['model'].get('num_landmarks', 5),
            horizontal_flip_prob=0.0
        )

    # Test dataset loading before creating DataLoader
    logger.info("Testing dataset loading...")
    test_sample = train_dataset[0]
    logger.info(f"Test sample keys: {test_sample.keys()}, image shape: {test_sample['image'].shape}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


class TransformSubset:
    """Subset with custom transforms applied."""

    def __init__(self, dataset, indices, transform, horizontal_flip_prob):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.horizontal_flip_prob = horizontal_flip_prob
        self.flip_swap = dataset.flip_swap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, expr_label = self.dataset.samples[real_idx]

        image = Image.open(img_path).convert('RGB')

        landmarks = self.dataset.landmarks.get(img_path)
        has_landmarks = landmarks is not None

        if landmarks is None:
            landmarks = np.zeros((self.dataset.num_landmarks, 2), dtype=np.float32)
        else:
            landmarks = landmarks.copy()

        # Apply synchronized horizontal flip
        if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            if has_landmarks:
                landmarks[:, 0] = 1.0 - landmarks[:, 0]
                landmarks = landmarks[self.flip_swap]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'expression': torch.tensor(expr_label, dtype=torch.long),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'has_landmarks': torch.tensor(has_landmarks, dtype=torch.bool)
        }


def compute_losses(model: MultitaskFaceModel, outputs: dict, targets: dict, device: str, config: dict) -> tuple:
    """Compute expression and landmark losses."""
    expression_logits = outputs['expression_logits']
    landmarks_pred = outputs['landmarks']
    
    expression_loss = F.cross_entropy(expression_logits, targets['expression'])
    
    landmarks_gt = targets['landmarks']
    has_landmarks = targets['has_landmarks']
    
    landmark_loss = F.mse_loss(landmarks_pred, landmarks_gt, reduction='none')
    landmark_loss = landmark_loss.mean(dim=(1, 2))
    
    num_valid = has_landmarks.sum().item()
    if num_valid == 0:
        # No valid landmarks in batch, set loss to 0
        landmark_loss = torch.tensor(0.0, device=landmark_loss.device, requires_grad=True)
    else:
        landmark_loss = (landmark_loss * has_landmarks.float()).sum() / num_valid
    
    if model.use_uncertainty_weighting:
        total_loss, weighted_losses = model.uncertainty([expression_loss, landmark_loss])
        return total_loss, expression_loss, landmark_loss, weighted_losses
    else:
        lambda_expr = config.get('training', {}).get('expression_weight', 1.0)
        lambda_landmark = config.get('training', {}).get('landmark_weight', 1.0)
        total_loss = lambda_expr * expression_loss + lambda_landmark * landmark_loss
        return total_loss, expression_loss, landmark_loss, [expression_loss, landmark_loss]


def train_epoch(model: MultitaskFaceModel, loader: DataLoader, optimizer, device: str,
                config: dict, epoch: int = None, warmup_scheduler=None) -> dict:
    """Train for one epoch."""
    model.train()
    log_interval = config.get('log_interval', 100)
    max_batches = config.get('max_batches')

    total_loss = 0.0
    expr_correct = 0
    expr_total = 0
    landmark_errors = []
    num_batches = 0

    logger.info(f"Starting training loop, {len(loader)} batches...")

    for batch_idx, batch in enumerate(loader, 1):
        if max_batches and batch_idx > max_batches:
            logger.info(f"Reached max_batches limit ({max_batches}), stopping")
            break

        images = batch['image'].to(device)
        expressions = batch['expression'].to(device)
        landmarks = batch['landmarks'].to(device)
        has_landmarks = batch['has_landmarks'].to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        
        total_loss_val, expr_loss, lm_loss, weighted_losses = compute_losses(
            model, outputs, {
                'expression': expressions,
                'landmarks': landmarks,
                'has_landmarks': has_landmarks
            }, device, config
        )

        # Check for NaN/Inf before backward
        if not torch.isfinite(total_loss_val):
            error_msg = f"Loss is not finite: {total_loss_val.item()}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        total_loss_val.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        with torch.no_grad():
            _, expr_pred = outputs['expression_logits'].max(1)
            expr_correct += expr_pred.eq(expressions).sum().item()
            expr_total += expressions.size(0)

            valid_landmarks = has_landmarks.bool()
            if valid_landmarks.any():
                lm_error = F.mse_loss(
                    outputs['landmarks'][valid_landmarks],
                    landmarks[valid_landmarks],
                    reduction='mean'
                ).item()
                landmark_errors.append(lm_error)

        total_loss += total_loss_val.item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            expr_acc = 100.0 * expr_correct / expr_total
            lm_err = np.mean(landmark_errors) if landmark_errors else 0.0
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"  Batch {batch_idx}/{len(loader)}: "
                f"loss={total_loss_val.item():.4f}, "
                f"expr_loss={expr_loss.item():.4f}, "
                f"lm_loss={lm_loss.item():.4f}, "
                f"expr_acc={expr_acc:.2f}%, "
                f"lm_error={lm_err:.4f}, "
                f"lr={current_lr:.6f}"
            )

    if num_batches == 0:
        logger.error("No batches processed! Returning zero metrics.")
        return {
            'loss': 0.0,
            'expression_accuracy': 0.0,
            'landmark_error': 0.0
        }
    
    avg_loss = total_loss / num_batches
    expr_accuracy = 100.0 * expr_correct / expr_total
    avg_lm_error = np.mean(landmark_errors) if landmark_errors else 0.0
    
    logger.info(f"Epoch metrics - Loss: {avg_loss:.4f}, Expr Acc: {expr_accuracy:.2f}%, LM Error: {avg_lm_error:.4f}")

    return {
        'loss': avg_loss,
        'expression_accuracy': expr_accuracy,
        'landmark_error': avg_lm_error
    }


def evaluate(model: MultitaskFaceModel, loader: DataLoader, device: str, config: dict) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    max_batches = config.get('max_batches')

    total_loss = 0.0
    expr_correct = 0
    expr_total = 0
    landmark_errors = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            if max_batches and batch_idx > max_batches:
                break

            images = batch['image'].to(device)
            expressions = batch['expression'].to(device)
            landmarks = batch['landmarks'].to(device)
            has_landmarks = batch['has_landmarks'].to(device)

            outputs = model(images)
            total_loss_val, expr_loss, lm_loss, _ = compute_losses(
                model, outputs, {
                    'expression': expressions,
                    'landmarks': landmarks,
                    'has_landmarks': has_landmarks
                }, device, config
            )

            _, expr_pred = outputs['expression_logits'].max(1)
            expr_correct += expr_pred.eq(expressions).sum().item()
            expr_total += expressions.size(0)

            valid_landmarks = has_landmarks.bool()
            if valid_landmarks.any():
                lm_error = F.mse_loss(
                    outputs['landmarks'][valid_landmarks],
                    landmarks[valid_landmarks],
                    reduction='mean'
                ).item()
                landmark_errors.append(lm_error)

            total_loss += total_loss_val.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    expr_accuracy = 100.0 * expr_correct / expr_total
    avg_lm_error = np.mean(landmark_errors) if landmark_errors else 0.0

    return {
        'loss': avg_loss,
        'expression_accuracy': expr_accuracy,
        'landmark_error': avg_lm_error
    }


def main():
    import traceback
    
    parser = argparse.ArgumentParser(description='Train multitask face model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    set_random_seeds(config.get('seed'))

    output_dir = config.get('output_dir') or f"outputs/phase2_multitask/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file and console
    setup_logging(output_dir)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    device = config['device']
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")

    model = create_model(config)
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = create_optimizer(model, config)
    train_loader, val_loader = create_dataloaders(config)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    if len(train_loader) == 0:
        logger.error("Train dataloader is empty! Cannot train.")
        print("ERROR: Train dataloader is empty!", file=sys.stderr, flush=True)
        return
    
    if len(val_loader) == 0:
        logger.warning("Val dataloader is empty! Validation will be skipped.")

    num_batches_per_epoch = len(train_loader)
    main_scheduler, warmup_scheduler = create_scheduler(optimizer, config, num_batches_per_epoch)

    # Initialize metrics logger (CSV + TensorBoard)
    metrics_logger = MetricsLogger(output_dir)

    max_epochs = config['training']['max_epochs']
    best_val_acc = 0.0
    checkpoint_interval = config.get('checkpoint_interval', 5)

    for epoch in range(1, max_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{max_epochs}")
        logger.info(f"{'='*50}")

        train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch, warmup_scheduler)
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Expr Acc: {train_metrics['expression_accuracy']:.2f}%, "
            f"LM Error: {train_metrics['landmark_error']:.4f}"
        )

        val_metrics = evaluate(model, val_loader, device, config)
        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Expr Acc: {val_metrics['expression_accuracy']:.2f}%, "
            f"LM Error: {val_metrics['landmark_error']:.4f}"
        )

        # Get uncertainty weights for logging
        uncertainty_weights = None
        if model.use_uncertainty_weighting:
            uncertainty_weights = model.uncertainty.get_weights()
            logger.info(
                f"Uncertainty weights - Expr: {uncertainty_weights['expression_weight']:.4f}, "
                f"LM: {uncertainty_weights['landmark_weight']:.4f}"
            )

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics_logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, uncertainty_weights)

        if main_scheduler and (warmup_scheduler is None or epoch > config['training']['scheduler'].get('warmup_epochs', 0)):
            main_scheduler.step()

        # Save best model
        if val_metrics['expression_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['expression_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['expression_accuracy'],
                'val_loss': val_metrics['loss'],
                'config': config,
            }, output_dir / 'best_model.pt')
            logger.info(f"Saved best model (val_acc={best_val_acc:.2f}%)")

        # Save periodic checkpoint
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['expression_accuracy'],
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
            logger.info(f"Saved checkpoint at epoch {epoch}")

    metrics_logger.close()

    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")
    logger.info(f"Output saved to: {output_dir}")
    logger.info(f"{'='*50}")


if __name__ == '__main__':
    import traceback
    import sys
    
    # Wrap main() to catch and print any exceptions
    # This is the ONLY place we use try/except - at the top level entry point
    # to ensure errors are visible
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user", file=sys.stderr, flush=True)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
