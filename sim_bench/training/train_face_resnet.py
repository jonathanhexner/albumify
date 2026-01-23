"""
Training script for face classification with ArcFace loss.

Trains a ResNet model with ArcFace loss on face datasets like CASIA-WebFace.
The resulting model produces embeddings for face verification/identification.

Usage:
    python -m sim_bench.training.train_face_resnet --config configs/face/resnet50_arcface.yaml
"""
import argparse
import yaml
import logging
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sim_bench.models.face_resnet import FaceResNet, create_transform
from sim_bench.datasets.face_dataset import create_face_dataset, create_train_val_split

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


def create_model(config: dict) -> FaceResNet:
    """Create FaceResNet model from config."""
    model = FaceResNet(config['model'])
    device = config['device']
    return model.to(device)


def create_optimizer(model: FaceResNet, config: dict):
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


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler if configured."""
    scheduler_cfg = config['training'].get('scheduler')
    if scheduler_cfg is None:
        return None

    scheduler_type = scheduler_cfg.get('type', 'step')

    if scheduler_type == 'step':
        step_size = scheduler_cfg.get('step_size', 10)
        gamma = scheduler_cfg.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'multistep':
        milestones = scheduler_cfg.get('milestones', [10, 20, 30])
        gamma = scheduler_cfg.get('gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = scheduler_cfg.get('T_max', config['training']['max_epochs'])
        eta_min = scheduler_cfg.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return None


def create_dataloaders(config: dict) -> tuple:
    """Create train/val dataloaders."""
    # Create transforms
    transform_config = config.get('transform', {})
    train_transform = create_transform(transform_config, is_train=True)
    val_transform = create_transform(transform_config, is_train=False)

    # Create full dataset with train transform (we'll subset later)
    full_dataset = create_face_dataset(config['data'], transform=train_transform)

    # Create train/val split
    val_ratio = config['data'].get('val_ratio', 0.1)
    seed = config.get('seed', 42)
    train_idx, val_idx = create_train_val_split(full_dataset, val_ratio, seed)
    logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val samples")

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_idx)

    # Create val dataset with val transform
    val_full = create_face_dataset(config['data'], transform=val_transform)
    val_dataset = Subset(val_full, val_idx)

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    pin_memory = config['device'] == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Important for BatchNorm stability
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def train_epoch(model: FaceResNet, loader: DataLoader, optimizer,
                device: str, config: dict, epoch: int = None) -> tuple:
    """
    Train for one epoch.

    Args:
        epoch: Current epoch number (for telemetry logging)

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    log_interval = config.get('log_interval', 100)
    max_batches = config.get('max_batches')  # For quick testing

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader, 1):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass with labels for ArcFace margin
        logits = model(images, labels)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        # Collect telemetry after optimizer.step()
        from sim_bench.telemetry.face_telemetry import collect_telemetry
        collect_telemetry(model, optimizer, batch_idx, epoch, device, config)

        # Compute accuracy using logits WITHOUT margin (inference mode)
        # The training logits have ArcFace margin which penalizes correct class
        with torch.no_grad():
            eval_logits = model(images)  # No labels = no margin
            _, predicted = eval_logits.max(1)

        total_loss += loss.item()
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        if batch_idx % log_interval == 0:
            acc = 100.0 * correct / total
            logger.info(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}, acc={acc:.2f}%")

        if max_batches and batch_idx >= max_batches:
            logger.info(f"  Stopping after {max_batches} batches (max_batches limit)")
            break

    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model: FaceResNet, loader: DataLoader, device: str, config: dict,
             output_dir: Path = None, epoch: int = None) -> tuple:
    """
    Evaluate model on validation set.

    Returns:
        (avg_loss, accuracy, top5_accuracy)
    """
    model.eval()
    max_batches = config.get('max_batches')

    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass without labels (inference mode)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            # Top-1 accuracy
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)
            total_loss += loss.item()
            num_batches += 1

            if max_batches and batch_idx >= max_batches:
                break

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    top5_accuracy = 100.0 * correct_top5 / total

    return avg_loss, accuracy, top5_accuracy


def train_model(model: FaceResNet, train_loader: DataLoader, val_loader: DataLoader,
                optimizer, scheduler, config: dict, output_dir: Path) -> tuple:
    """
    Main training loop with early stopping.

    Returns:
        (best_accuracy, history)
    """
    device = config['device']
    max_epochs = config['training']['max_epochs']
    patience = config['training'].get('early_stop_patience', 10)

    best_acc = -1.0  # Start at -1 to ensure first model is saved
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_top5_acc': [],
        'learning_rate': []
    }

    # Initialize telemetry
    from sim_bench.telemetry.face_telemetry import init_telemetry
    init_telemetry(config)

    for epoch in range(max_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}/{max_epochs} (lr={current_lr:.6f})")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, config, epoch)
        logger.info(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")

        # Validate
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, device, config, output_dir, epoch)
        logger.info(f"  Val: loss={val_loss:.4f}, acc={val_acc:.2f}%, top5={val_top5:.2f}%")

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_top5_acc'].append(val_top5)
        history['learning_rate'].append(current_lr)

        # Check for improvement
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_top5_acc': val_top5,
                'config': config
            }, output_dir / 'best_model.pt')
            logger.info(f"  New best model saved (acc={best_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save checkpoint every N epochs
        checkpoint_interval = config.get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_acc,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_acc, history


def plot_training_curves(history: dict, output_dir: Path):
    """Plot and save training curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[0, 1].plot(epochs, history['val_top5_acc'], 'g--', label='Val Top-5')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning rate plot
    axes[1, 0].plot(epochs, history['learning_rate'], 'k-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')

    # Combined accuracy/loss
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    ax1.plot(epochs, history['val_acc'], 'b-', label='Val Acc')
    ax2.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax2.set_ylabel('Loss', color='r')
    ax1.set_title('Validation Metrics')
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()


def extract_and_save_embeddings(model: FaceResNet, loader: DataLoader,
                                 device: str, output_path: Path):
    """Extract embeddings from trained model and save to file."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label']

            embeddings = model.extract_embedding(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    np.savez(output_path, embeddings=embeddings, labels=labels)
    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Face ResNet with ArcFace')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup output directory
    output_dir_str = config.get('output_dir')
    if output_dir_str:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path(f"outputs/face/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(output_dir)

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
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        logger.info(f"Using scheduler: {config['training']['scheduler']['type']}")

    # Train
    logger.info("Starting training...")
    best_acc, history = train_model(model, train_loader, val_loader, optimizer, scheduler, config, output_dir)

    # Plot curves
    plot_training_curves(history, output_dir)

    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract embeddings from validation set
    if config.get('save_embeddings', False):
        logger.info("Extracting embeddings...")
        extract_and_save_embeddings(model, val_loader, config['device'],
                                    output_dir / 'val_embeddings.npz')

    # Save final results
    results = {
        'best_val_acc': best_acc,
        'best_val_top5_acc': checkpoint.get('val_top5_acc', 0),
        'final_epoch': len(history['train_loss']),
        'num_classes': config['model']['num_classes']
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete. Best val accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
