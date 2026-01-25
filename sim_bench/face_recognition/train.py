"""
Training script for face recognition with ArcFace loss.

Trains a ResNet model on CASIA-WebFace with:
- ArcFace loss for discriminative embeddings
- Identity-based train/val split (no identity overlap)
- LFW-style pair verification validation every N epochs

Usage:
    python -m sim_bench.face_recognition.train --config configs/face/resnet50_arcface.yaml
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from sim_bench.models.face_resnet import FaceResNet, create_transform
from sim_bench.datasets.face_dataset import MXNetRecordDataset, PKSampler

from .utils import (
    load_config, set_random_seeds, setup_logging,
    create_optimizer, create_scheduler
)
from .plotting import plot_training_curves, plot_verification_roc
from .validation import validate_lfw, save_verification_results

logger = logging.getLogger(__name__)


class IdentitySubset:
    """Dataset subset that preserves identity information for samplers."""

    def __init__(self, dataset, indices, label_remap: dict = None):
        """
        Args:
            dataset: Original dataset
            indices: Indices to include
            label_remap: Optional dict to remap labels (for contiguous labels)
        """
        self.dataset = dataset
        self.indices = indices
        self.label_remap = label_remap

        # Build samples list for PKSampler compatibility
        self.samples = []
        for new_idx, orig_idx in enumerate(indices):
            offset, orig_label = dataset.samples[orig_idx]
            label = label_remap[orig_label] if label_remap else orig_label
            self.samples.append((offset, label))

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        if self.label_remap:
            orig_label = item['label'].item()
            item['label'] = torch.tensor(self.label_remap[orig_label], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.indices)


def create_identity_split(dataset, val_ratio: float = 0.1, seed: int = 42):
    """
    Split dataset by identity (not by image).

    Ensures train and val sets have NO overlapping identities.

    Args:
        dataset: Dataset with samples attribute [(offset, label), ...]
        val_ratio: Fraction of identities for validation
        seed: Random seed

    Returns:
        train_indices, val_indices, train_label_remap, val_label_remap
    """
    # Group indices by label
    label_to_indices = {}
    for idx, (offset, label) in enumerate(dataset.samples):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # Split identities
    all_labels = list(label_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(all_labels)

    n_val_ids = max(1, int(len(all_labels) * val_ratio))
    val_labels = set(all_labels[:n_val_ids])
    train_labels = set(all_labels[n_val_ids:])

    logger.info(f"Identity split: {len(train_labels)} train, {len(val_labels)} val identities")

    # Collect indices
    train_indices = []
    val_indices = []

    for label, indices in label_to_indices.items():
        if label in train_labels:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)

    # Create label remapping for contiguous labels (required for classification head)
    train_label_remap = {old: new for new, old in enumerate(sorted(train_labels))}
    val_label_remap = {old: new for new, old in enumerate(sorted(val_labels))}

    logger.info(f"Samples: {len(train_indices)} train, {len(val_indices)} val")

    return train_indices, val_indices, train_label_remap, val_label_remap


def create_dataloaders(config: dict):
    """Create train/val dataloaders with identity-based split."""
    transform_config = config.get('transform', {})
    train_transform = create_transform(transform_config, is_train=True)
    val_transform = create_transform(transform_config, is_train=False)

    # Load full dataset
    full_dataset = MXNetRecordDataset(
        rec_path=config['data']['rec_path'],
        transform=train_transform,
        num_classes=config['model'].get('num_classes')
    )

    # Split by identity
    val_ratio = config['data'].get('val_ratio', 0.1)
    seed = config.get('seed', 42)
    train_idx, val_idx, train_remap, val_remap = create_identity_split(
        full_dataset, val_ratio, seed
    )

    # Update num_classes in config for training set
    num_train_classes = len(train_remap)
    config['model']['num_classes'] = num_train_classes
    logger.info(f"Training with {num_train_classes} classes")

    # Create subset datasets
    train_dataset = IdentitySubset(full_dataset, train_idx, train_remap)

    # Val dataset with val transform
    val_full = MXNetRecordDataset(
        rec_path=config['data']['rec_path'],
        transform=val_transform
    )
    val_dataset = IdentitySubset(val_full, val_idx, val_remap)

    # Dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    pin_memory = config['device'] == 'cuda'

    # Check for PK sampling
    sampler_config = config['data'].get('sampler', {})
    use_pk = sampler_config.get('type', 'random') == 'pk'

    if use_pk:
        p = sampler_config.get('p_identities', 16)
        k = sampler_config.get('k_images', 4)
        pk_sampler = PKSampler(train_dataset, p_identities=p, k_images=k, seed=seed)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=pk_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
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

    return train_loader, val_loader, val_transform


def train_epoch(model, loader, optimizer, device, config, warmup_scheduler=None):
    """
    Train for one epoch.

    Returns:
        avg_loss, accuracy (computed once at end of epoch)
    """
    model.train()
    log_interval = config.get('log_interval', 100)
    max_batches = config.get('max_batches')  # For sanity check / debugging

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader, 1):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward with labels for ArcFace margin
        logits = model(images, labels)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        # Accumulate for epoch-end accuracy
        with torch.no_grad():
            # Use inference mode (no margin) for accuracy
            eval_logits = model(images)
            correct += (eval_logits.argmax(1) == labels).sum().item()
            total += labels.numel()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}, lr={current_lr:.6f}")

        if max_batches and batch_idx >= max_batches:
            break

    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate_classification(model, loader, device, max_batches=None):
    """Evaluate classification accuracy on validation set."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
            total_loss += loss.item()
            num_batches += 1

            if max_batches and batch_idx >= max_batches:
                break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                config, output_dir, warmup_scheduler=None, val_transform=None):
    """Main training loop."""
    device = config['device']
    max_epochs = config['training']['max_epochs']
    warmup_epochs = config['training'].get('scheduler', {}).get('warmup_epochs', 0)
    val_interval = config.get('verification_interval', 5)

    # LFW validation setup
    lfw_path = config['data'].get('lfw_bin')
    if lfw_path:
        lfw_path = Path(lfw_path)
        if not lfw_path.exists():
            logger.warning(f"LFW bin not found: {lfw_path}")
            lfw_path = None

    best_lfw_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lfw_acc': [],
        'learning_rate': []
    }

    for epoch in range(max_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\nEpoch {epoch + 1}/{max_epochs} (lr={current_lr:.6f})")

        # Train
        use_warmup = warmup_scheduler is not None and epoch < warmup_epochs
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, config,
            warmup_scheduler=warmup_scheduler if use_warmup else None
        )
        logger.info(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")

        # Classification validation
        max_batches = config.get('max_batches')
        val_loss, val_acc = evaluate_classification(model, val_loader, device, max_batches)
        logger.info(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")

        # LFW verification validation (every N epochs)
        lfw_acc = None
        if lfw_path and (epoch + 1) % val_interval == 0:
            logger.info(f"  Running LFW verification...")
            result = validate_lfw(model, lfw_path, device, val_transform)
            lfw_acc = result.accuracy
            logger.info(f"  LFW:   acc={lfw_acc:.2f}%, threshold={result.threshold:.4f}")

            # Save detailed results
            save_verification_results(
                result,
                output_dir / f'lfw_epoch_{epoch + 1}.json'
            )

            # Save ROC curve
            plot_verification_roc(
                result.fpr, result.tpr, result.auc_score,
                output_dir / f'lfw_roc_epoch_{epoch + 1}.png'
            )

            # Save best model based on LFW accuracy
            if lfw_acc > best_lfw_acc:
                best_lfw_acc = lfw_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'lfw_acc': lfw_acc,
                    'config': config
                }, output_dir / 'best_model.pt')
                logger.info(f"  New best LFW accuracy: {best_lfw_acc:.2f}%")

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lfw_acc'].append(lfw_acc)
        history['learning_rate'].append(current_lr)

        # Checkpoint
        checkpoint_interval = config.get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    # Save final history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_lfw_acc, history


def main():
    parser = argparse.ArgumentParser(description='Train Face Recognition')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup output directory
    output_dir_str = config.get('output_dir')
    if output_dir_str:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path(f"outputs/face_recognition/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(output_dir)

    # Setup logging
    setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Set seeds
    set_random_seeds(config.get('seed', 42))

    # Create dataloaders first (updates num_classes in config)
    logger.info("Creating dataloaders...")
    train_loader, val_loader, val_transform = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model with updated num_classes
    logger.info("Creating model...")
    model = FaceResNet(config['model']).to(config['device'])
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler, warmup_scheduler = create_scheduler(
        optimizer, config, num_batches_per_epoch=len(train_loader)
    )

    # Train
    logger.info("Starting training...")
    best_acc, history = train_model(
        model, train_loader, val_loader, optimizer, scheduler,
        config, output_dir, warmup_scheduler, val_transform
    )

    # Plot curves
    plot_training_curves(history, output_dir)

    # Final results
    results = {
        'best_lfw_acc': best_acc,
        'final_epoch': len(history['train_loss']),
        'num_classes': config['model']['num_classes']
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTraining complete. Best LFW accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
