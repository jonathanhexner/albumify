"""
End-to-end Siamese CNN training.

Trains a Siamese CNN with shared weights for pairwise image quality comparison.
Slower than frozen features mode since images are processed through CNN each batch.

Usage:
    python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
    python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/vgg16.yaml --quick-experiment 0.1
"""
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import copy
import json
import random
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.datasets.siamese_dataloaders import get_dataset_from_loader
from sim_bench.datasets.transform_factory import create_transform
from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker
from sim_bench.utils.model_inspection import inspect_model_output
from sim_bench.training.model_comparison import dump_model_to_csv

logger = logging.getLogger(__name__)


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for torch, numpy, and Python's random module.
    Optionally sets CUDA seeds if CUDA is available.
    
    Args:
        seed: Integer seed value. If None, seeds are not set.
    """
    if seed is None:
        return
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_batch_metrics(logits, winners):
    """
    Compute loss and accuracy for a batch.

    Returns:
        loss: Cross-entropy loss
        accuracy: Accuracy (fraction correct)
        num_correct: Number of correct predictions
        batch_size: Total number of samples
    """
    loss = F.cross_entropy(logits, winners)
    preds = logits.argmax(dim=-1)
    num_correct = (preds == winners).sum().item()
    batch_size = len(winners)
    accuracy = num_correct / batch_size
    return loss, accuracy, num_correct, batch_size


def create_optimizer(model, config):
    """Create optimizer with differential learning rates."""
    opt_name = config['training']['optimizer'].lower()
    base_lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']

    # Use differential learning rates: 1x for backbone, 10x for head
    use_diff_lr = config['training'].get('differential_lr', True)

    if use_diff_lr:
        param_groups = [
            {'params': model.get_1x_lr_params(), 'lr': base_lr},
            {'params': model.get_10x_lr_params(), 'lr': base_lr * 10}
        ]
        logger.info(f"Using differential LR: backbone={base_lr}, head={base_lr * 10}")
    else:
        param_groups = model.parameters()
        logger.info(f"Using single LR: {base_lr}")

    if opt_name == 'sgd':
        return torch.optim.SGD(
            param_groups,
            momentum=config['training']['momentum'],
            weight_decay=wd
        )
    else:  # adamw
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)


def create_model(config):
    """
    Create Siamese CNN + MLP ranker from config dict.

    Supports two model types:
    - 'siamese_cnn': Our SiameseCNNRanker implementation
    - 'reference': Reference model from Series-Photo-Selection

    Args:
        config: Configuration dict with 'model_type' key

    Returns:
        Model instance on the configured device
    """
    model_type = config.get('model_type', 'siamese_cnn')
    device = config['device']

    if model_type == 'reference':
        # Load reference model from Series-Photo-Selection
        import sys
        reference_path = config.get('reference_model_path', r'D:\Projects\Series-Photo-Selection')
        if reference_path not in sys.path:
            logger.info(f"Adding reference model path to sys.path: {reference_path}")
            sys.path.insert(0, reference_path)

        from models.ResNet50 import make_network
        model = make_network()
        logger.info("Created reference model from Series-Photo-Selection")

    elif model_type == 'siamese_cnn':
        # Our implementation
        model = SiameseCNNRanker(config['model'])
        logger.info(f"Created SiameseCNNRanker with backbone: {config['model']['cnn_backbone']}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'siamese_cnn' or 'reference'")

    return model.to(device)


def train_epoch(models_dict, optimizers_dict, loader, device, log_interval=10,
                epoch=None, output_dir=None, batch_comparison_interval=None, config=None):
    """
    Train one or more models for one epoch.

    Args:
        models_dict: Dict of {model_name: model} to train. First entry is primary model.
        optimizers_dict: Dict of {model_name: optimizer} matching models_dict
        loader: DataLoader
        device: Device
        log_interval: Log every N batches
        epoch: Current epoch (for logging)
        output_dir: Output directory (for comparison logs)
        batch_comparison_interval: Compare models every N batches (if None, no comparison)

    Returns:
        Dict of {model_name: (avg_loss, avg_acc)} for each model

    Example:
        >>> # Single model
        >>> models = {'main': model}
        >>> optimizers = {'main': optimizer}
        >>> results = train_epoch(models, optimizers, loader, device)
        >>> train_loss, train_acc = results['main']
        >>>
        >>> # Multiple models (with reference)
        >>> models = {'main': model, 'reference': ref_model}
        >>> optimizers = {'main': optimizer, 'reference': ref_optimizer}
        >>> results = train_epoch(models, optimizers, loader, device)
    """
    # Set all models to train mode
    for model in models_dict.values():
        model.train()

    # Track metrics for each model
    model_names = list(models_dict.keys())
    metrics = {
        name: {'total_loss': 0.0, 'total_correct': 0, 'total_samples': 0}
        for name in model_names
    }

    # Track comparison metrics only if we have multiple models
    # (Skipped when len(models_dict) == 1)
    comparison_log = [] if (len(models_dict) > 1 and batch_comparison_interval is not None) else None

    for batch_idx, batch in enumerate(loader, 1):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        winners = batch['winner'].to(device)

        # Train each model
        batch_metrics = {}
        for name, model in models_dict.items():
            optimizer = optimizers_dict[name]

            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass
            logits = model(img1, img2)
            loss, batch_acc, num_correct, batch_size = compute_batch_metrics(logits, winners)

            # Log batch predictions
            from sim_bench.utils.batch_logger import log_batch_predictions
            log_batch_predictions(
                config.get('batch_predictions_path', config['output_dir'] / 'telemetry' / 'batch_predictions.csv'),
                batch_idx, epoch,
                batch['image1'], batch['image2'],
                winners, logits
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Telemetry hook - collect metrics after optimizer.step()
            from sim_bench import telemetry
            telemetry.record(model, optimizer, batch_idx, epoch, device, batch)

            # Track metrics
            batch_loss = loss.item()
            metrics[name]['total_loss'] += batch_loss
            metrics[name]['total_correct'] += num_correct
            metrics[name]['total_samples'] += batch_size

            batch_metrics[name] = {'loss': batch_loss, 'acc': batch_acc}

        # Logging
        if batch_idx % log_interval == 0:
            if len(models_dict) == 1:
                # Single model: simple format
                name = model_names[0]
                logger.info(f"  Batch {batch_idx}/{len(loader)}: "
                          f"loss={batch_metrics[name]['loss']:.4f}, "
                          f"acc={batch_metrics[name]['acc']:.3f}")
            else:
                # Multiple models: compare format
                log_parts = [f"Batch {batch_idx}/{len(loader)}:"]
                for name in model_names:
                    log_parts.append(f"{name}_loss={batch_metrics[name]['loss']:.4f}, "
                                   f"{name}_acc={batch_metrics[name]['acc']:.3f}")
                logger.info("  " + " | ".join(log_parts))

        # Compare models if enabled (only runs when len(models_dict) > 1)
        if comparison_log is not None and batch_idx % batch_comparison_interval == 0:
            from sim_bench.training.model_comparison import compare_model_states

            # Compare first model with all others
            primary_name = model_names[0]
            primary_model = models_dict[primary_name]

            for ref_name in model_names[1:]:  # Empty loop if only 1 model
                ref_model = models_dict[ref_name]

                # Compare only MLP head
                mlp_filter = lambda name: 'mlp' in name or 'fc' in name
                comp = compare_model_states(primary_model.state_dict(), ref_model.state_dict(), mlp_filter)
                comp.update({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'primary_model': primary_name,
                    'reference_model': ref_name,
                    'primary_loss': batch_metrics[primary_name]['loss'],
                    'reference_loss': batch_metrics[ref_name]['loss'],
                    'primary_acc': batch_metrics[primary_name]['acc'],
                    'reference_acc': batch_metrics[ref_name]['acc'],
                    'loss_diff': abs(batch_metrics[primary_name]['loss'] - batch_metrics[ref_name]['loss']),
                    'acc_diff': abs(batch_metrics[primary_name]['acc'] - batch_metrics[ref_name]['acc'])
                })
                comparison_log.append(comp)

    # Save comparison log if we collected any
    if comparison_log:
        comp_dir = Path(output_dir) / "batch_comparisons"
        comp_dir.mkdir(parents=True, exist_ok=True)
        with open(comp_dir / f"epoch_{epoch:03d}.json", 'w') as f:
            json.dump(comparison_log, f, indent=2)

    # Compute and return average metrics for each model
    results = {}
    for name in model_names:
        avg_loss = metrics[name]['total_loss'] / len(loader)
        avg_acc = metrics[name]['total_correct'] / metrics[name]['total_samples'] if metrics[name]['total_samples'] > 0 else 0.0
        results[name] = (avg_loss, avg_acc)

    return results


def evaluate(model, loader, device, output_dir, epoch, split_name,
             inspect_k=6, log_interval=10):
    """
    Evaluate model with comprehensive diagnostics.

    Now only needs the loader - extracts dataset and metadata internally.

    Saves per-epoch diagnostics to diagnose overfitting:
    - Confusion matrix, per-class recalls
    - Sample predictions with probabilities
    - Per-series accuracy breakdown
    - Visual inspection of sample pairs
    """
    from sim_bench.training.diagnostics import (
        save_epoch_metrics, save_per_series_breakdown, inspect_series_pairs
    )

    # Extract dataset and metadata from loader
    dataset = get_dataset_from_loader(loader)

    model.eval()

    # Create directories
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    split_dir = epoch_dir / split_name
    metrics_dir = split_dir / 'metrics'
    inspect_dir = split_dir / 'inspect'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    inspect_dir.mkdir(parents=True, exist_ok=True)

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
    # Accuracy: total correct / total samples (handles variable batch sizes correctly)
    avg_acc = (all_preds == all_winners).mean()

    # Save diagnostics - pass dataset instead of pairs_df
    save_epoch_metrics(all_preds, all_winners, all_logprobs, all_image1, all_image2,
                       dataset, avg_loss, metrics_dir)
    save_per_series_breakdown(all_preds, all_winners, all_image1, all_image2, dataset, metrics_dir)

    # Visual inspection (PhotoTriage only - external datasets don't have series_id metadata)
    if inspect_k > 0 and hasattr(dataset, 'get_dataframe'):
        pairs_df = dataset.get_dataframe()
        series_id = pairs_df['series_id'].iloc[0]
        inspect_series_pairs(model, dataset, device, inspect_dir, series_id, k=inspect_k)

    logger.info(f"  {split_name.capitalize()}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")
    return avg_loss, avg_acc


def create_dataloaders(config, transform, batch_size):
    """
    Create dataloaders - supports both PhotoTriage and external sources.

    This is the ONE function you need to call - it handles everything:
    - Loading data (PhotoTriage or external)
    - Creating datasets
    - Creating DataLoaders

    Just set config['use_external_dataloader'] = True to switch sources!

    Args:
        config: Configuration dict with data source and parameters
        transform: Image transform (only used for PhotoTriage; external has its own)
        batch_size: Batch size for loaders

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Set RNG seeds for deterministic behavior
    # - random.seed() controls dataset shuffling (in make_shuffle_path)
    # - torch.manual_seed() controls DataLoader shuffling (torch.randperm)
    seed = config.get('seed')
    set_random_seeds(seed)

    use_external = config.get('use_external_dataloader', False)
    factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=seed)

    if use_external:
        # External dataloader (already has transforms built-in)
        import sys
        external_path = config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection')
        if external_path not in sys.path:
            logger.info(f"Adding external path to sys.path: {external_path}")
            sys.path.insert(0, external_path)

        from data.dataloader import MyDataset

        # Get image_root - use root_dir if image_root not specified
        image_root = config['data'].get('image_root')
        if image_root is None:
            # Fall back to root_dir if available
            image_root = config['data'].get('root_dir')

        if image_root is None:
            raise ValueError(
                "When using external dataloader, you must specify either 'image_root' or 'root_dir' in config['data']. "
                "Example: data:\n  image_root: D:\\path\\to\\train_val_imgs"
            )

        logger.info(f"Using external dataloader with image_root: {image_root}, seed: {config.get('seed')}")
        train_data = MyDataset(train=True, image_root=image_root, seed=config.get('seed'))
        val_data = MyDataset(train=False, image_root=image_root, seed=config.get('seed'))

        logger.info(f"External data: {len(train_data)} train, {len(val_data)} val")
        return factory.create_from_external(train_data, val_data, None)
    else:
        # PhotoTriage (original)
        data = PhotoTriageData(
            config['data']['root_dir'],
            config['data']['min_agreement'],
            config['data']['min_reviewers']
        )

        train_df, val_df, test_df = data.get_series_based_splits(
            0.8, 0.1, 0.1,
            config['seed'],
            config['data'].get('quick_experiment')
        )

        logger.info(f"PhotoTriage: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return factory.create_from_phototriage(data, train_df, val_df, test_df, transform)


def plot_training_curves(history, output_dir):
    """Plot and save training/validation curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()


def train_model(model, train_loader, val_loader, optimizer, config, output_dir):
    """
    Training loop with early stopping and comprehensive diagnostics.

    Simplified signature - only needs loaders, not DataFrames.
    Removed parameters: train_df, val_df, data, transform
    """
    best_val_acc = 0.0
    patience = 0
    device = config['device']
    log_interval = config.get('log_interval', 10)

    # Per-batch model dumping setup
    batch_comparison_interval = config.get('batch_comparison_interval')
    if batch_comparison_interval is not None:
        logger.info(f"Per-batch model dumping enabled: saving every {batch_comparison_interval} batches")

    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(config['training']['max_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['max_epochs']}:")

        # Train with dictionary interface (single model for now)
        models_dict = {'main': model}
        optimizers_dict = {'main': optimizer}

        results = train_epoch(
            models_dict, optimizers_dict, train_loader, device, log_interval,
            epoch=epoch, output_dir=output_dir,
            batch_comparison_interval=batch_comparison_interval,
            config=config
        )

        train_loss, train_acc = results['main']
        logger.info(f"  Train: loss={train_loss:.4f}, acc={train_acc:.3f}")

        # Comprehensive validation evaluation with diagnostics
        val_loss, val_acc = evaluate(
            model, val_loader, device,
            output_dir, epoch, 'val',
            inspect_k=6, log_interval=log_interval
        )

        # Compute mode gap diagnostic (BatchNorm check)
        from sim_bench.training.diagnostics import compute_mode_gap_diagnostic
        compute_mode_gap_diagnostic(model, train_loader, val_loader, device, output_dir, epoch, train_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, output_dir / 'best_model.pt')
            patience = 0
        else:
            patience += 1
            if patience >= config['training']['early_stop_patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_val_acc, history


def main():
    # Parse arguments first
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output-dir', default=None, help='Override output directory')
    parser.add_argument('--data-dir', default=None, help='Override data root directory')
    parser.add_argument('--device', default=None, help='Override device (cpu/cuda)')
    parser.add_argument('--quick-experiment', type=float, default=None, help='Use fraction of data')
    args = parser.parse_args()

    # Load and override config
    config = load_config(args.config)
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    if args.device:
        config['device'] = args.device
    if args.quick_experiment:
        config['data']['quick_experiment'] = args.quick_experiment

    # Setup output directory
    output_dir = Path(
        config.get('output_dir') or
        f"outputs/siamese_e2e/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (both console and file)
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging to: {log_file}")

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    logger.info(f"Training end-to-end {config['model']['cnn_backbone']} | Output: {output_dir}")

    # Set random seeds
    set_random_seeds(config['seed'])

    # Create model
    model = create_model(config)
    
    # Dump model state for comparison
    dump_model_to_csv(model, output_dir / 'model_state_after_creation.csv')

    # Create transform independently of model
    transform = create_transform(config)

    # Compare with reference if enabled
    if config.get('compare_with_reference', False):
        from sim_bench.training.model_comparison import compare_with_reference_model
        compare_with_reference_model(model, config, output_dir)

    # Create dataloaders for actual training (fresh RNG state)
    train_loader, val_loader, test_loader = create_dataloaders(
        config, transform, config['training']['batch_size']
    )
    # Verify first batch for debugging (creates temporary loaders)
    train_first_batch = next(iter(train_loader))
    val_first_batch = next(iter(val_loader))
    df_train_loader = pd.DataFrame({
        'image1': train_first_batch['image1'],
        'image2': train_first_batch['image2']
    })


    df_val_loader = pd.DataFrame({
        'image1': val_first_batch['image1'],
        'image2': val_first_batch['image2']
    })
    print("First training batch:")
    print(df_train_loader)

    print("\nFirst validation batch:")
    print(df_val_loader)


    # Create optimizer
    optimizer = create_optimizer(model, config)
    set_random_seeds(config['seed'])
    print(f"Random state after optimizer creation: {random.getstate()[1][:5]}")
    print(f"Random state after optimizer creation: {np.random.get_state()[1][:5]}")
    print(f"Random state after optimizer creation: {torch.get_rng_state()[:10]}")
    batch = next(iter(train_loader))
    print("Train loader inspect_model_output iter: ", str(batch['image1']), str(batch['image2']))
    set_random_seeds(config['seed'])
    # Add output_dir to config and initialize telemetry
    config['output_dir'] = output_dir
    from sim_bench import telemetry
    telemetry.init(config, val_loader)
    set_random_seeds(config['seed'])

    # Dump model state before inspection
    dump_model_to_csv(model, output_dir / 'model_state_before_inspection.csv')

    # Inspect model output before training
    logger.info("\nInspecting model output before training...")
    df_inspect = inspect_model_output(
        model, train_loader, config['device'],
        save_path=output_dir / 'initial_model_inspection.csv'
    )
    logger.info(f"Initial batch accuracy: {df_inspect['correct'].mean():.3f}")
    logger.info(f"Sample predictions:\n{df_inspect.head()}")
    set_random_seeds(config['seed'])

    # Set batch predictions path if enabled
    if config.get('log_batch_predictions', False):
        config['batch_predictions_path'] = output_dir / 'telemetry' / 'batch_predictions.csv'
        logger.info(f"Batch prediction logging enabled: {config['batch_predictions_path']}")

    # Train - simplified call with only loaders
    best_val_acc, history = train_model(
        model, train_loader, val_loader, optimizer, config, output_dir
    )

    # Test - comprehensive evaluation on test set (if available)
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_epoch = checkpoint['epoch']

    if test_loader is not None:
        test_loss, test_acc = evaluate(
            model, test_loader, config['device'],
            output_dir, final_epoch, 'test',
            inspect_k=6, log_interval=config.get('log_interval', 10)
        )
        logger.info(f"Test accuracy: {test_acc:.3f}")
    else:
        logger.info("No test set available (external dataloader only has train/val)")
        test_loss, test_acc = None, None

    # Save final results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'test_acc': test_acc,
            'test_loss': test_loss,
            'best_val_acc': best_val_acc,
            'final_epoch': len(history['train_loss'])
        }, f, indent=2)

    # Plot training curves
    plot_training_curves(history, output_dir)


if __name__ == '__main__':
    main()
