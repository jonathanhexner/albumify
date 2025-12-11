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
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image

from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker
from sim_bench.quality_assessment.trained_models.phototriage_multifeature import compute_pairwise_accuracy

logger = logging.getLogger(__name__)


class EndToEndPairDataset(Dataset):
    """Dataset that loads raw images for end-to-end training."""
    
    def __init__(self, pairs_df: pd.DataFrame, image_dir: str, transform):
        self.pairs_df = pairs_df
        self.image_dir = Path(image_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        
        img1_path = self.image_dir / row['image1']
        img2_path = self.image_dir / row['image2']
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        
        return {
            'img1': img1_tensor,
            'img2': img2_tensor,
            'winner': torch.tensor(int(row['winner']), dtype=torch.long),
            'image1': row['image1'],
            'image2': row['image2']
        }


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_optimizer(model, config):
    """Create optimizer from config."""
    opt_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']

    if opt_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['training']['momentum'],
            weight_decay=wd
        )
    else:  # adamw (default)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def create_model(config, output_dir):
    """Create Siamese CNN + MLP ranker from config dict."""
    return SiameseCNNRanker(config['model']).to(config['device'])


def train_epoch(model, loader, optimizer, device, log_interval=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for batch_idx, batch in enumerate(loader, 1):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        winners = batch['winner'].to(device)

        log_probs = model(img1, img2)
        loss = F.nll_loss(log_probs, 1 - winners)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = compute_pairwise_accuracy(log_probs, winners)
        
        total_loss += batch_loss
        total_acc += batch_acc

        if batch_idx % log_interval == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)}: loss={batch_loss:.4f}, acc={batch_acc:.3f}")

    return total_loss / len(loader), total_acc / len(loader)


def evaluate(model, loader, device, log_interval=10):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            winners = batch['winner'].to(device)

            log_probs = model(img1, img2)
            loss = F.nll_loss(log_probs, 1 - winners)

            batch_loss = loss.item()
            batch_acc = compute_pairwise_accuracy(log_probs, winners)
            
            total_loss += batch_loss
            total_acc += batch_acc

            if batch_idx % log_interval == 0:
                logger.info(f"  Eval Batch {batch_idx}/{len(loader)}: loss={batch_loss:.4f}, acc={batch_acc:.3f}")

    return total_loss / len(loader), total_acc / len(loader)


def load_data(config):
    """Load and split PhotoTriage data."""
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

    logger.info(f"Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return data, train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, data, transform, batch_size):
    """Create PyTorch data loaders."""
    train_dataset = EndToEndPairDataset(train_df, data.train_val_img_dir, transform)
    val_dataset = EndToEndPairDataset(val_df, data.train_val_img_dir, transform)
    test_dataset = EndToEndPairDataset(test_df, data.train_val_img_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, optimizer, config, output_dir):
    """Training loop with early stopping."""
    best_val_acc = 0.0
    patience = 0
    device = config['device']
    log_interval = config.get('log_interval', 10)
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(config['training']['max_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['max_epochs']}:")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, log_interval)
        logger.info(f"  Train: loss={train_loss:.4f}, acc={train_acc:.3f}")
        
        val_loss, val_acc = evaluate(model, val_loader, device, log_interval)
        logger.info(f"  Val: loss={val_loss:.4f}, acc={val_acc:.3f}")
        
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
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output-dir', default=None, help='Override output directory')
    parser.add_argument('--quick-experiment', type=float, default=None, help='Use fraction of data')
    args = parser.parse_args()

    # Load and override config
    config = load_config(args.config)
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.quick_experiment:
        config['data']['quick_experiment'] = args.quick_experiment

    # Setup output directory
    output_dir = Path(
        config.get('output_dir') or
        f"outputs/siamese_e2e/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    logger.info(f"Training end-to-end {config['model']['cnn_backbone']} | Output: {output_dir}")

    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load data
    data, train_df, val_df, test_df = load_data(config)

    # Create model
    model = create_model(config, output_dir)
    transform = model.preprocess

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, data, transform,
        config['training']['batch_size']
    )

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Train
    best_val_acc, history = train_model(model, train_loader, val_loader, optimizer, config, output_dir)

    # Test
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, config['device'])

    logger.info(f"Test accuracy: {test_acc:.3f}")

    # Save final results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'test_acc': test_acc,
            'test_loss': test_loss,
            'best_val_acc': best_val_acc,
            'final_epoch': len(history['train_loss'])
        }, f, indent=2)
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        epochs = range(1, len(history['train_loss']) + 1)
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
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")


if __name__ == '__main__':
    main()
