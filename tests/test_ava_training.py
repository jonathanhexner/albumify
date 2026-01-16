"""
Sanity test for AVA training pipeline.

Creates synthetic data to verify the entire pipeline works:
- Model creation
- Dataset loading
- Training loop
- Evaluation
- Saving predictions
"""
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from sim_bench.models.ava_resnet import AVAResNet, create_transform
from sim_bench.datasets.ava_dataset import AVADataset, load_ava_labels, create_splits
from sim_bench.training.train_ava_resnet import (
    create_model, create_optimizer, create_dataloaders,
    compute_loss, compute_mean_score, train_epoch, evaluate
)


def create_synthetic_ava_data(temp_dir: Path, num_images: int = 50):
    """Create synthetic AVA.txt and fake images for testing."""
    # Create image directory
    image_dir = temp_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic AVA.txt
    rows = []
    for i in range(num_images):
        image_id = f"{100000 + i}"

        # Random vote distribution (roughly Gaussian around score 5-6)
        votes = np.random.dirichlet(np.ones(10) * 2) * 200
        votes = votes.astype(int)
        votes = np.clip(votes, 1, 100)  # Ensure at least 1 vote per bin

        # AVA.txt format: image_id, challenge_id, votes_1-10, tag1, tag2, challenge_ref
        row = [image_id, 1] + list(votes) + [0, 0, 1]
        rows.append(row)

        # Create a simple colored image (color varies with mean score)
        mean_score = sum((i+1) * v for i, v in enumerate(votes)) / sum(votes)
        color_val = int((mean_score - 1) / 9 * 255)
        img = Image.new('RGB', (256, 256), (color_val, 128, 255 - color_val))
        img.save(image_dir / f"{image_id}.jpg")

    # Save AVA.txt
    ava_txt = temp_dir / "AVA.txt"
    with open(ava_txt, 'w') as f:
        for row in rows:
            f.write(' '.join(str(x) for x in row) + '\n')

    return ava_txt, image_dir


def test_model_creation():
    """Test model can be created with different configs."""
    print("Testing model creation...")

    # Distribution mode
    config = {
        'backbone': 'resnet50',
        'pretrained': True,
        'mlp_hidden_dims': [256],
        'dropout': 0.2,
        'output_mode': 'distribution'
    }
    model = AVAResNet(config)

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    print(f"  Distribution mode: output shape {out.shape} OK")

    # Regression mode
    config['output_mode'] = 'regression'
    model = AVAResNet(config)
    out = model(x)
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    print(f"  Regression mode: output shape {out.shape} OK")

    print("  Model creation: PASSED\n")


def test_transforms():
    """Test transform creation."""
    print("Testing transforms...")

    config = {
        'resize_size': 256,
        'crop_size': 224,
        'augmentation': {
            'horizontal_flip': 0.5,
            'random_crop': True,
            'color_jitter': {'brightness': 0.1, 'contrast': 0.1}
        }
    }

    # Train transform (with augmentation)
    train_tf = create_transform(config, is_train=True)
    img = Image.new('RGB', (300, 400), (128, 128, 128))
    out = train_tf(img)
    assert out.shape == (3, 224, 224), f"Expected (3, 224, 224), got {out.shape}"
    print(f"  Train transform: output shape {out.shape} OK")

    # Val transform (no augmentation)
    val_tf = create_transform(config, is_train=False)
    out = val_tf(img)
    assert out.shape == (3, 224, 224), f"Expected (3, 224, 224), got {out.shape}"
    print(f"  Val transform: output shape {out.shape} OK")

    print("  Transforms: PASSED\n")


def test_dataset():
    """Test dataset loading with synthetic data."""
    print("Testing dataset...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        ava_txt, image_dir = create_synthetic_ava_data(temp_dir, num_images=20)

        # Load labels
        df = load_ava_labels(ava_txt)
        assert len(df) == 20, f"Expected 20 rows, got {len(df)}"
        assert 'mean_score' in df.columns
        print(f"  Loaded {len(df)} labels OK")

        # Create splits
        train_idx, val_idx, test_idx = create_splits(df, 0.6, 0.2, seed=42)
        assert len(train_idx) == 12
        assert len(val_idx) == 4
        assert len(test_idx) == 4
        print(f"  Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test OK")

        # Create dataset
        transform = create_transform({}, is_train=False)
        dataset = AVADataset(df, image_dir, transform, train_idx, 'distribution')

        sample = dataset[0]
        assert 'image' in sample and sample['image'].shape == (3, 224, 224)
        assert 'target' in sample and sample['target'].shape == (10,)
        assert 'image_id' in sample
        print(f"  Sample: image={sample['image'].shape}, target={sample['target'].shape} OK")

    print("  Dataset: PASSED\n")


def test_loss_functions():
    """Test loss computation."""
    print("Testing loss functions...")

    # Distribution mode - KL div
    output = torch.randn(4, 10)
    target = torch.softmax(torch.randn(4, 10), dim=1)
    loss = compute_loss(output, target, 'distribution', 'kl_div')
    assert loss.ndim == 0  # scalar
    print(f"  KL div loss: {loss.item():.4f} OK")

    # Regression mode - MSE
    output = torch.randn(4, 1)
    target = torch.randn(4)
    loss = compute_loss(output, target, 'regression', 'mse')
    assert loss.ndim == 0
    print(f"  MSE loss: {loss.item():.4f} OK")

    print("  Loss functions: PASSED\n")


def test_mean_score_computation():
    """Test mean score computation from model output."""
    print("Testing mean score computation...")

    # Distribution mode
    # Create output that should predict ~5.5 mean
    output = torch.zeros(2, 10)
    output[:, 4] = 10  # High weight on score 5
    output[:, 5] = 10  # High weight on score 6

    mean_scores = compute_mean_score(output, 'distribution')
    assert mean_scores.shape == (2,)
    assert 5.0 < mean_scores[0].item() < 6.0
    print(f"  Distribution mean scores: {mean_scores.tolist()} OK")

    # Regression mode
    output = torch.tensor([[7.5], [3.2]])
    mean_scores = compute_mean_score(output, 'regression')
    assert mean_scores.shape == (2,)
    assert abs(mean_scores[0].item() - 7.5) < 0.01
    print(f"  Regression mean scores: {mean_scores.tolist()} OK")

    print("  Mean score computation: PASSED\n")


def test_full_training_loop():
    """Test full training loop with synthetic data."""
    print("Testing full training loop...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        ava_txt, image_dir = create_synthetic_ava_data(temp_dir, num_images=30)
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        config = {
            'data': {
                'ava_txt': str(ava_txt),
                'image_dir': str(image_dir),
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'num_workers': 0
            },
            'model': {
                'backbone': 'resnet50',
                'pretrained': True,
                'mlp_hidden_dims': [64],  # Small for speed
                'dropout': 0.1,
                'output_mode': 'distribution'
            },
            'transform': {
                'resize_size': 256,
                'crop_size': 224,
                'augmentation': {'horizontal_flip': 0.5}
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 0.001,
                'differential_lr': True,
                'optimizer': 'adamw',
                'weight_decay': 0.0001,
                'loss_type': 'kl_div'
            },
            'device': 'cpu',
            'seed': 42,
            'log_interval': 5,
            'save_val_predictions': True
        }

        # Create components
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        train_loader, val_loader, test_loader = create_dataloaders(config)

        print(f"  Created model, optimizer, dataloaders OK")
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Run one training epoch
        train_loss, train_preds, train_gts = train_epoch(
            model, train_loader, optimizer, config['device'], config
        )
        print(f"  Train epoch: loss={train_loss:.4f} OK")

        # Run evaluation
        val_loss, val_spearman, _, _, _, _, _ = evaluate(
            model, val_loader, config['device'], config, output_dir, epoch=0
        )
        print(f"  Val epoch: loss={val_loss:.4f}, spearman={val_spearman:.4f} OK")

        # Check predictions file was saved
        pred_file = output_dir / 'predictions' / 'val_epoch_000.parquet'
        assert pred_file.exists(), f"Predictions file not found: {pred_file}"
        pred_df = pd.read_parquet(pred_file)
        assert 'image_id' in pred_df.columns
        assert 'pred_mean' in pred_df.columns
        assert 'gt_mean' in pred_df.columns
        print(f"  Predictions saved: {len(pred_df)} rows OK")

    print("  Full training loop: PASSED\n")


def run_all_tests():
    """Run all sanity tests."""
    print("=" * 60)
    print("AVA Training Pipeline - Sanity Tests")
    print("=" * 60 + "\n")

    test_model_creation()
    test_transforms()
    test_dataset()
    test_loss_functions()
    test_mean_score_computation()
    test_full_training_loop()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
