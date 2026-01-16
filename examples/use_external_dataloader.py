"""
Example demonstrating how to use external dataloaders with the Siamese training pipeline.

This example shows how to integrate dataloaders from external sources
(like D:\Projects\Series-Photo-Selection\data\dataloader.py) with the
sim-bench training pipeline.
"""
import sys
from pathlib import Path

# Example: Using external dataloader from Series-Photo-Selection
# Uncomment and adjust the path to match your setup:
# sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')

import torch
from sim_bench.datasets.siamese_dataloaders import create_dataloaders_from_external


def example_with_internal_dataloaders():
    """Standard usage with internal PhotoTriage dataloaders."""
    from sim_bench.datasets.phototriage_data import PhotoTriageData
    from sim_bench.datasets.siamese_dataloaders import create_phototriage_dataloaders
    from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker

    # Load data
    data = PhotoTriageData(
        root_dir='data/phototriage',
        min_agreement=0.7,
        min_reviewers=2
    )

    train_df, val_df, test_df = data.get_series_based_splits(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    # Create model (provides the transform)
    model_config = {
        'cnn_backbone': 'resnet50',
        'mlp_hidden_sizes': [512, 256],
        'freeze_backbone': False
    }
    model = SiameseCNNRanker(model_config)
    transform = model.preprocess

    # Create dataloaders using the standard interface
    train_loader, val_loader, test_loader = create_phototriage_dataloaders(
        data=data,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        transform=transform,
        batch_size=16,
        num_workers=0
    )

    print(f"Created internal dataloaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch format:")
    print(f"  img1 shape: {batch['img1'].shape}")
    print(f"  img2 shape: {batch['img2'].shape}")
    print(f"  winner shape: {batch['winner'].shape}")
    print(f"  image1 (sample): {batch['image1'][0]}")
    print(f"  image2 (sample): {batch['image2'][0]}")


def example_with_external_dataloaders():
    """
    Example usage with external dataloaders (Series-Photo-Selection format).

    External datasets like MyDataset already return the correct dictionary format,
    so they can be used directly with our training pipeline.
    """
    # Import external dataloader
    # NOTE: Uncomment this when you have the path set up correctly
    # from data.dataloader import MyDataset

    # For demonstration, we'll show the pattern:
    print("\nExample with external dataloaders:")
    print("=" * 60)
    print("""
# 1. Import the external dataset
from data.dataloader import MyDataset

# 2. Create external dataset instances
# MyDataset already returns {'img1', 'img2', 'winner', 'image1', 'image2'}
train_data = MyDataset(
    train=True,
    image_root=r'D:\\Similar Images\\automatic_triage_photo_series\\train_val\\train_val_imgs',
    seed=42  # For reproducible data loading
)
val_data = MyDataset(
    train=False,
    image_root=r'D:\\Similar Images\\automatic_triage_photo_series\\train_val\\train_val_imgs',
    seed=42  # For reproducible data loading
)

# 3. Create dataloaders directly - no adapter needed!
from sim_bench.datasets.siamese_dataloaders import create_dataloaders_from_external

train_loader, val_loader = create_dataloaders_from_external(
    external_train_dataset=train_data,
    external_val_dataset=val_data,
    batch_size=8,
    num_workers=0,
    shuffle_train=True
)

# 4. Use in training - the batches have the correct format!
for batch in train_loader:
    img1 = batch['img1']  # Shape: [B, 3, H, W]
    img2 = batch['img2']  # Shape: [B, 3, H, W]
    winner = batch['winner']  # Shape: [B]

    # Train your model...
    logits = model(img1, img2)
    loss = F.cross_entropy(logits, winner)
    # ...
    """)


def example_mixed_usage():
    """
    Example showing how both dataloader types are interchangeable.

    This demonstrates that you can switch between internal and external
    dataloaders without changing your training code.
    """
    print("\nInterchangeability example:")
    print("=" * 60)
    print("""
The key insight is that both dataloader types return batches with the same format:

batch = {
    'img1': torch.Tensor,      # Shape: [B, 3, H, W]
    'img2': torch.Tensor,      # Shape: [B, 3, H, W]
    'winner': torch.Tensor,    # Shape: [B], values 0 or 1
    'image1': str or list,     # Filenames (for diagnostics)
    'image2': str or list,     # Filenames (for diagnostics)
}

This means your training loop works identically with both:

# Works with BOTH internal and external dataloaders!
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        winners = batch['winner'].to(device)

        logits = model(img1, img2)
        loss = F.cross_entropy(logits, winners)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
    """)


def main():
    """Run all examples."""
    print("=" * 60)
    print("Siamese Dataloader Examples")
    print("=" * 60)

    # Show the usage patterns
    example_with_external_dataloaders()
    example_mixed_usage()

    # Uncomment to run the actual internal dataloader example
    # (requires phototriage data to be set up)
    # example_with_internal_dataloaders()

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("""
1. Internal dataloaders (PhotoTriage):
   - Use create_phototriage_dataloaders() or create_dataloaders()
   - Automatically handles PhotoTriageData format

2. External dataloaders (Series-Photo-Selection):
   - Use create_dataloaders_from_external()
   - External datasets (MyDataset) already return the correct format
   - No adapter needed - used directly!

3. Both are interchangeable in training code!
   - Same batch dictionary format
   - Same tensor shapes
   - Can switch between them without changing training logic
    """)


if __name__ == '__main__':
    main()
