"""
SIMPLEST POSSIBLE TEST - Run this to check file determinism.

Usage:
    python test_file_determinism.py
"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


def test_external_dataloader():
    """Test if external dataloader returns same files with same seed."""

    # Configuration
    seed = 42
    image_root = r'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs'
    external_path = r'D:\Projects\Series-Photo-Selection'
    batch_size = 8

    # Add external path
    if external_path not in sys.path:
        sys.path.insert(0, external_path)

    from data.dataloader import MyDataset
    from sim_bench.datasets.dataloader_factory import DataLoaderFactory

    print("="*70)
    print("TESTING FILE DETERMINISM")
    print("="*70)

    # Run 1
    print("\n=== RUN 1 ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data1 = MyDataset(train=True, image_root=image_root, seed=seed)
    factory1 = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=seed)
    train_loader1, _, _ = factory1.create_from_external(train_data1, train_data1, None)

    batch1 = next(iter(train_loader1))
    files1_run1 = batch1['image1']

    print(f"Batch size: {len(files1_run1)}")
    print(f"First 3 files:")
    for i in range(min(3, len(files1_run1))):
        print(f"  [{i}] {files1_run1[i]}")

    # Run 2 - FRESH START (same as if you restarted Python)
    print("\n=== RUN 2 (FRESH) ===")
    torch.manual_seed(seed)  # Reset seed
    np.random.seed(seed)     # Reset seed

    train_data2 = MyDataset(train=True, image_root=image_root, seed=seed)
    factory2 = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=seed)
    train_loader2, _, _ = factory2.create_from_external(train_data2, train_data2, None)

    batch2 = next(iter(train_loader2))
    files1_run2 = batch2['image1']

    print(f"Batch size: {len(files1_run2)}")
    print(f"First 3 files:")
    for i in range(min(3, len(files1_run2))):
        print(f"  [{i}] {files1_run2[i]}")

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    if files1_run1 == files1_run2:
        print("✅ FILES MATCH!")
        print("\nThe dataloader IS deterministic!")
        print("\nIf your results still differ, the issue is:")
        print("  - Image transforms (random crops/flips)")
        print("  - Model initialization")
        print("  - CUDA non-determinism")
    else:
        print("❌ FILES DIFFER!")
        print("\nThe dataloader is NON-DETERMINISTIC!")

        # Show first difference
        for i, (f1, f2) in enumerate(zip(files1_run1, files1_run2)):
            if f1 != f2:
                print(f"\nFirst difference at index {i}:")
                print(f"  Run1: {f1}")
                print(f"  Run2: {f2}")
                break

        print("\nPossible causes:")
        print("  1. External dataset doesn't respect seed")
        print("  2. External dataset has internal shuffling")
        print("  3. DataLoader shuffle is randomizing")

    print("="*70 + "\n")


if __name__ == '__main__':
    test_external_dataloader()
