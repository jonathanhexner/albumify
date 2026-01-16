"""
Simple script to check if dataloader returns the same files in the same order.

This is the FIRST thing to check - if files differ, nothing else matters.

Usage:
    python check_batch_files.py --config configs/example_external.yaml
    python check_batch_files.py --config configs/example_external.yaml --num-batches 5
"""
import argparse
import yaml
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.datasets.transform_factory import create_transform


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_dataloaders(config, transform, batch_size):
    """Create dataloaders."""
    use_external = config.get('use_external_dataloader', False)
    factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=config.get('seed'))

    if use_external:
        # External dataloader
        external_path = config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection')
        if external_path not in sys.path:
            print(f"Adding external path: {external_path}")
            sys.path.insert(0, external_path)

        from data.dataloader import MyDataset

        image_root = config['data'].get('image_root') or config['data'].get('root_dir')
        if not image_root:
            raise ValueError("Must specify image_root or root_dir")

        print(f"Using external dataloader:")
        print(f"  Path: {external_path}")
        print(f"  Image root: {image_root}")
        print(f"  Seed: {config.get('seed')}")

        train_data = MyDataset(train=True, image_root=image_root, seed=config.get('seed'))
        val_data = MyDataset(train=False, image_root=image_root, seed=config.get('seed'))

        print(f"  Train size: {len(train_data)}")
        print(f"  Val size: {len(val_data)}")

        return factory.create_from_external(train_data, val_data, None)
    else:
        # PhotoTriage
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

        print(f"Using PhotoTriage:")
        print(f"  Train size: {len(train_df)}")
        print(f"  Val size: {len(val_df)}")
        print(f"  Test size: {len(test_df)}")

        return factory.create_from_phototriage(data, train_df, val_df, test_df, transform)


def check_batch_files(config_path, num_batches=3, save_to_file=None):
    """
    Check which files appear in the first N batches.

    Args:
        config_path: Path to config YAML
        num_batches: Number of batches to check
        save_to_file: Optional file to save results
    """
    print("="*70)
    print("BATCH FILE CHECKER")
    print("="*70)

    # Load config
    config = load_config(config_path)
    print(f"\nConfig: {config_path}")
    print(f"Seed: {config.get('seed', 'NOT SET!')}")

    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n✓ Random seeds set to {seed}")

    # Create transform
    transform = create_transform(config)
    print(f"\nTransform type: {config.get('transform_type', 'default')}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, transform, config['training']['batch_size']
    )

    print(f"\n{'='*70}")
    print(f"CHECKING FIRST {num_batches} BATCHES")
    print(f"{'='*70}")

    all_files = []
    batch_summaries = []

    for batch_idx, batch in enumerate(train_loader, 1):
        if batch_idx > num_batches:
            break

        # Get filenames
        if 'image1' not in batch:
            print(f"\n❌ ERROR: Batch has no 'image1' field!")
            print(f"Batch keys: {batch.keys()}")
            return

        files1 = batch['image1']
        files2 = batch['image2']
        winners = batch['winner']

        print(f"\n{'='*70}")
        print(f"BATCH {batch_idx}")
        print(f"{'='*70}")
        print(f"Size: {len(files1)}")
        print(f"\nFirst 5 pairs:")

        for i in range(min(5, len(files1))):
            print(f"  [{i}] winner={winners[i].item()}")
            print(f"      img1: {files1[i]}")
            print(f"      img2: {files2[i]}")

        print(f"\nLast pair:")
        print(f"  [{len(files1)-1}] winner={winners[-1].item()}")
        print(f"      img1: {files1[-1]}")
        print(f"      img2: {files2[-1]}")

        # Store for comparison
        batch_summary = {
            'batch_idx': batch_idx,
            'size': len(files1),
            'files1': list(files1),
            'files2': list(files2),
            'winners': winners.tolist(),
        }
        batch_summaries.append(batch_summary)
        all_files.extend(files1)
        all_files.extend(files2)

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total batches checked: {num_batches}")
    print(f"Total images seen: {len(all_files)}")
    print(f"Unique images: {len(set(all_files))}")

    # Save to file if requested
    if save_to_file:
        import json
        output_file = Path(save_to_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(batch_summaries, f, indent=2)

        print(f"\n✓ Saved batch details to: {output_file}")
        print("\nTo compare two runs:")
        print(f"  1. Run: python check_batch_files.py --config {config_path} --save run1.json")
        print(f"  2. Run: python check_batch_files.py --config {config_path} --save run2.json")
        print(f"  3. Run: python check_batch_files.py --compare run1.json run2.json")

    print(f"\n{'='*70}")
    print("COPY THE ABOVE OUTPUT TO COMPARE WITH ANOTHER RUN")
    print(f"{'='*70}\n")

    return batch_summaries


def compare_batch_files(file1, file2):
    """
    Compare batch files from two different runs.

    Args:
        file1: Path to first batch summary JSON
        file2: Path to second batch summary JSON
    """
    import json

    print("="*70)
    print("COMPARING BATCH FILES")
    print("="*70)

    with open(file1) as f:
        batches1 = json.load(f)

    with open(file2) as f:
        batches2 = json.load(f)

    print(f"\nRun 1: {file1}")
    print(f"Run 2: {file2}")
    print(f"\nBatches in run 1: {len(batches1)}")
    print(f"Batches in run 2: {len(batches2)}")

    all_match = True

    for b1, b2 in zip(batches1, batches2):
        batch_idx = b1['batch_idx']

        print(f"\n{'='*70}")
        print(f"BATCH {batch_idx}")
        print(f"{'='*70}")

        # Check sizes
        if b1['size'] != b2['size']:
            print(f"❌ BATCH SIZE DIFFERS: {b1['size']} vs {b2['size']}")
            all_match = False
            continue

        print(f"Batch size: {b1['size']} ✓")

        # Check files1
        files1_match = b1['files1'] == b2['files1']
        files2_match = b1['files2'] == b2['files2']
        winners_match = b1['winners'] == b2['winners']

        if files1_match:
            print(f"✓ image1 files MATCH")
        else:
            print(f"❌ image1 files DIFFER")
            all_match = False

            # Show first difference
            for i, (f1, f2) in enumerate(zip(b1['files1'], b2['files1'])):
                if f1 != f2:
                    print(f"\n  First difference at index {i}:")
                    print(f"    Run1: {f1}")
                    print(f"    Run2: {f2}")
                    break

        if files2_match:
            print(f"✓ image2 files MATCH")
        else:
            print(f"❌ image2 files DIFFER")
            all_match = False

            # Show first difference
            for i, (f1, f2) in enumerate(zip(b1['files2'], b2['files2'])):
                if f1 != f2:
                    print(f"\n  First difference at index {i}:")
                    print(f"    Run1: {f1}")
                    print(f"    Run2: {f2}")
                    break

        if winners_match:
            print(f"✓ winners MATCH")
        else:
            print(f"❌ winners DIFFER")
            all_match = False

    print(f"\n{'='*70}")
    if all_match:
        print("✓✓✓ ALL BATCHES MATCH! ✓✓✓")
        print("\nFiles are identical. If your results differ, the issue is:")
        print("  - Image transforms (random crops/flips)")
        print("  - Model initialization")
        print("  - CUDA non-determinism")
    else:
        print("❌❌❌ BATCHES DIFFER ❌❌❌")
        print("\nThe dataloader is returning different files!")
        print("Possible causes:")
        print("  - DataLoader shuffling (check shuffle=True)")
        print("  - External dataset internal shuffling/sampling")
        print("  - Seed not being respected")
        print("  - Worker processes (check num_workers)")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Check batch files for determinism')
    parser.add_argument('--config', help='Path to config YAML')
    parser.add_argument('--num-batches', type=int, default=3,
                       help='Number of batches to check')
    parser.add_argument('--save', help='Save batch details to JSON file')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare two batch summary JSON files')
    args = parser.parse_args()

    if args.compare:
        # Compare mode
        compare_batch_files(args.compare[0], args.compare[1])
    elif args.config:
        # Check mode
        check_batch_files(args.config, args.num_batches, args.save)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
