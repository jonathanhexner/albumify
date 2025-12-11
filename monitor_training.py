"""
Monitor training progress by watching output files.

Usage:
    python monitor_training.py [output_dir]

Default output_dir: outputs/phototriage_binary_test
"""

import sys
import time
from pathlib import Path
import json

def monitor_training(output_dir: str = "outputs/phototriage_binary_test"):
    """Monitor training progress."""
    output_path = Path(output_dir)

    print(f"Monitoring training in: {output_path}")
    print("=" * 60)

    # Check for embeddings cache
    cache_file = output_path / "embeddings_cache.pkl"
    if cache_file.exists():
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"✓ Embeddings cache exists: {size_mb:.1f} MB")
    else:
        print("⏳ Computing embeddings (this takes ~30 min on CPU)...")

    # Check for splits
    train_csv = output_path / "train_pairs.csv"
    val_csv = output_path / "val_pairs.csv"
    test_csv = output_path / "test_pairs.csv"

    if train_csv.exists():
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()
        test_df = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame()

        print(f"✓ Data splits created:")
        print(f"  - Train: {len(train_df)} pairs")
        print(f"  - Val: {len(val_df)} pairs")
        print(f"  - Test: {len(test_df)} pairs")

    # Check for model checkpoints
    best_model = output_path / "best_model.pt"
    final_model = output_path / "final_model.pt"

    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 * 1024)
        print(f"✓ Best model saved: {size_mb:.1f} MB")

    if final_model.exists():
        size_mb = final_model.stat().st_size / (1024 * 1024)
        print(f"✓ Final model saved: {size_mb:.1f} MB")

    # Check for test results
    results_file = output_path / "test_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        print(f"\n{'='*60}")
        print("FINAL TEST RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {100*results['accuracy']:.2f}%")
        print(f"Precision: {100*results['precision']:.2f}%")
        print(f"Recall: {100*results['recall']:.2f}%")
        print(f"F1 Score: {100*results['f1']:.2f}%")
        print(f"{'='*60}")
    else:
        print("\n⏳ Training in progress...")
        print("\nTo see live output, use:")
        print("  Python: Use BashOutput tool with shell ID: c5e517")
        print("  Or check files in:", output_path)

    # Check for training curves plot
    plot_file = output_path / "training_curves.png"
    if plot_file.exists():
        print(f"\n✓ Training curves plot saved: {plot_file}")

    print(f"\n{'='*60}")
    print("Output directory contents:")
    print(f"{'='*60}")
    if output_path.exists():
        for item in sorted(output_path.iterdir()):
            if item.is_file():
                size = item.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"  {item.name:40s} {size_str:>12s}")
    else:
        print("  Directory not created yet")

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/phototriage_binary_test"
    monitor_training(output_dir)
