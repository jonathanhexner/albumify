"""
Compare two models trained in parallel by analyzing their batch dumps.

Loads batch dumps from both our model and reference model, then compares:
- Parameter differences (L2 distance, cosine similarity)
- Prediction differences
- Loss/accuracy divergence
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_batch_dumps(dump_dir):
    """Load all batch dumps from directory."""
    dump_path = Path(dump_dir)
    dumps = []

    for epoch_dir in sorted(dump_path.glob("epoch_*")):
        batch_dir = epoch_dir / "batch_dumps"
        if not batch_dir.exists():
            batch_dir = epoch_dir  # Try epoch dir directly

        for dump_file in sorted(batch_dir.glob("batch_*.pt")):
            dump = torch.load(dump_file, map_location='cpu')
            dumps.append({
                'epoch': dump['epoch'],
                'batch': dump['batch'],
                'state_dict': dump['model_state_dict'],
                'loss': dump['loss'],
                'acc': dump['acc'],
                'path': dump_file
            })

    return dumps


def compare_state_dicts(state1, state2, param_filter=None):
    """
    Compare two model state dicts.

    Args:
        state1: First model state dict
        state2: Second model state dict
        param_filter: Function to filter parameter names (e.g., lambda name: 'mlp' in name)

    Returns:
        Dict with comparison metrics
    """
    param_names = [name for name in state1.keys() if name in state2]
    if param_filter:
        param_names = [name for name in param_names if param_filter(name)]

    l2_distances = []
    cos_sims = []
    max_diffs = []

    per_param = {}
    for name in param_names:
        p1 = state1[name].flatten()
        p2 = state2[name].flatten()

        if p1.shape != p2.shape:
            continue

        l2 = torch.norm(p1 - p2).item()
        cos = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
        maxd = torch.max(torch.abs(p1 - p2)).item()

        l2_distances.append(l2)
        cos_sims.append(cos)
        max_diffs.append(maxd)

        per_param[name] = {
            'l2_distance': float(l2),
            'cosine_similarity': float(cos),
            'max_abs_diff': float(maxd)
        }

    return {
        'avg_l2': float(np.mean(l2_distances)) if l2_distances else 0.0,
        'max_l2': float(np.max(l2_distances)) if l2_distances else 0.0,
        'avg_cos': float(np.mean(cos_sims)) if cos_sims else 1.0,
        'min_cos': float(np.min(cos_sims)) if cos_sims else 1.0,
        'per_param': per_param
    }


def compare_dump_sequences(our_dumps, ref_dumps, param_filter=None):
    """
    Compare two sequences of model dumps.

    Returns list of comparisons for matching (epoch, batch) pairs.
    """
    # Create lookup for reference dumps
    ref_lookup = {(d['epoch'], d['batch']): d for d in ref_dumps}

    comparisons = []
    for our_dump in our_dumps:
        key = (our_dump['epoch'], our_dump['batch'])
        if key not in ref_lookup:
            continue

        ref_dump = ref_lookup[key]

        comp = compare_state_dicts(our_dump['state_dict'], ref_dump['state_dict'], param_filter)
        comp.update({
            'epoch': our_dump['epoch'],
            'batch': our_dump['batch'],
            'our_loss': our_dump['loss'],
            'ref_loss': ref_dump['loss'],
            'our_acc': our_dump['acc'],
            'ref_acc': ref_dump['acc'],
            'loss_diff': abs(our_dump['loss'] - ref_dump['loss']),
            'acc_diff': abs(our_dump['acc'] - ref_dump['acc'])
        })
        comparisons.append(comp)

    return comparisons


def plot_model_comparison(comparisons, output_path):
    """Plot comparison between our model and reference model."""
    if not comparisons:
        print("No comparison data found")
        return

    df = pd.DataFrame(comparisons)
    df['step'] = range(len(df))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Weight differences
    axes[0, 0].plot(df['step'], df['avg_l2'], 'b-', alpha=0.7, label='Avg L2')
    axes[0, 0].plot(df['step'], df['max_l2'], 'r--', alpha=0.7, label='Max L2')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('L2 Distance')
    axes[0, 0].set_title('Weight Divergence (L2 Distance)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Cosine similarity
    axes[0, 1].plot(df['step'], df['avg_cos'], 'g-', alpha=0.7, label='Avg Cos')
    axes[0, 1].plot(df['step'], df['min_cos'], 'm--', alpha=0.7, label='Min Cos')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('Weight Similarity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([0, 1.05])

    # Loss comparison
    axes[1, 0].plot(df['step'], df['our_loss'], 'b-', alpha=0.7, label='Our Model')
    axes[1, 0].plot(df['step'], df['ref_loss'], 'r-', alpha=0.7, label='Reference')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Loss difference
    axes[1, 1].plot(df['step'], df['loss_diff'], 'purple', alpha=0.7)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('|Loss Difference|')
    axes[1, 1].set_title('Absolute Loss Difference')
    axes[1, 1].grid(True)

    # Accuracy comparison
    axes[2, 0].plot(df['step'], df['our_acc'], 'b-', alpha=0.7, label='Our Model')
    axes[2, 0].plot(df['step'], df['ref_acc'], 'r-', alpha=0.7, label='Reference')
    axes[2, 0].set_xlabel('Training Step')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].set_title('Training Accuracy Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Accuracy difference
    axes[2, 1].plot(df['step'], df['acc_diff'], 'orange', alpha=0.7)
    axes[2, 1].set_xlabel('Training Step')
    axes[2, 1].set_ylabel('|Accuracy Difference|')
    axes[2, 1].set_title('Absolute Accuracy Difference')
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison.png', dpi=150)
    print(f"Comparison plot saved to {output_path / 'model_comparison.png'}")
    plt.close()

    # Save CSV
    df.to_csv(output_path / 'model_comparison.csv', index=False)
    print(f"Comparison CSV saved to {output_path / 'model_comparison.csv'}")


def print_comparison_statistics(comparisons):
    """Print summary statistics for model comparison."""
    if not comparisons:
        return

    print("\nModel Comparison Statistics:")
    print(f"  Total checkpoints compared: {len(comparisons)}")
    print(f"\nWeight Divergence:")
    print(f"  Avg L2 distance: {np.mean([c['avg_l2'] for c in comparisons]):.6f}")
    print(f"  Max L2 distance: {np.max([c['max_l2'] for c in comparisons]):.6f}")
    print(f"  Avg cosine similarity: {np.mean([c['avg_cos'] for c in comparisons]):.6f}")
    print(f"\nLoss/Accuracy Divergence:")
    print(f"  Avg loss difference: {np.mean([c['loss_diff'] for c in comparisons]):.6f}")
    print(f"  Max loss difference: {np.max([c['loss_diff'] for c in comparisons]):.6f}")
    print(f"  Avg acc difference: {np.mean([c['acc_diff'] for c in comparisons]):.6f}")
    print(f"  Max acc difference: {np.max([c['acc_diff'] for c in comparisons]):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare two models via batch dumps")
    parser.add_argument('our_dir', help='Path to our model dump directory')
    parser.add_argument('ref_dir', help='Path to reference model dump directory')
    parser.add_argument('--output_dir', help='Where to save comparison results (default: our_dir)')
    parser.add_argument('--filter', choices=['all', 'mlp', 'backbone'], default='all',
                       help='Which parameters to compare')
    args = parser.parse_args()

    our_dir = Path(args.our_dir)
    ref_dir = Path(args.ref_dir)
    output_dir = Path(args.output_dir) if args.output_dir else our_dir

    print(f"Loading our model dumps from {our_dir}...")
    our_dumps = load_batch_dumps(our_dir)
    print(f"  Found {len(our_dumps)} dumps")

    print(f"Loading reference model dumps from {ref_dir}...")
    ref_dumps = load_batch_dumps(ref_dir)
    print(f"  Found {len(ref_dumps)} dumps")

    # Set parameter filter
    param_filter = None
    if args.filter == 'mlp':
        param_filter = lambda name: 'mlp' in name or 'fc' in name
    elif args.filter == 'backbone':
        param_filter = lambda name: 'mlp' not in name and 'fc' not in name

    print(f"\nComparing models (filter={args.filter})...")
    comparisons = compare_dump_sequences(our_dumps, ref_dumps, param_filter)
    print(f"  Found {len(comparisons)} matching checkpoints")

    if comparisons:
        print_comparison_statistics(comparisons)
        plot_model_comparison(comparisons, output_dir)

        # Save detailed comparison
        with open(output_dir / 'detailed_comparison.json', 'w') as f:
            json.dump(comparisons, f, indent=2)
        print(f"\nDetailed comparison saved to {output_dir / 'detailed_comparison.json'}")
    else:
        print("No matching checkpoints found!")


if __name__ == '__main__':
    main()
