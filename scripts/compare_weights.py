"""
Compare weight evolution between two experiments.

Loads weight snapshots from two experiment runs and analyzes:
- Per-layer L2 distance (how much weights differ)
- Per-layer cosine similarity (do they point in same direction?)
- Weight distribution comparisons (histograms)
- Divergence timeline (when do models start differing?)

Usage:
    python scripts/compare_weights.py \\
        --exp1 outputs/siamese_e2e/20260113_073023 \\
        --exp2 outputs/siamese_e2e/20260111_005327 \\
        --output outputs/weight_comparison
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_snapshot(snapshot_path: Path) -> Dict:
    """Load a weight snapshot file.
    
    Args:
        snapshot_path: Path to .pt checkpoint file
    
    Returns:
        Dictionary with model_state_dict, batch_idx, epoch
    """
    return torch.load(snapshot_path, map_location='cpu')


def find_snapshots(exp_dir: Path) -> List[Path]:
    """Find all weight snapshot files in experiment directory.
    
    Args:
        exp_dir: Experiment output directory
    
    Returns:
        Sorted list of snapshot paths (by epoch, then batch)
    """
    snapshots_dir = exp_dir / 'telemetry' / 'weight_snapshots'
    
    if not snapshots_dir.exists():
        logger.warning(f"No snapshots directory found: {snapshots_dir}")
        return []
    
    snapshots = list(snapshots_dir.glob('weights_epoch*.pt'))
    
    # Sort by epoch and batch
    def sort_key(path):
        name = path.stem
        parts = name.replace('weights_epoch', '').split('_batch')
        epoch = int(parts[0])
        batch = int(parts[1]) if len(parts) > 1 else 0
        return (epoch, batch)
    
    snapshots.sort(key=sort_key)
    
    logger.info(f"Found {len(snapshots)} snapshots in {exp_dir.name}")
    return snapshots


def compute_layer_distance(state1: Dict, state2: Dict, layer_name: str) -> float:
    """Compute L2 distance between two layer weights.
    
    Args:
        state1: First model state dict
        state2: Second model state dict
        layer_name: Name of layer to compare
    
    Returns:
        L2 distance (Euclidean norm of difference)
    """
    if layer_name not in state1 or layer_name not in state2:
        return np.nan
    
    w1 = state1[layer_name]
    w2 = state2[layer_name]
    
    diff = w1 - w2
    return float(torch.norm(diff).item())


def compute_layer_cosine_sim(state1: Dict, state2: Dict, layer_name: str) -> float:
    """Compute cosine similarity between two layer weights.
    
    Args:
        state1: First model state dict
        state2: Second model state dict
        layer_name: Name of layer to compare
    
    Returns:
        Cosine similarity (-1 to 1, 1 means same direction)
    """
    if layer_name not in state1 or layer_name not in state2:
        return np.nan
    
    w1 = state1[layer_name].flatten()
    w2 = state2[layer_name].flatten()
    
    cos_sim = torch.nn.functional.cosine_similarity(w1, w2, dim=0)
    return float(cos_sim.item())


def compare_snapshot_pair(
    snap1_path: Path,
    snap2_path: Path
) -> Tuple[Dict, pd.DataFrame]:
    """Compare two snapshots (same checkpoint index from different experiments).
    
    Args:
        snap1_path: Path to first snapshot
        snap2_path: Path to second snapshot
    
    Returns:
        (metadata, per_layer_df)
    """
    snap1 = load_snapshot(snap1_path)
    snap2 = load_snapshot(snap2_path)
    
    state1 = snap1['model_state_dict']
    state2 = snap2['model_state_dict']
    
    # Get all layer names (intersection)
    layers = set(state1.keys()) & set(state2.keys())
    
    # Compute per-layer metrics
    layer_data = []
    for layer_name in sorted(layers):
        l2_dist = compute_layer_distance(state1, state2, layer_name)
        cos_sim = compute_layer_cosine_sim(state1, state2, layer_name)
        
        layer_data.append({
            'layer_name': layer_name,
            'l2_distance': l2_dist,
            'cosine_similarity': cos_sim
        })
    
    per_layer_df = pd.DataFrame(layer_data)
    
    metadata = {
        'epoch': snap1['epoch'],
        'batch_idx': snap1['batch_idx'],
        'checkpoint_name': snap1_path.stem
    }
    
    return metadata, per_layer_df


def compare_experiments(exp1_dir: Path, exp2_dir: Path, output_dir: Path):
    """Compare weight evolution between two experiments.
    
    Args:
        exp1_dir: First experiment directory
        exp2_dir: Second experiment directory
        output_dir: Output directory for comparison results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("WEIGHT COMPARISON ANALYSIS")
    logger.info("="*80)
    logger.info(f"Experiment 1: {exp1_dir}")
    logger.info(f"Experiment 2: {exp2_dir}")
    
    # Find snapshots
    snaps1 = find_snapshots(exp1_dir)
    snaps2 = find_snapshots(exp2_dir)
    
    if not snaps1 or not snaps2:
        logger.error("❌ No snapshots found in one or both experiments")
        logger.error("   Weight snapshots must be enabled in telemetry config:")
        logger.error("   telemetry:")
        logger.error("     track_weight_snapshots: true")
        logger.error("     weight_snapshot_every_n: 500")
        return
    
    # Match snapshots by checkpoint index
    n_checkpoints = min(len(snaps1), len(snaps2))
    logger.info(f"\nComparing {n_checkpoints} checkpoint pairs...")
    
    all_comparisons = []
    
    for i in range(n_checkpoints):
        logger.info(f"  Checkpoint {i+1}/{n_checkpoints}: {snaps1[i].stem}")
        metadata, per_layer_df = compare_snapshot_pair(snaps1[i], snaps2[i])
        
        # Add metadata to per-layer results
        for col, val in metadata.items():
            per_layer_df[col] = val
        
        all_comparisons.append(per_layer_df)
    
    # Combine all checkpoints
    full_df = pd.concat(all_comparisons, ignore_index=True)
    
    # Save detailed CSV
    csv_path = output_dir / 'layer_divergence_detailed.csv'
    full_df.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Detailed results saved: {csv_path}")
    
    # Compute summary statistics
    summary_stats = full_df.groupby('checkpoint_name').agg({
        'l2_distance': ['mean', 'std', 'max'],
        'cosine_similarity': ['mean', 'min']
    }).reset_index()
    
    summary_path = output_dir / 'summary_stats.csv'
    summary_stats.to_csv(summary_path, index=False)
    logger.info(f"✅ Summary stats saved: {summary_path}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_divergence_heatmap(full_df, output_dir)
    plot_divergence_timeline(full_df, output_dir)
    plot_layer_ranking(full_df, output_dir)
    
    # Generate text summary
    generate_text_summary(full_df, output_dir, exp1_dir, exp2_dir)
    
    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


def plot_divergence_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of L2 distance over time."""
    # Pivot to create checkpoint × layer matrix
    pivot = df.pivot(
        index='layer_name',
        columns='checkpoint_name',
        values='l2_distance'
    )
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'L2 Distance'})
    ax.set_title('Per-Layer Weight Divergence Over Training')
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Layer')
    plt.tight_layout()
    
    plot_path = output_dir / 'divergence_heatmap.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"  📊 Heatmap: {plot_path}")
    plt.close()


def plot_divergence_timeline(df: pd.DataFrame, output_dir: Path):
    """Plot average divergence over time."""
    timeline = df.groupby('checkpoint_name')['l2_distance'].mean().reset_index()
    timeline['checkpoint_idx'] = range(len(timeline))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timeline['checkpoint_idx'], timeline['l2_distance'], 'b-o', linewidth=2)
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Average L2 Distance')
    ax.set_title('Model Divergence Timeline (Average Across All Layers)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_dir / 'divergence_timeline.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"  📈 Timeline: {plot_path}")
    plt.close()


def plot_layer_ranking(df: pd.DataFrame, output_dir: Path):
    """Plot layer ranking by final divergence."""
    # Get final checkpoint
    final_checkpoint = df['checkpoint_name'].iloc[-1]
    final_df = df[df['checkpoint_name'] == final_checkpoint].copy()
    final_df = final_df.sort_values('l2_distance', ascending=False)
    
    # Take top 20 most divergent layers
    top_layers = final_df.head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_layers)), top_layers['l2_distance'], color='coral')
    ax.set_yticks(range(len(top_layers)))
    ax.set_yticklabels(top_layers['layer_name'], fontsize=8)
    ax.set_xlabel('L2 Distance')
    ax.set_title('Top 20 Most Divergent Layers (Final Checkpoint)')
    ax.invert_yaxis()
    plt.tight_layout()
    
    plot_path = output_dir / 'layer_ranking.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"  🏆 Layer ranking: {plot_path}")
    plt.close()


def generate_text_summary(df: pd.DataFrame, output_dir: Path, exp1_dir: Path, exp2_dir: Path):
    """Generate human-readable text summary."""
    summary_lines = []
    
    summary_lines.append("="*80)
    summary_lines.append("WEIGHT COMPARISON SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"\nExperiment 1: {exp1_dir.name}")
    summary_lines.append(f"Experiment 2: {exp2_dir.name}")
    
    # Overall statistics
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("OVERALL DIVERGENCE")
    summary_lines.append("="*80)
    
    overall_mean = df['l2_distance'].mean()
    overall_max = df['l2_distance'].max()
    
    summary_lines.append(f"Average L2 distance: {overall_mean:.4f}")
    summary_lines.append(f"Maximum L2 distance: {overall_max:.4f}")
    summary_lines.append(f"Average cosine similarity: {df['cosine_similarity'].mean():.4f}")
    
    # Final checkpoint analysis
    final_checkpoint = df['checkpoint_name'].iloc[-1]
    final_df = df[df['checkpoint_name'] == final_checkpoint]
    
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append(f"FINAL CHECKPOINT: {final_checkpoint}")
    summary_lines.append("="*80)
    summary_lines.append(f"Average L2 distance: {final_df['l2_distance'].mean():.4f}")
    summary_lines.append(f"Average cosine similarity: {final_df['cosine_similarity'].mean():.4f}")
    
    # Most divergent layers
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("TOP 10 MOST DIVERGENT LAYERS (FINAL)")
    summary_lines.append("="*80)
    
    top_10 = final_df.nlargest(10, 'l2_distance')
    for i, row in enumerate(top_10.itertuples(), 1):
        summary_lines.append(f"{i:2d}. {row.layer_name:50s} L2={row.l2_distance:.4f} cos={row.cosine_similarity:.4f}")
    
    # Save summary
    summary_path = output_dir / 'SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"  📄 Summary: {summary_path}")
    
    # Also print to console
    print('\n' + '\n'.join(summary_lines))


def main():
    parser = argparse.ArgumentParser(description='Compare weight evolution between experiments')
    parser.add_argument('--exp1', type=str, required=True, help='First experiment directory')
    parser.add_argument('--exp2', type=str, required=True, help='Second experiment directory')
    parser.add_argument('--output', type=str, default='outputs/weight_comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    exp1_dir = Path(args.exp1)
    exp2_dir = Path(args.exp2)
    output_dir = Path(args.output)
    
    if not exp1_dir.exists():
        logger.error(f"❌ Experiment 1 not found: {exp1_dir}")
        return
    
    if not exp2_dir.exists():
        logger.error(f"❌ Experiment 2 not found: {exp2_dir}")
        return
    
    compare_experiments(exp1_dir, exp2_dir, output_dir)


if __name__ == '__main__':
    main()
