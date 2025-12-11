"""
Analyze pairwise quality assessment results by user-labeled attributes.

This script combines:
1. Method predictions (from merged_pairwise_results.csv)
2. User labels/attributes (from PhotoTriage dataset)
3. Computes per-attribute accuracy for each method
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sim_bench.analysis.utils import get_project_root


def create_pair_id(series_id: int, compare_id1: int, compare_id2: int) -> str:
    """
    Create pair_id matching the format used in pairwise benchmark.

    Format: {series_id}_{compare_id1}_{compare_id2}
    """
    return f"{series_id}_{compare_id1}_{compare_id2}"


def load_user_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load user labels for image pairs.

    Args:
        labels_path: Path to photo_triage_pairs_keyword_labels.csv

    Returns:
        DataFrame with columns:
        - pair_id: constructed from series_id, compareID1, compareID2
        - series_id, compareID1, compareID2
        - majority_label: main reason for preference
        - label_*: count for each attribute category
        - Agreement: user agreement level (0-1)
    """
    df = pd.read_csv(labels_path)

    # Create pair_id to match pairwise results
    df['pair_id'] = df.apply(
        lambda row: create_pair_id(row['series_id'], row['compareID1'], row['compareID2']),
        axis=1
    )

    return df


def merge_results_with_labels(
    merged_results_path: Path,
    labels_path: Path
) -> pd.DataFrame:
    """
    Merge pairwise results with user labels.

    Args:
        merged_results_path: Path to merged_pairwise_results.csv
        labels_path: Path to photo_triage_pairs_keyword_labels.csv

    Returns:
        Combined DataFrame with method results and user labels
    """
    # Load data
    results_df = pd.read_csv(merged_results_path)
    labels_df = load_user_labels(labels_path)

    print(f"Loaded {len(results_df)} pairwise results")
    print(f"Loaded {len(labels_df)} labeled pairs")

    # Merge on pair_id
    merged = results_df.merge(labels_df, on='pair_id', how='inner', suffixes=('', '_label'))

    print(f"Merged: {len(merged)} pairs with both results and labels")
    print(f"Coverage: {len(merged)/len(results_df)*100:.1f}% of results have labels")

    return merged


def compute_accuracy_by_attribute(
    merged_df: pd.DataFrame,
    methods: list[str],
    attribute_col: str
) -> pd.DataFrame:
    """
    Compute accuracy for each method broken down by attribute value.

    Args:
        merged_df: Combined results + labels DataFrame
        methods: List of method names
        attribute_col: Column name for attribute (e.g., 'majority_label', 'label_sharpness')

    Returns:
        DataFrame with columns: attribute_value, count, {method}_accuracy...
    """
    results = []

    for attribute_value, group in merged_df.groupby(attribute_col):
        row = {
            'attribute': attribute_col,
            'value': attribute_value,
            'count': len(group)
        }

        for method in methods:
            correct_col = f'{method}_correct'
            if correct_col in group.columns:
                row[f'{method}_accuracy'] = group[correct_col].mean()

        results.append(row)

    return pd.DataFrame(results).sort_values('count', ascending=False)


def compute_accuracy_by_label_presence(
    merged_df: pd.DataFrame,
    methods: list[str],
    label_cols: list[str]
) -> pd.DataFrame:
    """
    Compute accuracy when a specific label is present (count > 0).

    Args:
        merged_df: Combined results + labels DataFrame
        methods: List of method names
        label_cols: List of label columns (e.g., ['label_sharpness', 'label_composition'])

    Returns:
        DataFrame showing accuracy when each label is present vs absent
    """
    results = []

    for label_col in label_cols:
        if label_col not in merged_df.columns:
            continue

        # Pairs where this label was mentioned (count > 0)
        label_present = merged_df[merged_df[label_col] > 0]
        label_absent = merged_df[merged_df[label_col] == 0]

        row = {
            'label': label_col.replace('label_', ''),
            'present_count': len(label_present),
            'absent_count': len(label_absent)
        }

        for method in methods:
            correct_col = f'{method}_correct'
            if correct_col in merged_df.columns:
                row[f'{method}_present'] = label_present[correct_col].mean()
                row[f'{method}_absent'] = label_absent[correct_col].mean()
                row[f'{method}_diff'] = row[f'{method}_present'] - row[f'{method}_absent']

        results.append(row)

    return pd.DataFrame(results).sort_values('present_count', ascending=False)


def plot_accuracy_by_majority_label(
    accuracy_df: pd.DataFrame,
    methods: list[str],
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Plot heatmap of accuracy by majority label for each method.

    Args:
        accuracy_df: DataFrame from compute_accuracy_by_attribute()
        methods: List of method names
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Prepare data for heatmap
    accuracy_cols = [f'{m}_accuracy' for m in methods if f'{m}_accuracy' in accuracy_df.columns]

    if not accuracy_cols:
        print("No accuracy columns found")
        return None

    # Create matrix: rows = labels, columns = methods
    heatmap_data = accuracy_df[['value'] + accuracy_cols].set_index('value')
    heatmap_data.columns = [col.replace('_accuracy', '') for col in heatmap_data.columns]

    # Sort by overall difficulty (average accuracy across methods)
    heatmap_data['avg'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('avg', ascending=True)
    heatmap_data = heatmap_data.drop('avg', axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Accuracy'},
        ax=ax
    )

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Majority Label', fontsize=12)
    ax.set_title('Method Accuracy by User-Labeled Attribute', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_method_performance_by_label(
    label_presence_df: pd.DataFrame,
    method: str,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot how a specific method performs when different labels are present/absent.

    Args:
        label_presence_df: DataFrame from compute_accuracy_by_label_presence()
        method: Method name to analyze
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    present_col = f'{method}_present'
    absent_col = f'{method}_absent'
    diff_col = f'{method}_diff'

    if present_col not in label_presence_df.columns:
        print(f"Method {method} not found")
        return None

    # Filter to labels with sufficient samples
    df = label_presence_df[label_presence_df['present_count'] >= 10].copy()
    df = df.sort_values(diff_col, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Accuracy when present vs absent
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax1.bar(x - width/2, df[present_col], width, label='Label Present', color='steelblue')
    bars2 = ax1.bar(x + width/2, df[absent_col], width, label='Label Absent', color='coral')

    ax1.set_xlabel('Attribute', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title(f'{method}: Accuracy by Attribute Presence', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['label'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 2: Difference (performance boost when label present)
    ax2 = axes[1]
    colors = ['green' if d > 0 else 'red' for d in df[diff_col]]
    bars = ax2.barh(x, df[diff_col], color=colors, alpha=0.7)

    ax2.set_yticks(x)
    ax2.set_yticklabels(df['label'])
    ax2.set_xlabel('Accuracy Difference (Present - Absent)', fontsize=11)
    ax2.set_title(f'{method}: Performance Boost by Attribute', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


__all__ = [
    'create_pair_id',
    'load_user_labels',
    'merge_results_with_labels',
    'compute_accuracy_by_attribute',
    'compute_accuracy_by_label_presence',
    'plot_accuracy_by_majority_label',
    'plot_method_performance_by_label',
]


if __name__ == '__main__':
    PROJECT_ROOT = get_project_root()

    # Paths
    BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "pairwise_benchmark_3hour" / "pairwise_20251120_100520"
    MERGED_RESULTS = BENCHMARK_DIR / "merged_pairwise_results.csv"
    LABELS_PATH = Path(r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_keyword_labels.csv")

    print(f"{'='*80}")
    print("PAIRWISE ANALYSIS BY USER ATTRIBUTES")
    print(f"{'='*80}\n")

    # Merge results with labels
    print("Step 1: Merging results with labels...")
    merged_df = merge_results_with_labels(MERGED_RESULTS, LABELS_PATH)

    # Get method names
    methods = [col.replace('_correct', '') for col in merged_df.columns if col.endswith('_correct')]
    print(f"\nMethods: {', '.join(methods)}\n")

    # Compute accuracy by majority label
    print("Step 2: Computing accuracy by majority label...")
    majority_label_accuracy = compute_accuracy_by_attribute(merged_df, methods, 'majority_label')
    print("\nAccuracy by Majority Label:")
    print(majority_label_accuracy.to_string(index=False))

    # Save results
    output_path = BENCHMARK_DIR / "accuracy_by_majority_label.csv"
    majority_label_accuracy.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path.name}")

    # Compute accuracy by label presence
    print("\n" + "="*80)
    print("Step 3: Computing accuracy by label presence...")
    label_cols = [col for col in merged_df.columns if col.startswith('label_') and col != 'label_no_reason_given']
    label_presence_accuracy = compute_accuracy_by_label_presence(merged_df, methods, label_cols)
    print("\nTop attributes by frequency:")
    print(label_presence_accuracy[['label', 'present_count']].head(10).to_string(index=False))

    # Save results
    output_path = BENCHMARK_DIR / "accuracy_by_label_presence.csv"
    label_presence_accuracy.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path.name}")

    # Create visualizations
    print("\n" + "="*80)
    print("Step 4: Creating visualizations...")

    # Plot 1: Heatmap of accuracy by majority label
    fig1 = plot_accuracy_by_majority_label(majority_label_accuracy, methods, figsize=(14, 8))
    if fig1:
        fig1_path = BENCHMARK_DIR / "accuracy_by_majority_label_heatmap.png"
        fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {fig1_path.name}")
        plt.close(fig1)

    # Plot 2: Per-method analysis
    for method in ['Sharpness', 'Combined-RuleBased', 'CLIP-Aesthetic-LAION']:
        fig = plot_method_performance_by_label(label_presence_accuracy, method, figsize=(14, 6))
        if fig:
            fig_path = BENCHMARK_DIR / f"{method}_performance_by_attribute.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"[SAVED] {fig_path.name}")
            plt.close(fig)

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"{'='*80}")
