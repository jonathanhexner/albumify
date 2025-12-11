"""
PhotoTriage Prediction Visualization Utilities

Visualizes pairwise comparison predictions for the PhotoTriage dataset.
Shows image pairs side-by-side with ground truth and predicted winners.

This module is specifically designed for the PhotoTriage dataset structure:
- Image directory: D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs
- Expected dataframe columns: img1, img2, true_winner, pred_winner
- Optional columns: pred_prob, series_id, majority_label, label_* attributes

Reference:
Chang, H., Yu, F., Wang, J., Ashley, D., & Finkelstein, A. (2016).
Automatic Triage for a Photo Series. ACM Transactions on Graphics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Optional, List
import numpy as np


def visualize_prediction_results(
    df: pd.DataFrame,
    image_dir: str = r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs",
    num_samples: int = 10,
    sample_type: str = "random",
    figsize: tuple = (15, 3),
    show_correct_only: bool = False,
    show_incorrect_only: bool = False,
    seed: Optional[int] = 42
) -> plt.Figure:
    """
    Visualize pairwise comparison predictions.

    Args:
        df: DataFrame with columns ['img1', 'img2', 'true_winner', 'pred_winner']
            - img1, img2: Image filenames
            - true_winner: 0 if img1 is better, 1 if img2 is better
            - pred_winner: 0 if img1 predicted better, 1 if img2 predicted better
        image_dir: Directory containing images
        num_samples: Number of pairs to visualize
        sample_type: 'random', 'first', or 'hardest' (lowest confidence)
        figsize: Figure size (width, height per row)
        show_correct_only: Only show correctly predicted pairs
        show_incorrect_only: Only show incorrectly predicted pairs
        seed: Random seed for reproducibility

    Returns:
        Matplotlib figure object
    """
    image_dir = Path(image_dir)

    # Filter based on correctness
    if show_correct_only:
        df = df[df['true_winner'] == df['pred_winner']].copy()
    elif show_incorrect_only:
        df = df[df['true_winner'] != df['pred_winner']].copy()

    # Sample pairs
    if sample_type == "random":
        if seed is not None:
            df = df.sample(n=min(num_samples, len(df)), random_state=seed)
        else:
            df = df.sample(n=min(num_samples, len(df)))
    elif sample_type == "first":
        df = df.head(num_samples)
    elif sample_type == "hardest":
        # Hardest = closest to 0.5 probability (if available)
        if 'pred_prob' in df.columns:
            df = df.copy()
            df['confidence'] = abs(df['pred_prob'] - 0.5)
            df = df.nsmallest(num_samples, 'confidence')
        else:
            df = df.head(num_samples)

    # Create figure
    n_pairs = len(df)
    fig, axes = plt.subplots(n_pairs, 3, figsize=(figsize[0], figsize[1] * n_pairs))

    # Handle single row case
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, row) in enumerate(df.iterrows()):
        img1_path = image_dir / row['img1']
        img2_path = image_dir / row['img2']

        true_winner = int(row['true_winner'])
        pred_winner = int(row['pred_winner'])
        correct = (true_winner == pred_winner)

        # Load images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            continue

        # Plot Image 1
        axes[idx, 0].imshow(img1)
        axes[idx, 0].axis('off')

        # Build title for img1
        title1 = f"Image 1\n"
        if true_winner == 0:
            title1 += "✓ True Winner"
        if pred_winner == 0:
            title1 += "\n⭐ Predicted Winner"

        # Color based on correctness
        if true_winner == 0:
            color = 'green' if correct else 'red'
            axes[idx, 0].set_title(title1, fontsize=10, fontweight='bold', color=color)
        else:
            axes[idx, 0].set_title(title1, fontsize=10)

        # Add border for winner
        if true_winner == 0:
            for spine in axes[idx, 0].spines.values():
                spine.set_edgecolor('green' if correct else 'red')
                spine.set_linewidth(3)

        # Plot comparison info in middle
        axes[idx, 1].axis('off')

        # Build comparison text
        comparison_text = []
        comparison_text.append("Ground Truth: " + ("Image 1 ✓" if true_winner == 0 else "Image 2 ✓"))
        comparison_text.append("Predicted: " + ("Image 1 ⭐" if pred_winner == 0 else "Image 2 ⭐"))

        if 'pred_prob' in row:
            prob = row['pred_prob']
            comparison_text.append(f"P(img1 wins) = {prob:.3f}")
            comparison_text.append(f"Confidence: {abs(prob - 0.5) * 2:.1%}")

        status = "✅ CORRECT" if correct else "❌ INCORRECT"
        status_color = 'green' if correct else 'red'

        # Display text
        axes[idx, 1].text(0.5, 0.6, "\n".join(comparison_text),
                         ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[idx, 1].text(0.5, 0.2, status,
                         ha='center', va='center', fontsize=12, fontweight='bold',
                         color=status_color)

        # Add metadata if available
        metadata_text = []
        if 'series_id' in row:
            metadata_text.append(f"Series: {row['series_id']}")
        if 'majority_label' in row and pd.notna(row['majority_label']):
            metadata_text.append(f"Reason: {row['majority_label']}")

        if metadata_text:
            axes[idx, 1].text(0.5, 0.05, "\n".join(metadata_text),
                             ha='center', va='center', fontsize=8,
                             style='italic', color='gray')

        # Plot Image 2
        axes[idx, 2].imshow(img2)
        axes[idx, 2].axis('off')

        # Build title for img2
        title2 = f"Image 2\n"
        if true_winner == 1:
            title2 += "✓ True Winner"
        if pred_winner == 1:
            title2 += "\n⭐ Predicted Winner"

        # Color based on correctness
        if true_winner == 1:
            color = 'green' if correct else 'red'
            axes[idx, 2].set_title(title2, fontsize=10, fontweight='bold', color=color)
        else:
            axes[idx, 2].set_title(title2, fontsize=10)

        # Add border for winner
        if true_winner == 1:
            for spine in axes[idx, 2].spines.values():
                spine.set_edgecolor('green' if correct else 'red')
                spine.set_linewidth(3)

    plt.tight_layout()

    # Add overall title
    accuracy = (df['true_winner'] == df['pred_winner']).mean()
    fig.suptitle(f"Pairwise Comparison Results (Accuracy: {accuracy:.1%}, n={len(df)})",
                 fontsize=14, fontweight='bold', y=1.0)
    plt.subplots_adjust(top=0.98)

    return fig


def visualize_by_attribute(
    df: pd.DataFrame,
    attribute: str = 'majority_label',
    image_dir: str = r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs",
    num_samples_per_category: int = 3,
    figsize: tuple = (15, 3)
) -> plt.Figure:
    """
    Visualize predictions grouped by quality attribute.

    Args:
        df: DataFrame with prediction results
        attribute: Column name to group by (e.g., 'majority_label', 'label_sharpness')
        image_dir: Directory containing images
        num_samples_per_category: Number of samples to show per category
        figsize: Figure size per row

    Returns:
        Matplotlib figure object
    """
    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in dataframe")

    # Get top categories
    top_categories = df[attribute].value_counts().head(5).index.tolist()

    all_figs = []

    for category in top_categories:
        if pd.isna(category) or category == '':
            continue

        # Filter to this category
        df_cat = df[df[attribute] == category]

        if len(df_cat) == 0:
            continue

        # Calculate accuracy for this category
        accuracy = (df_cat['true_winner'] == df_cat['pred_winner']).mean()

        print(f"\n{attribute} = '{category}': {len(df_cat)} pairs, {accuracy:.1%} accuracy")

        # Visualize samples
        fig = visualize_prediction_results(
            df_cat,
            image_dir=image_dir,
            num_samples=min(num_samples_per_category, len(df_cat)),
            sample_type='random',
            figsize=figsize
        )

        # Update title
        fig.suptitle(f"{attribute} = '{category}' (Accuracy: {accuracy:.1%}, n={len(df_cat)})",
                    fontsize=14, fontweight='bold', y=1.0)

        all_figs.append(fig)

    return all_figs


def compare_epochs(
    df_epoch1: pd.DataFrame,
    df_epoch2: pd.DataFrame,
    image_dir: str = r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs",
    num_samples: int = 5,
    show_changed_only: bool = True,
    figsize: tuple = (15, 3)
) -> plt.Figure:
    """
    Compare predictions between two epochs.

    Shows pairs where the prediction changed between epochs.

    Args:
        df_epoch1: Predictions from epoch 1
        df_epoch2: Predictions from epoch 2
        image_dir: Directory containing images
        num_samples: Number of pairs to show
        show_changed_only: Only show pairs where prediction changed
        figsize: Figure size per row

    Returns:
        Matplotlib figure object
    """
    # Merge the two dataframes
    df_merged = df_epoch1.merge(
        df_epoch2,
        on=['img1', 'img2', 'true_winner'],
        suffixes=('_ep1', '_ep2')
    )

    if show_changed_only:
        # Only show where prediction changed
        df_merged = df_merged[df_merged['pred_winner_ep1'] != df_merged['pred_winner_ep2']]

    # Sample
    df_merged = df_merged.head(num_samples)

    image_dir = Path(image_dir)
    n_pairs = len(df_merged)

    fig, axes = plt.subplots(n_pairs, 3, figsize=(figsize[0], figsize[1] * n_pairs))

    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, row) in enumerate(df_merged.iterrows()):
        img1_path = image_dir / row['img1']
        img2_path = image_dir / row['img2']

        true_winner = int(row['true_winner'])
        pred_winner_ep1 = int(row['pred_winner_ep1'])
        pred_winner_ep2 = int(row['pred_winner_ep2'])

        # Load images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            continue

        # Plot Image 1
        axes[idx, 0].imshow(img1)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title("Image 1", fontsize=10)

        # Plot comparison
        axes[idx, 1].axis('off')

        comparison_text = []
        comparison_text.append(f"Ground Truth: Image {true_winner + 1}")
        comparison_text.append(f"\nEpoch 1 Pred: Image {pred_winner_ep1 + 1}")
        comparison_text.append(f"Epoch 2 Pred: Image {pred_winner_ep2 + 1}")

        ep1_correct = (true_winner == pred_winner_ep1)
        ep2_correct = (true_winner == pred_winner_ep2)

        if ep1_correct and ep2_correct:
            status = "Stayed Correct ✅→✅"
            color = 'green'
        elif not ep1_correct and not ep2_correct:
            status = "Stayed Wrong ❌→❌"
            color = 'red'
        elif not ep1_correct and ep2_correct:
            status = "Fixed! ❌→✅"
            color = 'blue'
        else:
            status = "Broke ✅→❌"
            color = 'orange'

        axes[idx, 1].text(0.5, 0.5, "\n".join(comparison_text),
                         ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[idx, 1].text(0.5, 0.2, status,
                         ha='center', va='center', fontsize=11, fontweight='bold',
                         color=color)

        # Plot Image 2
        axes[idx, 2].imshow(img2)
        axes[idx, 2].axis('off')
        axes[idx, 2].set_title("Image 2", fontsize=10)

    plt.tight_layout()

    # Calculate statistics
    n_changed = (df_merged['pred_winner_ep1'] != df_merged['pred_winner_ep2']).sum()
    fig.suptitle(f"Epoch 1 → Epoch 2 Comparison ({n_changed}/{len(df_merged)} predictions changed)",
                 fontsize=14, fontweight='bold', y=1.0)

    return fig


# Example usage
if __name__ == "__main__":
    # Example: Load predictions from CSV
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m sim_bench.datasets.visualize_phototriage_predictions <path_to_predictions.csv>")
        print("\nExample:")
        print("  python -m sim_bench.datasets.visualize_phototriage_predictions outputs/train_labels_epoch1.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} predictions from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nOverall accuracy: {(df['true_winner'] == df['pred_winner']).mean():.1%}")

    # Visualize random samples
    fig = visualize_prediction_results(df, num_samples=10)
    plt.show()

    # Visualize incorrect predictions
    print("\n=== Showing Incorrect Predictions ===")
    fig = visualize_prediction_results(df, num_samples=5, show_incorrect_only=True)
    plt.show()
