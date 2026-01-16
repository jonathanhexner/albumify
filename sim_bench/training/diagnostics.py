"""
Diagnostic utilities for debugging overfitting.

Separated from training code to keep concerns clean.
"""
import json
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from sim_bench.quality_assessment.trained_models.phototriage_multifeature import compute_pairwise_accuracy

logger = logging.getLogger(__name__)


def extract_series_id(filename: str) -> str:
    """Extract series_id from filename.

    PhotoTriage format: image1.jpg, image2.jpg, ... (series_id from metadata)
    External format: 000001-01.JPG (series_id = 000001, image_num = 01)

    Args:
        filename: Image filename (can be full path or just basename)

    Returns:
        series_id string, or 'unknown' if can't extract
    """
    from pathlib import Path
    basename = Path(filename).name

    # Try external format: 000001-01.JPG
    if '-' in basename:
        parts = basename.split('-')
        if len(parts) >= 2:
            return parts[0]  # Return series_id portion

    # Otherwise return 'unknown'
    return 'unknown'


def compute_confusion_matrix(y_true, y_pred):
    """Simple 2x2 confusion matrix without sklearn dependency."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = ((y_true == 0) & (y_pred == 0)).sum()
    cm[0, 1] = ((y_true == 0) & (y_pred == 1)).sum()
    cm[1, 0] = ((y_true == 1) & (y_pred == 0)).sum()
    cm[1, 1] = ((y_true == 1) & (y_pred == 1)).sum()

    return cm


def evaluate_simple(model, loader, device, mode='eval'):
    """Quick accuracy calculation with specified model mode."""
    if mode == 'eval':
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in loader:
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            winners = batch['winner'].to(device)

            logits = model(img1, img2)
            loss = F.cross_entropy(logits, winners)

            total_loss += loss.item()
            total_acc += compute_pairwise_accuracy(logits, winners)

    return total_acc / len(loader), total_loss / len(loader)


def compute_mode_gap_diagnostic(model, train_loader, val_loader, device, output_dir, epoch, train_loop_acc):
    """Diagnose BatchNorm issues by comparing train/eval modes."""
    logger.info("  Computing mode gap diagnostic...")

    train_eval_acc, _ = evaluate_simple(model, train_loader, device, mode='eval')
    val_eval_acc, _ = evaluate_simple(model, val_loader, device, mode='eval')
    val_trainmode_acc, _ = evaluate_simple(model, val_loader, device, mode='train')

    mode_gap = {
        'epoch': epoch,
        'train_loop_acc': float(train_loop_acc),
        'train_eval_acc': float(train_eval_acc),
        'val_eval_acc': float(val_eval_acc),
        'val_trainmode_acc': float(val_trainmode_acc),
        'bn_gap': float(val_trainmode_acc - val_eval_acc)
    }

    epoch_metrics_dir = output_dir / f"epoch_{epoch:03d}" / "metrics"
    epoch_metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(epoch_metrics_dir / 'mode_gap.json', 'w') as f:
        json.dump(mode_gap, f, indent=2)

    logger.info(f"  BN gap={mode_gap['bn_gap']:.4f}")
    return mode_gap


def save_epoch_metrics(all_preds, all_winners, all_logprobs, all_image1, all_image2,
                       dataset, avg_loss, metrics_dir):
    """
    Save comprehensive metrics for one epoch.

    Args:
        dataset: Dataset instance (has get_dataframe() method for PhotoTriage, optional for external)
    """
    # Check if dataset has metadata
    has_metadata = hasattr(dataset, 'get_dataframe')
    pairs_df = dataset.get_dataframe() if has_metadata else None

    # Confusion matrix
    cm = compute_confusion_matrix(all_winners, all_preds)
    recall_0 = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() > 0 else 0.0
    recall_1 = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() > 0 else 0.0

    summary = {
        'loss': float(avg_loss),
        'acc': float((all_preds == all_winners).mean()),
        'target_counts': {'0': int((all_winners == 0).sum()), '1': int((all_winners == 1).sum())},
        'pred_counts': {'0': int((all_preds == 0).sum()), '1': int((all_preds == 1).sum())},
        'confusion': cm.tolist(),
        'recall_class0': recall_0,
        'recall_class1': recall_1
    }

    with open(metrics_dir / 'eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save all predictions (no limit)
    n_dump = len(all_preds)
    # Convert logits to probabilities using softmax
    all_probs = np.exp(all_logprobs) / np.exp(all_logprobs).sum(axis=1, keepdims=True)
    dump_data = {
        'image1': all_image1[:n_dump],
        'image2': all_image2[:n_dump],
        'winner': all_winners[:n_dump].tolist(),
        'pred': all_preds[:n_dump].tolist(),
        'logit0': [float(logit[0]) for logit in all_logprobs[:n_dump]],
        'logit1': [float(logit[1]) for logit in all_logprobs[:n_dump]],
        'p0': [float(prob[0]) for prob in all_probs[:n_dump]],
        'p1': [float(prob[1]) for prob in all_probs[:n_dump]]
    }

    # Add metadata - PhotoTriage has agreement/num_reviewers, both have series_id
    if pairs_df is not None:
        # PhotoTriage - get all metadata from pairs_df
        agreements = []
        num_reviewers_list = []
        series_ids = []

        for img1, img2 in zip(all_image1[:n_dump], all_image2[:n_dump]):
            match = pairs_df[(pairs_df['image1'] == img1) & (pairs_df['image2'] == img2)]
            if len(match) > 0:
                row = match.iloc[0]
                agreements.append(float(row['agreement']))
                num_reviewers_list.append(int(row['num_reviewers']))
                series_ids.append(str(row['series_id']))
            else:
                agreements.append(0.0)
                num_reviewers_list.append(0)
                series_ids.append('unknown')

        dump_data['agreement'] = agreements
        dump_data['num_reviewers'] = num_reviewers_list
        dump_data['series_id'] = series_ids
    else:
        # External - extract series_id from filename (000001-01.JPG -> 000001)
        dump_data['series_id'] = [extract_series_id(img1) for img1 in all_image1[:n_dump]]

    pd.DataFrame(dump_data).to_csv(metrics_dir / 'eval_dump.csv', index=False)


def save_per_series_breakdown(all_preds, all_winners, all_image1, all_image2, dataset, metrics_dir):
    """
    Compute and save per-series accuracy breakdown.

    Args:
        dataset: Dataset instance (has get_dataframe() method for PhotoTriage, optional for external)
    """
    # Get series_id - try dataset first, then extract from filenames
    if hasattr(dataset, 'get_dataframe'):
        # PhotoTriage - get from metadata
        pairs_df = dataset.get_dataframe()
        series_map = {(row['image1'], row['image2']): row['series_id']
                     for _, row in pairs_df.iterrows()}
        all_series_ids = [series_map.get((img1, img2), 'unknown')
                         for img1, img2 in zip(all_image1, all_image2)]
    else:
        # External - extract from filename (000001-01.JPG -> 000001)
        all_series_ids = [extract_series_id(img1) for img1 in all_image1]

    # Group by series
    series_data = defaultdict(lambda: {'winners': [], 'preds': []})
    for series_id, winner, pred in zip(all_series_ids, all_winners, all_preds):
        series_data[series_id]['winners'].append(winner)
        series_data[series_id]['preds'].append(pred)

    series_acc_list = []
    for series_id, data in series_data.items():
        winners = np.array(data['winners'])
        preds = np.array(data['preds'])
        acc = (winners == preds).mean()
        series_acc_list.append({
            'series_id': series_id,
            'n_pairs': len(winners),
            'acc': float(acc)
        })

    df_series = pd.DataFrame(series_acc_list).sort_values('acc')
    df_series.to_csv(metrics_dir / 'per_series.csv', index=False)

    # Top/bottom 10
    summary_series = {
        'bottom_10': df_series.head(10).to_dict('records'),
        'top_10': df_series.tail(10).to_dict('records')
    }
    with open(metrics_dir / 'per_series_summary.json', 'w') as f:
        json.dump(summary_series, f, indent=2)


def inspect_series_pairs(model, dataset, device, inspect_dir, series_id, k=6):
    """
    Inspect k pairs from one series with visualization (PhotoTriage only).

    Args:
        dataset: Dataset instance (has get_dataframe(), get_image_dir(), transform)
    """
    # Check dataset has required methods (PhotoTriage only)
    if not (hasattr(dataset, 'get_dataframe') and
            hasattr(dataset, 'get_image_dir') and
            hasattr(dataset, 'transform')):
        logger.info("Skipping series inspection (dataset missing required methods)")
        return

    # Extract metadata from dataset
    pairs_df = dataset.get_dataframe()
    image_dir = dataset.get_image_dir()
    transform = dataset.transform

    df_inspect = pairs_df[pairs_df['series_id'] == series_id].head(k)
    if len(df_inspect) == 0:
        return

    inspect_data = []
    for _, row in df_inspect.iterrows():
        try:
            img1_pil = Image.open(image_dir / row['image1']).convert('RGB')
            img2_pil = Image.open(image_dir / row['image2']).convert('RGB')

            img1_t = transform(img1_pil).unsqueeze(0).to(device)
            img2_t = transform(img2_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, feat1, feat2, diff = model(img1_t, img2_t, return_feats=True)
                logits_swap, _, _, _ = model(img2_t, img1_t, return_feats=True)

            pred = logits.argmax().item()
            pred_swap = logits_swap.argmax().item()
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

            inspect_data.append({
                'image1': row['image1'],
                'image2': row['image2'],
                'winner': int(row['winner']),
                'pred': pred,
                'p0': float(probs[0]),
                'p1': float(probs[1]),
                'correct': pred == row['winner'],
                'feat1_norm': float(torch.norm(feat1).item()),
                'feat2_norm': float(torch.norm(feat2).item()),
                'diff_norm': float(torch.norm(diff).item()),
                'cosine_sim': float(F.cosine_similarity(feat1, feat2).item()),
                'pred_swapped': pred_swap,
                'flips': pred_swap != pred
            })
        except Exception as e:
            logger.warning(f"Failed to inspect pair {row['image1']}, {row['image2']}: {e}")

    if not inspect_data:
        return

    # Save CSV
    pd.DataFrame(inspect_data).to_csv(inspect_dir / 'inspected_pairs.csv', index=False)

    # Create visualization
    try:
        import matplotlib.pyplot as plt
        n_pairs = len(inspect_data)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(8, 3*n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)

        for i, (_, data) in enumerate(zip(df_inspect.iterrows(), inspect_data)):
            img1 = Image.open(image_dir / data['image1'])
            axes[i, 0].imshow(img1)
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f"IMG1 {'✓' if data['winner']==0 else ''}", fontsize=10)

            img2 = Image.open(image_dir / data['image2'])
            axes[i, 1].imshow(img2)
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"IMG2 {'✓' if data['winner']==1 else ''}", fontsize=10)

            caption = (f"GT:{data['winner']} Pred:{data['pred']} "
                       f"P:[{data['p0']:.2f},{data['p1']:.2f}] "
                       f"Cos:{data['cosine_sim']:.3f} "
                       f"Swap:{'YES' if data['flips'] else 'NO'}")
            fig.text(0.5, 1 - (i+0.8)/n_pairs, caption, ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(inspect_dir / 'pairs_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
