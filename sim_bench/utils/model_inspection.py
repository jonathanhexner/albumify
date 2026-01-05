"""
Utility functions for inspecting model outputs.

Generic inspection tools that work with any model and dataloader.
Useful for debugging, verifying model behavior, and checking predictions.
"""
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
import torch
logger = logging.getLogger(__name__)


def inspect_model_output(model, loader, device, save_path=None):
    """
    Pass first batch through model and return results as a DataFrame.

    Args:
        model: The model to inspect
        loader: DataLoader to get sample batch from
        device: Device to run inference on
        save_path: Optional path to save CSV results (default: None)

    Returns:
        pd.DataFrame: Results with columns:
            - image1, image2: Image filenames (if available)
            - winner: Ground truth (0 or 1)
            - pred: Prediction (0 or 1)
            - correct: Whether prediction matches ground truth
            - logit0, logit1: Raw logits
            - prob0, prob1: Probabilities after softmax
 
    Example:
        >>> from sim_bench.utils.model_inspection import inspect_model_output
        >>> df = inspect_model_output(model, train_loader, 'cuda')
        >>> df = inspect_model_output(model, val_loader, 'cuda', save_path='results.csv')
    """

    batch = next(iter(loader))

    img1 = batch['img1'].to(device)
    img2 = batch['img2'].to(device)
    winners = batch['winner'].to(device)

    with torch.no_grad():
        logits = model(img1, img2)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    # Create DataFrame
    data = {
        'winner': winners.cpu().numpy(),
        'pred': preds.cpu().numpy(),
        'correct': (preds == winners).cpu().numpy(),
        'logit0': logits[:, 0].cpu().numpy(),
        'logit1': logits[:, 1].cpu().numpy(),
        'prob0': probs[:, 0].cpu().numpy(),
        'prob1': probs[:, 1].cpu().numpy(),
    }

    # Add image filenames if available
    if 'image1' in batch:
        data['image1'] = batch['image1']
        data['image2'] = batch['image2']

    df = pd.DataFrame(data)

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Results saved to: {save_path}")

    return df


def inspect_model_weights(model, layer_names=None, output_file=None):
    """
    Inspect model weights and statistics.

    Args:
        model: The model to inspect
        layer_names: List of layer names to inspect (default: None, inspects all)
        output_file: Optional path to save inspection results (default: None, prints only)

    Returns:
        dict: Weight statistics for each layer

    Example:
        >>> from sim_bench.utils.model_inspection import inspect_model_weights
        >>> inspect_model_weights(model)
        >>> inspect_model_weights(model, layer_names=['mlp.0', 'mlp.2'])
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL WEIGHT INSPECTION")
    lines.append("=" * 70)

    stats = {}
    for name, param in model.named_parameters():
        if layer_names is None or any(ln in name for ln in layer_names):
            if param.requires_grad:
                weight_stats = {
                    'shape': list(param.shape),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'num_params': param.numel()
                }
                stats[name] = weight_stats

                lines.append(f"\n{name}:")
                lines.append(f"  Shape: {weight_stats['shape']}")
                lines.append(f"  Params: {weight_stats['num_params']:,}")
                lines.append(f"  Mean: {weight_stats['mean']:.6f}")
                lines.append(f"  Std: {weight_stats['std']:.6f}")
                lines.append(f"  Range: [{weight_stats['min']:.6f}, {weight_stats['max']:.6f}]")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Total trainable parameters: {total_params:,}")
    lines.append("=" * 70)

    output_text = "\n".join(lines)

    # Print to console
    print(output_text)
    if logger.hasHandlers():
        logger.info("\n" + output_text)

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output_text)
        msg = f"Weight inspection results saved to: {output_file}"
        print(msg)
        if logger.hasHandlers():
            logger.info(msg)

    return stats
