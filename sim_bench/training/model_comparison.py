"""
Model comparison utility for verifying initialization against reference implementation.

Compares our SiameseCNNRanker with the reference ResNet50 model from
D:\Projects\Series-Photo-Selection\models\ResNet50.py
"""
import logging
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_reference_model(ref_model_path):
    """
    Load reference model from external codebase.

    Args:
        ref_model_path: Path to ResNet50.py file

    Returns:
        Reference model instance
    """
    spec = importlib.util.spec_from_file_location("ref_resnet", ref_model_path)
    ref_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_module)

    ref_model = ref_module.make_network()
    ref_model.eval()
    return ref_model


def compare_model_params(our_model, ref_model):
    """
    Compare parameter counts and shapes between models.

    Returns dict with:
    - total_params: Parameter count comparison
    - backbone_params: Backbone parameter count
    - head_params: Head parameter count
    - shape_mismatches: List of any shape differences
    """
    our_params = {name: p.shape for name, p in our_model.named_parameters()}
    ref_params = {name: p.shape for name, p in ref_model.named_parameters()}

    our_total = sum(p.numel() for p in our_model.parameters())
    ref_total = sum(p.numel() for p in ref_model.parameters())

    our_backbone = sum(p.numel() for p in our_model.backbone.parameters())
    ref_backbone = sum(p.numel() for n, p in ref_model.named_parameters() if 'fc' not in n)

    our_head = sum(p.numel() for p in our_model.mlp.parameters())
    ref_head = sum(p.numel() for n, p in ref_model.named_parameters() if 'fc' in n)

    return {
        'total_params': {'ours': our_total, 'ref': ref_total, 'match': our_total == ref_total},
        'backbone_params': {'ours': our_backbone, 'ref': ref_backbone},
        'head_params': {'ours': our_head, 'ref': ref_head},
        'our_param_count': len(our_params),
        'ref_param_count': len(ref_params)
    }


def compare_backbone_weights(our_model, ref_model, n_layers=5):
    """
    Compare backbone weights layer by layer.

    Samples first n_layers and compares:
    - Weight values (should be identical for ImageNet pretrained)
    - Mean, std, min, max statistics
    - Cosine similarity

    Returns dict with statistics per layer.
    """
    our_backbone_params = list(our_model.backbone.named_parameters())
    ref_backbone_params = [(n, p) for n, p in ref_model.named_parameters() if 'fc' not in n]

    layer_comparisons = []
    for i, ((our_name, our_param), (ref_name, ref_param)) in enumerate(
        zip(our_backbone_params[:n_layers], ref_backbone_params[:n_layers])
    ):
        if our_param.shape != ref_param.shape:
            layer_comparisons.append({
                'layer_idx': i,
                'our_name': our_name,
                'ref_name': ref_name,
                'shape_mismatch': True,
                'our_shape': list(our_param.shape),
                'ref_shape': list(ref_param.shape)
            })
            continue

        our_flat = our_param.detach().cpu().flatten()
        ref_flat = ref_param.detach().cpu().flatten()

        # Cosine similarity
        cos_sim = F.cosine_similarity(our_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()

        # L2 distance
        l2_dist = torch.norm(our_flat - ref_flat).item()

        # Statistics
        layer_comparisons.append({
            'layer_idx': i,
            'our_name': our_name,
            'ref_name': ref_name,
            'shape': list(our_param.shape),
            'cosine_similarity': float(cos_sim),
            'l2_distance': float(l2_dist),
            'our_stats': {
                'mean': float(our_param.mean()),
                'std': float(our_param.std()),
                'min': float(our_param.min()),
                'max': float(our_param.max())
            },
            'ref_stats': {
                'mean': float(ref_param.mean()),
                'std': float(ref_param.std()),
                'min': float(ref_param.min()),
                'max': float(ref_param.max())
            }
        })

    return {
        'n_layers_compared': len(layer_comparisons),
        'layers': layer_comparisons
    }


def compare_forward_pass(our_model, ref_model, device):
    """
    Compare forward pass outputs.

    Args:
        our_model: Our SiameseCNNRanker
        ref_model: Reference ResNet50
        device: Device to use for comparison

    Returns dict with:
    - feature_diff: Difference in extracted features
    - output_diff: Difference in final outputs
    - note: Accounts for our LogSoftmax vs their raw logits
    """
    # Create dummy inputs
    dummy_img1 = torch.randn(2, 3, 224, 224).to(device)
    dummy_img2 = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        # Our model
        our_model.eval()
        our_logprobs, our_feat1, our_feat2, our_diff = our_model(dummy_img1, dummy_img2, return_feats=True)

        # Reference model
        ref_model.eval()
        ref_out = ref_model(dummy_img1, dummy_img2)

    # Compare features (should be identical for same inputs if using same pretrained weights)
    feat1_l2 = torch.norm(our_feat1 - our_feat2).item() if hasattr(ref_model, 'backbone') else -1.0

    # Compare outputs
    # Our model outputs log probabilities, ref outputs raw logits
    our_logits = torch.exp(our_logprobs)  # Convert log probs to probs
    ref_logits = ref_out

    output_l2 = torch.norm(our_logits - ref_logits).item()

    return {
        'input_shape': list(dummy_img1.shape),
        'our_output_shape': list(our_logprobs.shape),
        'ref_output_shape': list(ref_out.shape),
        'feature_diff_norm': float(feat1_l2),
        'output_l2_distance': float(output_l2),
        'our_output_sample': our_logprobs[0].cpu().tolist(),
        'ref_output_sample': ref_logits[0].cpu().tolist(),
        'note': 'Our model outputs LogSoftmax, ref outputs raw logits'
    }


def print_comparison_report(results):
    """Pretty print comparison results."""
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT")
    print("="*80)

    # Parameter comparison
    print("\n[PARAMETER COMPARISON]")
    params = results['param_comparison']
    print(f"  Total params:    Ours={params['total_params']['ours']:,}, Ref={params['total_params']['ref']:,}, Match={params['total_params']['match']}")
    print(f"  Backbone params: Ours={params['backbone_params']['ours']:,}, Ref={params['backbone_params']['ref']:,}")
    print(f"  Head params:     Ours={params['head_params']['ours']:,}, Ref={params['head_params']['ref']:,}")

    # Backbone weights
    print("\n[BACKBONE WEIGHTS COMPARISON]")
    backbone = results['backbone_weights']
    print(f"  Compared {backbone['n_layers_compared']} layers")
    for layer in backbone['layers'][:3]:  # Show first 3
        if layer.get('shape_mismatch'):
            print(f"  Layer {layer['layer_idx']}: SHAPE MISMATCH - ours={layer['our_shape']}, ref={layer['ref_shape']}")
        else:
            print(f"  Layer {layer['layer_idx']}: cos_sim={layer['cosine_similarity']:.6f}, l2_dist={layer['l2_distance']:.6e}")

    # Forward pass
    print("\n[FORWARD PASS COMPARISON]")
    fwd = results['forward_pass']
    print(f"  Input shape: {fwd['input_shape']}")
    print(f"  Output L2 distance: {fwd['output_l2_distance']:.6e}")
    print(f"  Note: {fwd['note']}")

    print("\n" + "="*80 + "\n")


def compare_model_states(model_state1, model_state2, param_names=None):
    """
    Compare two model state dicts.

    Args:
        model_state1: First model state dict
        model_state2: Second model state dict
        param_names: Optional list of parameter names to compare (if None, compare all)

    Returns:
        Dict with per-parameter comparison:
        - l2_distance: L2 norm of difference
        - cosine_similarity: Cosine similarity
        - max_abs_diff: Maximum absolute difference
    """
    if param_names is None:
        param_names = list(model_state1.keys())

    comparisons = {}
    for name in param_names:
        if name not in model_state1 or name not in model_state2:
            continue

        p1 = model_state1[name].detach().cpu().flatten()
        p2 = model_state2[name].detach().cpu().flatten()

        # Skip if shapes don't match
        if p1.shape != p2.shape:
            continue

        l2_dist = torch.norm(p1 - p2).item()
        cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
        max_diff = torch.max(torch.abs(p1 - p2)).item()

        comparisons[name] = {
            'l2_distance': float(l2_dist),
            'cosine_similarity': float(cos_sim),
            'max_abs_diff': float(max_diff)
        }

    return comparisons


def save_batch_comparison(model, ref_state_dict, epoch, batch_idx, output_dir):
    """
    Compare current model with reference state and save results.

    Args:
        model: Current model
        ref_state_dict: Reference model state dict to compare against
        epoch: Current epoch number
        batch_idx: Current batch index
        output_dir: Directory to save comparison results
    """
    batch_dir = Path(output_dir) / f"epoch_{epoch:03d}" / "batch_comparisons"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Compare only MLP head parameters (backbone shouldn't change much)
    mlp_param_names = [name for name in model.state_dict().keys() if 'mlp' in name]

    comparisons = compare_model_states(
        model.state_dict(),
        ref_state_dict,
        param_names=mlp_param_names
    )

    # Compute summary statistics
    all_l2 = [c['l2_distance'] for c in comparisons.values()]
    all_cos = [c['cosine_similarity'] for c in comparisons.values()]

    summary = {
        'epoch': epoch,
        'batch': batch_idx,
        'avg_l2_distance': float(np.mean(all_l2)) if all_l2 else 0.0,
        'max_l2_distance': float(np.max(all_l2)) if all_l2 else 0.0,
        'avg_cosine_sim': float(np.mean(all_cos)) if all_cos else 0.0,
        'min_cosine_sim': float(np.min(all_cos)) if all_cos else 0.0,
        'per_param': comparisons
    }

    with open(batch_dir / f'batch_{batch_idx:04d}.json', 'w') as f:
        json.dump(summary, f, indent=2)


def compare_with_reference_model(our_model, config, output_dir):
    """
    Main entry point: Compare our model with reference model.

    This is the ONLY function called from training script.
    It encapsulates all comparison logic.

    Args:
        our_model: Our SiameseCNNRanker instance
        config: Full config dict (for device, future options)
        output_dir: Where to save comparison report

    Side effects:
        - Prints comparison report to console
        - Saves model_comparison.json to output_dir
    """
    # Hardcoded reference path
    ref_model_path = r'D:\Projects\Series-Photo-Selection\models\ResNet50.py'
    device = config['device']

    logger.info("Comparing model with reference implementation...")

    # Load reference
    ref_model = load_reference_model(ref_model_path)
    ref_model = ref_model.to(device)

    # Run comparisons
    results = {
        'param_comparison': compare_model_params(our_model, ref_model),
        'backbone_weights': compare_backbone_weights(our_model, ref_model, n_layers=5),
        'forward_pass': compare_forward_pass(our_model, ref_model, device)
    }

    # Print report
    print_comparison_report(results)

    # Save to file
    with open(Path(output_dir) / 'model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Comparison report saved to {output_dir}/model_comparison.json")


def dump_model_to_csv(model, output_path):
    """
    Dump model to CSV for easy comparison in Beyond Compare.
    
    Args:
        model: PyTorch model
        output_path: Path to save CSV file
    """
    rows = []
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            data = param.data.cpu()
            rows.append({
                'layer_name': name,
                'shape': str(list(param.shape)),
                'num_params': int(param.numel()),
                'mean': f"{data.mean().item():.10f}",
                'std': f"{data.std().item():.10f}",
                'min': f"{data.min().item():.10f}",
                'max': f"{data.max().item():.10f}",
            })
    
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Model dump saved to: {output_path}")
