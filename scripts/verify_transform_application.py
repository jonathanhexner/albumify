"""
Verify that transforms produce similar outputs.

Loads sample images and applies both internal and external transforms,
then compares the resulting tensors statistically and visually.
"""
import sys
from pathlib import Path
import yaml
import logging
import torch
from PIL import Image
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.datasets.transform_factory import create_transform


def get_external_transform():
    """Import and return the external transform."""
    external_path = r'D:\Projects\Series-Photo-Selection'
    if external_path not in sys.path:
        sys.path.insert(0, external_path)
    
    from data.dataloader import transform as external_transform
    return external_transform


def compute_tensor_stats(tensor: torch.Tensor, name: str):
    """Compute statistics for a tensor."""
    stats = {
        'name': name,
        'shape': str(tuple(tensor.shape)),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
    }
    
    # Per-channel statistics
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        for c, channel_name in enumerate(['R', 'G', 'B']):
            stats[f'{channel_name}_mean'] = tensor[c].mean().item()
            stats[f'{channel_name}_std'] = tensor[c].std().item()
            stats[f'{channel_name}_min'] = tensor[c].min().item()
            stats[f'{channel_name}_max'] = tensor[c].max().item()
    
    return stats


def compare_transforms(image_paths: list, internal_transform, external_transform):
    """Compare transform outputs on sample images."""
    logger.info("\n" + "="*80)
    logger.info("TRANSFORM VERIFICATION")
    logger.info("="*80)
    
    results = []
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        logger.info(f"\nProcessing: {img_path.name}")
        logger.info(f"  Original size: {img.size}")
        
        # Apply internal transform
        internal_tensor = internal_transform(img)
        internal_stats = compute_tensor_stats(internal_tensor, f'internal_{img_path.name}')
        
        # Apply external transform
        external_tensor = external_transform(img)
        external_stats = compute_tensor_stats(external_tensor, f'external_{img_path.name}')
        
        # Compare
        diff = (internal_tensor - external_tensor).abs()
        
        comparison = {
            'image': img_path.name,
            'original_width': img.size[0],
            'original_height': img.size[1],
            'internal_shape': internal_stats['shape'],
            'external_shape': external_stats['shape'],
            'internal_mean': internal_stats['mean'],
            'external_mean': external_stats['mean'],
            'internal_std': internal_stats['std'],
            'external_std': external_stats['std'],
            'mean_abs_diff': diff.mean().item(),
            'max_abs_diff': diff.max().item(),
            'shapes_match': internal_stats['shape'] == external_stats['shape']
        }
        
        # Add per-channel comparison
        if 'R_mean' in internal_stats and 'R_mean' in external_stats:
            for channel in ['R', 'G', 'B']:
                comparison[f'{channel}_mean_internal'] = internal_stats[f'{channel}_mean']
                comparison[f'{channel}_mean_external'] = external_stats[f'{channel}_mean']
                comparison[f'{channel}_diff'] = abs(internal_stats[f'{channel}_mean'] - external_stats[f'{channel}_mean'])
        
        results.append(comparison)
        
        # Log summary
        if comparison['shapes_match']:
            logger.info(f"  ✅ Shapes match: {internal_stats['shape']}")
        else:
            logger.error(f"  ❌ Shape mismatch! Internal: {internal_stats['shape']}, External: {external_stats['shape']}")
        
        logger.info(f"  Mean absolute difference: {comparison['mean_abs_diff']:.6f}")
        logger.info(f"  Max absolute difference: {comparison['max_abs_diff']:.6f}")
        
        if comparison['mean_abs_diff'] < 0.01:
            logger.info(f"  ✅ Transforms produce nearly identical outputs")
        elif comparison['mean_abs_diff'] < 0.1:
            logger.warning(f"  ⚠️  Transforms produce similar but not identical outputs")
        else:
            logger.error(f"  ❌ Transforms produce significantly different outputs!")
    
    return pd.DataFrame(results)


def main():
    # Load configs to get transform settings
    internal_config = yaml.safe_load(open('outputs/siamese_e2e/20260113_073023/config.yaml'))
    
    logger.info("Setting up transforms...")
    logger.info("  Internal: Using config-based transform")
    internal_transform = create_transform(internal_config)
    
    logger.info("  External: Importing from Series-Photo-Selection")
    external_transform = get_external_transform()
    
    # Find sample images
    image_root = Path(internal_config['data']['image_root'])
    
    # Get 20 sample images
    image_files = list(image_root.glob('*.JPG'))[:20]
    
    if len(image_files) == 0:
        logger.error(f"No images found in {image_root}")
        return
    
    logger.info(f"\nFound {len(image_files)} sample images")
    
    # Compare transforms
    results = compare_transforms(image_files, internal_transform, external_transform)
    
    # Save results
    output_dir = Path('outputs/transform_verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'stats.csv'
    results.to_csv(output_path, index=False)
    logger.info(f"\n✅ Statistics saved to: {output_path}")
    
    # Overall summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    all_shapes_match = results['shapes_match'].all()
    avg_mean_diff = results['mean_abs_diff'].mean()
    max_mean_diff = results['mean_abs_diff'].max()
    
    logger.info(f"\nShape consistency: {'✅ ALL MATCH' if all_shapes_match else '❌ MISMATCH DETECTED'}")
    logger.info(f"Average mean abs difference: {avg_mean_diff:.6f}")
    logger.info(f"Maximum mean abs difference: {max_mean_diff:.6f}")
    
    if avg_mean_diff < 0.01:
        logger.info("\n✅ TRANSFORMS ARE EQUIVALENT")
        logger.info("The internal and external transforms produce nearly identical outputs.")
    elif avg_mean_diff < 0.1:
        logger.warning("\n⚠️  TRANSFORMS ARE SIMILAR BUT NOT IDENTICAL")
        logger.warning("Small differences detected. Review the statistics for details.")
    else:
        logger.error("\n❌ TRANSFORMS ARE SIGNIFICANTLY DIFFERENT!")
        logger.error("This is likely the source of the performance difference.")
    
    logger.info("\n" + "="*80)
    logger.info("Transform verification complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
