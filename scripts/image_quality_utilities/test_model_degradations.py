"""
Unified benchmark for testing image quality models on synthetic degradations.

Supports multiple model types (Siamese, AVA, IQA) through a unified interface.
Models and test parameters are configured via YAML files.

Usage:
    python scripts/image_quality_utilities/test_model_degradations.py \
        --config configs/image_quality_benchmarks/degradation_test.yaml
"""

import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from sim_bench.image_quality_models.base_model import BaseQualityModel
from sim_bench.image_quality_models.model_factory import create_model
from sim_bench.image_processing.degradation import ImageDegradationProcessor

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_phototriage_images(checkpoint_dir: Path, num_images: int) -> List[str]:
    """
    Get sample images from PhotoTriage validation set.
    
    Args:
        checkpoint_dir: Siamese model output directory
        num_images: Number of images to sample
        
    Returns:
        List of image filenames
    """
    eval_csv = checkpoint_dir / 'epoch_002' / 'val' / 'metrics' / 'eval_dump.csv'
    
    if not eval_csv.exists():
        raise FileNotFoundError(f"PhotoTriage validation metrics not found: {eval_csv}")
    
    df = pd.read_csv(eval_csv)
    all_images = set(df['image1'].tolist() + df['image2'].tolist())
    sampled = list(all_images)[:num_images]
    
    logger.info(f"Sampled {len(sampled)} images from PhotoTriage validation set")
    return sampled


def get_ava_images(predictions_dir: Path, num_images: int) -> List[str]:
    """
    Get sample images from AVA validation set.
    
    Args:
        predictions_dir: Directory containing val_epoch_*.parquet files
        num_images: Number of images to sample
        
    Returns:
        List of image IDs
    """
    # Find latest validation predictions
    pred_files = sorted(predictions_dir.glob('val_epoch_*.parquet'))
    if not pred_files:
        raise FileNotFoundError(f"No validation predictions found in {predictions_dir}")
    
    latest_pred = pred_files[-1]
    df = pd.read_parquet(latest_pred)
    
    # Sample images
    sampled_ids = df['image_id'].head(num_images).tolist()
    sampled = [f"{img_id}.jpg" for img_id in sampled_ids]
    
    logger.info(f"Sampled {len(sampled)} images from AVA validation set")
    return sampled


def get_test_images(images_config: dict) -> tuple:
    """
    Get test images based on configuration.
    
    Returns:
        (image_list, image_dir)
    """
    source = images_config['source']
    num_images = images_config['num_images']
    
    if source == 'phototriage':
        checkpoint_dir = Path(images_config['checkpoint_dir'])
        image_dir = Path(images_config['image_root'])
        images = get_phototriage_images(checkpoint_dir, num_images)
    elif source == 'ava':
        predictions_dir = Path(images_config['predictions_dir'])
        image_dir = Path(images_config['image_dir'])
        images = get_ava_images(predictions_dir, num_images)
    else:
        raise ValueError(f"Unknown image source: {source}")
    
    return images, image_dir


def generate_degraded_variants(
    image_paths: List[str],
    image_dir: Path,
    output_dir: Path,
    degradation_config: dict
) -> pd.DataFrame:
    """
    Generate degraded variants of images.
    
    Args:
        image_paths: List of image filenames
        image_dir: Directory containing original images
        output_dir: Output directory for degraded images
        degradation_config: Dict with degradation parameters
        
    Returns:
        DataFrame with degradation metadata
    """
    processor = ImageDegradationProcessor(output_dir=output_dir)
    
    records = []
    
    for img_name in tqdm(image_paths, desc="Generating degradations"):
        img_path = image_dir / img_name
        
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        
        # Blur degradations
        for sigma in degradation_config.get('blur_sigmas', []):
            degraded_path = processor.apply_gaussian_blur(img_path, sigma=sigma)
            records.append({
                'original': img_name,
                'degraded': degraded_path.name,
                'degraded_path': str(degraded_path),
                'type': 'blur',
                'level': sigma
            })
        
        # JPEG compression
        for quality in degradation_config.get('jpeg_qualities', []):
            degraded_path = processor.apply_jpeg_compression(img_path, quality=quality)
            records.append({
                'original': img_name,
                'degraded': degraded_path.name,
                'degraded_path': str(degraded_path),
                'type': 'jpeg',
                'level': quality
            })
        
        # Exposure adjustment
        for stops in degradation_config.get('exposure_stops', []):
            degraded_path = processor.apply_exposure_adjustment(img_path, stops=stops)
            records.append({
                'original': img_name,
                'degraded': degraded_path.name,
                'degraded_path': str(degraded_path),
                'type': 'exposure',
                'level': stops
            })
        
        # Edge crops
        for pct in degradation_config.get('crop_edge_pcts', []):
            for side in ['right', 'bottom']:
                degraded_path = processor.apply_edge_crop(img_path, crop_percentage=pct, side=side)
                records.append({
                    'original': img_name,
                    'degraded': degraded_path.name,
                    'degraded_path': str(degraded_path),
                    'type': 'crop_edge',
                    'level': f'{side}_{int(pct*100)}pct'
                })
        
        # Corner crops
        for frac in degradation_config.get('crop_corner_fracs', []):
            for corner in ['top_left', 'bottom_right']:
                degraded_path = processor.apply_corner_crop(img_path, crop_fraction=frac, corner=corner)
                records.append({
                    'original': img_name,
                    'degraded': degraded_path.name,
                    'degraded_path': str(degraded_path),
                    'type': 'crop_corner',
                    'level': f'{corner}_{int(frac*100)}pct'
                })
        
        # Aspect distortion crops
        for frac in degradation_config.get('crop_aspect_fracs', []):
            for mode in ['tall', 'wide']:
                degraded_path = processor.apply_aspect_distortion_crop(img_path, aspect_mode=mode, keep_fraction=frac)
                records.append({
                    'original': img_name,
                    'degraded': degraded_path.name,
                    'degraded_path': str(degraded_path),
                    'type': 'crop_aspect',
                    'level': f'{mode}_{int(frac*100)}pct'
                })
        
        # Center content crops
        for frac in degradation_config.get('crop_center_fracs', []):
            degraded_path = processor.apply_center_content_crop(img_path, crop_fraction=frac)
            records.append({
                'original': img_name,
                'degraded': degraded_path.name,
                'degraded_path': str(degraded_path),
                'type': 'crop_center',
                'level': f'{int(frac*100)}pct'
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} degraded variants")
    
    return df


def evaluate_all_models(
    models: List[BaseQualityModel],
    degradations_df: pd.DataFrame,
    image_dir: Path
) -> pd.DataFrame:
    """
    Evaluate all models on degradations.
    
    Args:
        models: List of model instances
        degradations_df: DataFrame with degradation metadata
        image_dir: Directory containing original images
        
    Returns:
        DataFrame with unified results from all models
    """
    all_results = []
    
    for model in models:
        logger.info(f"Evaluating {model.name}...")
        
        for _, row in tqdm(degradations_df.iterrows(), total=len(degradations_df), desc=model.name):
            original_path = image_dir / row['original']
            degraded_path = Path(row['degraded_path'])
            
            # Use unified interface
            result = model.compare_images(original_path, degraded_path)
            
            all_results.append({
                'model_name': model.name,
                'model_type': model.model_type,
                'original': row['original'],
                'degraded': row['degraded'],
                'degradation_type': row['type'],
                'degradation_level': row['level'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'score_original': result.get('score_img1'),
                'score_degraded': result.get('score_img2'),
                'correct': result['prediction'] == 1,  # Original should be better
            })
    
    return pd.DataFrame(all_results)


def compute_summary_statistics(results_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics across all models and degradation types.
    
    Args:
        results_df: Unified results DataFrame
        
    Returns:
        Summary statistics dictionary
    """
    summary = {
        'overall': {},
        'by_degradation': {}
    }
    
    # Overall stats per model
    for model_name in results_df['model_name'].unique():
        model_subset = results_df[results_df['model_name'] == model_name]
        
        summary['overall'][model_name] = {
            'accuracy': float(model_subset['correct'].mean()),
            'avg_confidence': float(model_subset['confidence'].mean()),
            'total_pairs': len(model_subset)
        }
        
        # Add score drop for models that provide scores
        if model_subset['score_original'].notna().any():
            score_drops = model_subset['score_original'] - model_subset['score_degraded']
            summary['overall'][model_name]['avg_score_drop'] = float(score_drops.mean())
    
    # Stats by degradation type
    for deg_type in results_df['degradation_type'].unique():
        deg_subset = results_df[results_df['degradation_type'] == deg_type]
        summary['by_degradation'][deg_type] = {}
        
        for model_name in deg_subset['model_name'].unique():
            model_deg_subset = deg_subset[deg_subset['model_name'] == model_name]
            summary['by_degradation'][deg_type][model_name] = float(model_deg_subset['correct'].mean())
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Unified model degradation benchmark')
    parser.add_argument('--config', required=True, help='Path to benchmark config YAML')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    logger.info(f"Loading config: {args.config}")
    config = load_config(Path(args.config))
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all models from config
    logger.info("Creating models...")
    models = [create_model(m) for m in config['models']]
    logger.info(f"Loaded {len(models)} models")
    
    # Get test images
    logger.info("Loading test images...")
    test_images, image_dir = get_test_images(config['images'])
    logger.info(f"Using {len(test_images)} images from {image_dir}")
    
    # Generate degraded variants
    logger.info("Generating degraded variants...")
    degradations_df = generate_degraded_variants(
        test_images,
        image_dir,
        output_dir / 'degraded_images',
        config['degradations']
    )
    degradations_df.to_csv(output_dir / 'degradations_metadata.csv', index=False)
    
    # Evaluate all models
    logger.info("Evaluating models on degradations...")
    results = evaluate_all_models(models, degradations_df, image_dir)
    results.to_csv(output_dir / 'unified_results.csv', index=False)
    
    # Compute summary
    logger.info("Computing summary statistics...")
    summary = compute_summary_statistics(results)
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*60)
    logger.info(f"\nModels tested: {', '.join([m['name'] for m in config['models']])}")
    logger.info(f"Total test pairs: {len(results) // len(models)}")
    logger.info("\nOverall accuracy by model:")
    for model_name, stats in summary['overall'].items():
        logger.info(f"  {model_name}: {stats['accuracy']:.1%} "
                   f"(confidence: {stats['avg_confidence']:.2f})")
    
    logger.info("\nAccuracy by degradation type:")
    for deg_type, model_stats in summary['by_degradation'].items():
        logger.info(f"  {deg_type}:")
        for model_name, acc in model_stats.items():
            logger.info(f"    {model_name}: {acc:.1%}")
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
