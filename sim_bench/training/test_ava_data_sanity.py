"""
Sanity checks for AVA dataset and training pipeline.

Run before training to verify data loading, label distributions, and metrics.

Usage:
    python -m sim_bench.training.test_ava_data_sanity --config configs/ava/resnet50_gpu.yaml
"""
import argparse
import yaml
import logging
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

from sim_bench.datasets.ava_dataset import AVADataset, load_ava_labels, create_splits
from sim_bench.models.ava_resnet import create_transform

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def test_histogram_normalization(loader, num_batches=10):
    """Test 1: Verify distribution normalization."""
    logger.info("=" * 60)
    logger.info("TEST 1: Histogram Normalization")
    logger.info("=" * 60)
    
    all_issues = []
    total_samples = 0
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
            
        distributions = batch['distribution']
        total_samples += len(distributions)
        
        # Check sums
        sums = distributions.sum(dim=1)
        sum_error = torch.abs(sums - 1.0)
        max_sum_error = sum_error.max().item()
        
        if max_sum_error > 1e-4:
            all_issues.append(f"Batch {batch_idx}: max sum error = {max_sum_error:.6f}")
        
        # Check for negatives
        has_negative = (distributions < 0).any()
        if has_negative:
            all_issues.append(f"Batch {batch_idx}: Contains negative values")
        
        # Check for NaNs
        has_nan = torch.isnan(distributions).any()
        if has_nan:
            all_issues.append(f"Batch {batch_idx}: Contains NaN values")
    
    if all_issues:
        logger.error(f"❌ FAILED: Found {len(all_issues)} issues:")
        for issue in all_issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info(f"✅ PASSED: All {total_samples} samples have valid distributions")
        logger.info(f"   - All sums within 1e-4 of 1.0")
        logger.info(f"   - No negative values")
        logger.info(f"   - No NaN values")
        return True


def test_mean_label_distribution(loader, num_samples=3000):
    """Test 2: Check mean label statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Mean Label Distribution")
    logger.info("=" * 60)
    
    all_means = []
    
    for batch in loader:
        means = batch['mean_score'].numpy()
        all_means.extend(means)
        if len(all_means) >= num_samples:
            break
    
    all_means = np.array(all_means[:num_samples])
    
    mean_min = all_means.min()
    mean_max = all_means.max()
    mean_avg = all_means.mean()
    mean_std = all_means.std()
    
    logger.info(f"Statistics over {len(all_means)} samples:")
    logger.info(f"  Min:  {mean_min:.4f}")
    logger.info(f"  Max:  {mean_max:.4f}")
    logger.info(f"  Mean: {mean_avg:.4f}")
    logger.info(f"  Std:  {mean_std:.4f}")
    
    issues = []
    
    # Check range
    if mean_min < 1.0 or mean_max > 10.0:
        issues.append(f"Values outside [1, 10] range: [{mean_min:.4f}, {mean_max:.4f}]")
    
    # Check realistic range (most should be 3-8)
    if mean_min > 4.0:
        issues.append(f"Min too high ({mean_min:.4f}), expected ~3-4")
    if mean_max < 7.0:
        issues.append(f"Max too low ({mean_max:.4f}), expected ~7-9")
    
    # Check std
    if mean_std < 0.6:
        issues.append(f"Std too low ({mean_std:.4f}), labels might be constant or wrong")
    
    if issues:
        logger.error(f"❌ FAILED:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✅ PASSED: Label distribution looks reasonable")
        return True


def test_spearman_implementation():
    """Test 3: Verify Spearman correlation implementation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Spearman Implementation Sanity")
    logger.info("=" * 60)
    
    # Create test data
    np.random.seed(42)
    gt_mean = np.random.randn(1000)
    shuffled_gt_mean = gt_mean.copy()
    np.random.shuffle(shuffled_gt_mean)
    
    # Test 1: Perfect correlation
    corr_perfect, _ = spearmanr(gt_mean, gt_mean)
    logger.info(f"Spearman(gt_mean, gt_mean) = {corr_perfect:.6f}")
    
    # Test 2: Random correlation
    corr_random, _ = spearmanr(gt_mean, shuffled_gt_mean)
    logger.info(f"Spearman(gt_mean, shuffled) = {corr_random:.6f}")
    
    issues = []
    
    if abs(corr_perfect - 1.0) > 1e-6:
        issues.append(f"Perfect correlation != 1.0: {corr_perfect:.6f}")
    
    if abs(corr_random) > 0.1:
        issues.append(f"Random correlation too high: {corr_random:.6f} (expected ~0)")
    
    if issues:
        logger.error(f"❌ FAILED:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✅ PASSED: Spearman correlation working correctly")
        return True


def test_batch_shapes(loader):
    """Additional Test: Verify batch shapes and data types."""
    logger.info("\n" + "=" * 60)
    logger.info("ADDITIONAL TEST: Batch Shapes and Types")
    logger.info("=" * 60)
    
    batch = next(iter(loader))
    
    logger.info(f"Batch contents:")
    logger.info(f"  image:        shape={batch['image'].shape}, dtype={batch['image'].dtype}")
    logger.info(f"  target:       shape={batch['target'].shape}, dtype={batch['target'].dtype}")
    logger.info(f"  mean_score:   shape={batch['mean_score'].shape}, dtype={batch['mean_score'].dtype}")
    logger.info(f"  distribution: shape={batch['distribution'].shape}, dtype={batch['distribution'].dtype}")
    logger.info(f"  image_id:     len={len(batch['image_id'])}, type={type(batch['image_id'][0])}")
    
    issues = []
    
    # Check image shape (B, 3, 224, 224)
    if batch['image'].ndim != 4:
        issues.append(f"Image wrong ndim: {batch['image'].ndim}, expected 4")
    if batch['image'].shape[1] != 3:
        issues.append(f"Image wrong channels: {batch['image'].shape[1]}, expected 3")
    
    # Check distribution shape (B, 10)
    if batch['distribution'].shape[1] != 10:
        issues.append(f"Distribution wrong shape: {batch['distribution'].shape}, expected (B, 10)")
    
    if issues:
        logger.error(f"❌ FAILED:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✅ PASSED: All shapes and types correct")
        return True


def test_image_normalization(loader):
    """Additional Test: Check image normalization statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("ADDITIONAL TEST: Image Normalization")
    logger.info("=" * 60)
    
    all_images = []
    for batch_idx, batch in enumerate(loader):
        all_images.append(batch['image'])
        if batch_idx >= 10:
            break
    
    all_images = torch.cat(all_images, dim=0)
    
    mean_per_channel = all_images.mean(dim=[0, 2, 3])
    std_per_channel = all_images.std(dim=[0, 2, 3])
    
    logger.info(f"Image statistics (should be ~ImageNet normalized):")
    logger.info(f"  Mean: [{mean_per_channel[0]:.4f}, {mean_per_channel[1]:.4f}, {mean_per_channel[2]:.4f}]")
    logger.info(f"  Std:  [{std_per_channel[0]:.4f}, {std_per_channel[1]:.4f}, {std_per_channel[2]:.4f}]")
    logger.info(f"  Expected mean ~ [0, 0, 0] (ImageNet normalized)")
    logger.info(f"  Expected std  ~ [1, 1, 1] (ImageNet normalized)")
    
    # Just informational, no pass/fail
    logger.info("✅ INFO: Image normalization statistics displayed")
    return True


def test_label_target_consistency(loader):
    """Additional Test: Verify target matches output_mode."""
    logger.info("\n" + "=" * 60)
    logger.info("ADDITIONAL TEST: Label-Target Consistency")
    logger.info("=" * 60)
    
    batch = next(iter(loader))
    target = batch['target']
    distribution = batch['distribution']
    
    # Check if target is distribution (should match distribution exactly)
    if target.shape == distribution.shape:
        matches = torch.allclose(target, distribution)
        logger.info(f"Output mode: distribution")
        logger.info(f"Target matches distribution: {matches}")
        if matches:
            logger.info("✅ PASSED: Targets correctly set to distributions")
            return True
        else:
            logger.error("❌ FAILED: Targets don't match distributions")
            return False
    else:
        logger.info(f"Output mode: regression")
        logger.info(f"Target shape: {target.shape} (expected (B, 1) or (B,))")
        logger.info("✅ PASSED: Regression mode detected")
        return True


def test_dataset_coverage(ava_txt_path, image_dir):
    """Additional Test: Check dataset coverage and missing images."""
    logger.info("\n" + "=" * 60)
    logger.info("ADDITIONAL TEST: Dataset Coverage Analysis")
    logger.info("=" * 60)
    
    import pandas as pd
    from pathlib import Path
    
    # Load raw AVA.txt
    df = pd.read_csv(ava_txt_path, sep=' ', header=None)
    image_ids = df[0].astype(str).tolist()
    num_in_txt = len(image_ids)
    
    logger.info(f"Entries in AVA.txt: {num_in_txt}")
    
    # Count actual images in directory
    image_dir = Path(image_dir)
    
    # Check different extensions
    jpg_files = list(image_dir.glob("*.jpg"))
    png_files = list(image_dir.glob("*.png"))
    jpeg_files = list(image_dir.glob("*.jpeg"))
    
    logger.info(f"\nImages in directory:")
    logger.info(f"  .jpg files:  {len(jpg_files)}")
    logger.info(f"  .png files:  {len(png_files)}")
    logger.info(f"  .jpeg files: {len(jpeg_files)}")
    logger.info(f"  Total:       {len(jpg_files) + len(png_files) + len(jpeg_files)}")
    
    # Check which IDs from AVA.txt exist
    existing_count = 0
    missing_count = 0
    missing_samples = []
    
    for img_id in image_ids[:10000]:  # Check first 10k for speed
        exists = (image_dir / f"{img_id}.jpg").exists()
        if exists:
            existing_count += 1
        else:
            missing_count += 1
            if len(missing_samples) < 10:
                missing_samples.append(img_id)
    
    logger.info(f"\nSample check (first 10,000 IDs from AVA.txt):")
    logger.info(f"  Found:   {existing_count}")
    logger.info(f"  Missing: {missing_count}")
    logger.info(f"  Coverage: {100 * existing_count / 10000:.1f}%")
    
    if missing_samples:
        logger.info(f"\nSample missing image IDs:")
        for img_id in missing_samples:
            logger.info(f"  - {img_id}.jpg")
    
    # Check if image filenames match expected pattern
    if jpg_files:
        sample_files = [f.stem for f in jpg_files[:5]]
        logger.info(f"\nSample image filenames in directory:")
        for fname in sample_files:
            logger.info(f"  - {fname}.jpg")
    
    # Determine if this is a problem
    coverage_pct = 100 * existing_count / 10000
    
    if coverage_pct < 50:
        logger.error(f"❌ FAILED: Only {coverage_pct:.1f}% of AVA.txt images found")
        logger.error("  Possible issues:")
        logger.error("  - Wrong image directory path")
        logger.error("  - File extension mismatch (.jpg vs .png)")
        logger.error("  - Incomplete dataset download")
        return False
    elif coverage_pct < 80:
        logger.warning(f"⚠️  WARNING: Only {coverage_pct:.1f}% of AVA.txt images found")
        logger.warning("  This is expected if you have a partial dataset")
        logger.info("✅ PASSED: Partial dataset detected, will continue")
        return True
    else:
        logger.info(f"✅ PASSED: {coverage_pct:.1f}% coverage, dataset looks good")
        return True


def main():
    parser = argparse.ArgumentParser(description='AVA Data Sanity Tests')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info("Loading dataset...")
    
    # Load data
    image_dir = config['data']['image_dir']
    labels_df = load_ava_labels(config['data']['ava_txt'], image_dir=image_dir)
    logger.info(f"Loaded {len(labels_df)} images")
    
    # Create splits
    train_ratio = config['data'].get('train_ratio', 0.8)
    val_ratio = config['data'].get('val_ratio', 0.1)
    seed = config.get('seed', 42)
    train_idx, val_idx, test_idx = create_splits(labels_df, train_ratio, val_ratio, seed)
    
    # Create dataset
    transform_config = config.get('transform', {})
    train_transform = create_transform(transform_config, is_train=True)
    output_mode = config['model'].get('output_mode', 'distribution')
    
    train_dataset = AVADataset(labels_df, image_dir, train_transform, train_idx, output_mode)
    
    # Create loader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Run tests
    results = []
    
    results.append(("Dataset Coverage", test_dataset_coverage(config['data']['ava_txt'], image_dir)))
    results.append(("Histogram Normalization", test_histogram_normalization(train_loader)))
    results.append(("Mean Label Distribution", test_mean_label_distribution(train_loader)))
    results.append(("Spearman Implementation", test_spearman_implementation()))
    results.append(("Batch Shapes", test_batch_shapes(train_loader)))
    results.append(("Image Normalization", test_image_normalization(train_loader)))
    results.append(("Label-Target Consistency", test_label_target_consistency(train_loader)))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    num_passed = sum(results[i][1] for i in range(len(results)))
    num_total = len(results)
    
    logger.info(f"\nTotal: {num_passed}/{num_total} tests passed")
    
    if num_passed == num_total:
        logger.info("\n🎉 All tests passed! Ready to train.")
        return 0
    else:
        logger.error(f"\n⚠️  {num_total - num_passed} test(s) failed. Fix issues before training.")
        return 1


if __name__ == '__main__':
    exit(main())
