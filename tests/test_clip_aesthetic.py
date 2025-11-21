"""
Test CLIP-based aesthetic assessment on PhotoTriage dataset.

This script compares CLIP aesthetic scores against:
1. Rule-based methods (sharpness, contrast, etc.)
2. Other quality assessment methods

Goal: Determine if CLIP text-prompt similarity can predict image quality.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.quality_assessment.clip_aesthetic import CLIPAestheticAssessor
from sim_bench.quality_assessment.rule_based import (
    SharpnessAssessor,
    ContrastAssessor,
    ExposureAssessor,
    CompositeQualityAssessor
)


def test_on_sample_images():
    """Quick test on sample images to verify CLIP aesthetic works."""
    print("="*80)
    print("QUICK TEST: CLIP Aesthetic on Sample Images")
    print("="*80)

    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print(f"Error: {samples_dir} not found")
        return

    sample_images = sorted(list(samples_dir.glob("*.jpg")))[:10]
    print(f"\nTesting on {len(sample_images)} images from {samples_dir}\n")

    # Initialize assessors
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}\n")
    except:
        device = 'cpu'

    clip_assessor = CLIPAestheticAssessor(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
        aggregation_method="weighted"
    )

    sharpness_assessor = SharpnessAssessor()
    contrast_assessor = ContrastAssessor()

    # Assess all images
    results = []
    for img_path in sample_images:
        print(f"Processing: {img_path.name}")

        clip_score = clip_assessor.assess_image(str(img_path))
        sharp_score = sharpness_assessor.assess_image(str(img_path))
        contrast_score = contrast_assessor.assess_image(str(img_path))

        results.append({
            'image': img_path.name,
            'clip_aesthetic': clip_score,
            'sharpness': sharp_score,
            'contrast': contrast_score,
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    # Compute correlations
    print("\n" + "="*80)
    print("CORRELATIONS")
    print("="*80)
    corr_matrix = df[['clip_aesthetic', 'sharpness', 'contrast']].corr()
    print(corr_matrix)

    # Rankings comparison
    print("\n" + "="*80)
    print("RANKING COMPARISON")
    print("="*80)
    print("\nTop 3 by CLIP Aesthetic:")
    top_clip = df.nlargest(3, 'clip_aesthetic')[['image', 'clip_aesthetic']]
    print(top_clip.to_string(index=False))

    print("\nTop 3 by Sharpness:")
    top_sharp = df.nlargest(3, 'sharpness')[['image', 'sharpness']]
    print(top_sharp.to_string(index=False))

    return df


def test_on_phototriage_burst():
    """Test on a PhotoTriage burst sequence (if available)."""
    print("\n" + "="*80)
    print("TEST: CLIP Aesthetic on PhotoTriage Burst")
    print("="*80)

    # Check if PhotoTriage dataset exists
    phototriage_config_path = Path("configs/dataset.phototriage.yaml")
    if not phototriage_config_path.exists():
        print("PhotoTriage config not found, skipping this test")
        return

    import yaml
    with open(phototriage_config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config.get('root', 'D:/PhotoTriage'))
    if not dataset_root.exists():
        print(f"PhotoTriage dataset not found at {dataset_root}, skipping")
        return

    # Find a burst sequence
    burst_dirs = list(dataset_root.glob("burst_*"))
    if not burst_dirs:
        print("No burst sequences found")
        return

    # Use first burst
    burst_dir = burst_dirs[0]
    burst_images = sorted(list(burst_dir.glob("*.jpg")) + list(burst_dir.glob("*.JPG")))

    if len(burst_images) < 3:
        print(f"Burst has only {len(burst_images)} images, skipping")
        return

    print(f"\nTesting on burst: {burst_dir.name}")
    print(f"Images in burst: {len(burst_images)}\n")

    # Initialize assessors
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    clip_assessor = CLIPAestheticAssessor(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
        aggregation_method="weighted"
    )

    composite_assessor = CompositeQualityAssessor()

    # Assess all images in burst
    results = []
    for img_path in burst_images:
        print(f"Processing: {img_path.name}")

        clip_score = clip_assessor.assess_image(str(img_path))
        composite_score = composite_assessor.assess_image(str(img_path))

        # Get detailed CLIP scores
        detailed = clip_assessor.get_detailed_scores(str(img_path))

        results.append({
            'image': img_path.name,
            'clip_aesthetic': clip_score,
            'composite_quality': composite_score,
            'detailed': detailed
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("BURST RESULTS")
    print("="*80)
    print(df[['image', 'clip_aesthetic', 'composite_quality']].to_string(index=False))

    # Show best image by each method
    print("\n" + "="*80)
    print("BEST IMAGE SELECTION")
    print("="*80)

    best_clip_idx = df['clip_aesthetic'].idxmax()
    best_composite_idx = df['composite_quality'].idxmax()

    print(f"\nBest by CLIP Aesthetic: {df.loc[best_clip_idx, 'image']}")
    print(f"  CLIP Score: {df.loc[best_clip_idx, 'clip_aesthetic']:.4f}")
    print(f"  Composite Score: {df.loc[best_clip_idx, 'composite_quality']:.4f}")

    print(f"\nBest by Composite Quality: {df.loc[best_composite_idx, 'image']}")
    print(f"  CLIP Score: {df.loc[best_composite_idx, 'clip_aesthetic']:.4f}")
    print(f"  Composite Score: {df.loc[best_composite_idx, 'composite_quality']:.4f}")

    agreement = best_clip_idx == best_composite_idx
    print(f"\nMethods Agree: {'YES' if agreement else 'NO'}")

    # Show detailed CLIP scores for best image
    print("\n" + "="*80)
    print(f"DETAILED CLIP SCORES (Best Image)")
    print("="*80)
    best_detailed = results[best_clip_idx]['detailed']
    for key, value in sorted(best_detailed.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key:40s}: {value:7.4f}")

    return df


def compare_aggregation_methods():
    """Compare different aggregation strategies."""
    print("\n" + "="*80)
    print("TEST: Comparing Aggregation Methods")
    print("="*80)

    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print(f"Error: {samples_dir} not found")
        return

    sample_images = sorted(list(samples_dir.glob("*.jpg")))[:5]
    print(f"\nTesting {len(sample_images)} images with different aggregation methods\n")

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    aggregation_methods = ['contrastive_only', 'weighted', 'mean']

    results = {method: [] for method in aggregation_methods}

    for method in aggregation_methods:
        print(f"\nTesting aggregation: {method}")
        assessor = CLIPAestheticAssessor(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=device,
            aggregation_method=method
        )

        for img_path in sample_images:
            score = assessor.assess_image(str(img_path))
            results[method].append(score)

    # Compare results
    df = pd.DataFrame(results)
    df['image'] = [img.name for img in sample_images]
    df = df[['image'] + aggregation_methods]

    print("\n" + "="*80)
    print("AGGREGATION METHOD COMPARISON")
    print("="*80)
    print(df.to_string(index=False))

    # Correlations between methods
    print("\n" + "="*80)
    print("CORRELATIONS BETWEEN AGGREGATION METHODS")
    print("="*80)
    corr = df[aggregation_methods].corr()
    print(corr)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CLIP AESTHETIC ASSESSMENT EVALUATION")
    print("="*80)
    print("\nThis script tests whether CLIP text-prompt similarity")
    print("can effectively assess image aesthetic quality.\n")

    # Test 1: Quick verification on samples
    try:
        print("\n[TEST 1] Quick verification on sample images")
        test_on_sample_images()
    except Exception as e:
        print(f"\nTest 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: PhotoTriage burst (if available)
    try:
        print("\n[TEST 2] PhotoTriage burst sequence")
        test_on_phototriage_burst()
    except Exception as e:
        print(f"\nTest 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Aggregation method comparison
    try:
        print("\n[TEST 3] Aggregation method comparison")
        compare_aggregation_methods()
    except Exception as e:
        print(f"\nTest 3 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Examine correlations with rule-based methods")
    print("2. Test on full PhotoTriage benchmark")
    print("3. Compare with NIMA/MUSIQ if available")
    print("4. Fine-tune prompt engineering based on results")
    print("="*80)


if __name__ == "__main__":
    main()
