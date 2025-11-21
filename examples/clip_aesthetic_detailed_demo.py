"""
CLIP Aesthetic Detailed Scores Demo

Shows how to get individual scores for each aesthetic attribute:
- Composition scores (well-composed vs poorly-composed)
- Subject placement scores
- Cropping scores
- Overall quality scores
- Positive attributes (professional, pleasing)
- Negative attributes (amateur, poor framing)
"""

import sys
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.config import setup_logging
from sim_bench.quality_assessment.clip_aesthetic import CLIPAestheticAssessor

logger = logging.getLogger(__name__)


def demo_detailed_scores():
    """
    Demonstrate getting detailed scores for each aesthetic attribute.
    """
    logger.info("="*80)
    logger.info("CLIP Aesthetic - Detailed Scores Demo")
    logger.info("="*80)

    # Create assessor
    logger.info("\nInitializing CLIP Aesthetic Assessor...")
    assessor = CLIPAestheticAssessor(
        model_name="ViT-B-32",
        device="cpu",
        aggregation_method="weighted"
    )

    # Show the prompts being used
    logger.info("\nContrastive Pairs:")
    for i, (pos, neg) in enumerate(assessor.CONTRASTIVE_PAIRS, 1):
        logger.info(f"  {i}. Positive: '{pos}'")
        logger.info(f"     Negative: '{neg}'")

    logger.info("\nPositive Attributes:")
    for i, attr in enumerate(assessor.POSITIVE_ATTRIBUTES, 1):
        logger.info(f"  {i}. '{attr}'")

    logger.info("\nNegative Attributes:")
    for i, attr in enumerate(assessor.NEGATIVE_ATTRIBUTES, 1):
        logger.info(f"  {i}. '{attr}'")

    # Find sample images
    samples_dir = Path("D:/Budapest2025_Google")
    if not samples_dir.exists():
        logger.error(f"Directory not found: {samples_dir}")
        logger.info("Please update the path to your images directory")
        return

    sample_images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg"))
    if not sample_images:
        logger.error("No images found in directory")
        return

    # Analyze first 5 images
    num_images = min(5, len(sample_images))
    logger.info(f"\nAnalyzing {num_images} images...")
    logger.info("="*80)

    for img_path in sample_images[:num_images]:
        logger.info(f"\nImage: {img_path.name}")
        logger.info("-"*80)

        # Get overall score
        overall_score = assessor.assess_image(str(img_path))
        logger.info(f"\nOverall Aesthetic Score: {overall_score:.4f}")

        # Get detailed scores
        detailed = assessor.get_detailed_scores(str(img_path))

        if detailed:
            # Show contrastive pair scores
            logger.info("\nContrastive Pair Scores (positive - negative):")
            for key, value in detailed.items():
                if key.startswith('contrast_'):
                    # Extract index and prompt text
                    parts = key.split('_', 2)
                    idx = int(parts[1])
                    prompt_text = parts[2] if len(parts) > 2 else ''

                    pos_text, neg_text = assessor.CONTRASTIVE_PAIRS[idx]
                    logger.info(f"  Pair {idx+1}: {value:+.4f}")
                    logger.info(f"    '{pos_text}' vs '{neg_text}'")

            # Show individual positive/negative scores for first pair as example
            logger.info("\nFirst Pair - Individual Scores:")
            pos_key = f"pos_0_{assessor.CONTRASTIVE_PAIRS[0][0][:30]}"
            neg_key = f"neg_0_{assessor.CONTRASTIVE_PAIRS[0][1][:30]}"

            if pos_key in detailed:
                logger.info(f"  Positive ('{assessor.CONTRASTIVE_PAIRS[0][0]}'): {detailed[pos_key]:.4f}")
            if neg_key in detailed:
                logger.info(f"  Negative ('{assessor.CONTRASTIVE_PAIRS[0][1]}'): {detailed[neg_key]:.4f}")

            # Show positive attributes
            logger.info("\nPositive Attribute Scores:")
            for key, value in detailed.items():
                if key.startswith('positive_'):
                    parts = key.split('_', 2)
                    idx = int(parts[1])
                    attr_text = assessor.POSITIVE_ATTRIBUTES[idx]
                    logger.info(f"  '{attr_text}': {value:.4f}")

            # Show negative attributes
            logger.info("\nNegative Attribute Scores (lower is better):")
            for key, value in detailed.items():
                if key.startswith('negative_'):
                    parts = key.split('_', 2)
                    idx = int(parts[1])
                    attr_text = assessor.NEGATIVE_ATTRIBUTES[idx]
                    logger.info(f"  '{attr_text}': {value:.4f}")

        logger.info("-"*80)

    logger.info("\n" + "="*80)
    logger.info("Demo Complete")
    logger.info("="*80)
    logger.info("\nKey Insights:")
    logger.info("  - Overall score: Weighted combination of all attributes")
    logger.info("  - Contrastive scores: Difference between positive and negative")
    logger.info("  - Higher positive scores = better aesthetic quality")
    logger.info("  - Lower negative scores = better aesthetic quality")
    logger.info("="*80)


def demo_batch_comparison():
    """
    Compare aesthetic scores across multiple images.
    """
    logger.info("\n\n" + "="*80)
    logger.info("CLIP Aesthetic - Batch Comparison Demo")
    logger.info("="*80)

    assessor = CLIPAestheticAssessor(
        model_name="ViT-B-32",
        device="cpu",
        aggregation_method="weighted"
    )

    # Find sample images
    samples_dir = Path("D:/Budapest2025_Google")
    if not samples_dir.exists():
        return

    sample_images = list(samples_dir.glob("*.jpg"))[:10]
    if len(sample_images) < 2:
        logger.warning("Need at least 2 images for comparison")
        return

    logger.info(f"\nComparing {len(sample_images)} images...")

    # Analyze all images
    results = []
    for img_path in sample_images:
        overall = assessor.assess_image(str(img_path))
        detailed = assessor.get_detailed_scores(str(img_path))

        results.append({
            'path': img_path,
            'overall': overall,
            'detailed': detailed
        })

    # Sort by overall score
    results.sort(key=lambda x: x['overall'], reverse=True)

    logger.info("\nRanked by Overall Aesthetic Quality:")
    logger.info("-"*80)

    for i, result in enumerate(results, 1):
        logger.info(f"{i}. {result['path'].name}")
        logger.info(f"   Overall Score: {result['overall']:.4f}")

        # Show composition score (first contrastive pair)
        if result['detailed']:
            comp_key = f"contrast_0_{assessor.CONTRASTIVE_PAIRS[0][0][:30]}"
            if comp_key in result['detailed']:
                logger.info(f"   Composition:   {result['detailed'][comp_key]:+.4f}")

    logger.info("="*80)

    # Show best and worst by each attribute
    logger.info("\nBest/Worst by Attribute:")
    logger.info("-"*80)

    for i, (pos_text, neg_text) in enumerate(assessor.CONTRASTIVE_PAIRS):
        key = f"contrast_{i}_{pos_text[:30]}"

        # Sort by this attribute
        attr_results = sorted(results, key=lambda x: x['detailed'].get(key, 0), reverse=True)

        best = attr_results[0]
        worst = attr_results[-1]

        logger.info(f"\n{pos_text}:")
        logger.info(f"  Best:  {best['path'].name} ({best['detailed'].get(key, 0):+.4f})")
        logger.info(f"  Worst: {worst['path'].name} ({worst['detailed'].get(key, 0):+.4f})")

    logger.info("="*80)


def main():
    """Run all demos."""
    setup_logging()

    try:
        demo_detailed_scores()
    except Exception as e:
        logger.error(f"Demo 1 failed: {e}", exc_info=True)

    try:
        demo_batch_comparison()
    except Exception as e:
        logger.error(f"Demo 2 failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
