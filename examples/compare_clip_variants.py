"""
Compare CLIP Aesthetic Assessors: Hardcoded vs Learned Prompts

This demo shows the difference between:
1. CLIPAestheticAssessor - Uses manually designed prompts
2. LearnedCLIPAestheticAssessor - Uses prompts learned from PhotoTriage dataset

The learned prompts are based on analysis of 34,827 user feedback reasons
from the PhotoTriage dataset, categorized into 6 themes:
- Focus/Sharpness (blurry, out of focus, sharp)
- Composition (cluttered, cropped, framing)
- Exposure/Lighting (dark, bright, overexposed)
- Color/Clarity (bad color, hazy, vibrant)
- Content/Interest (boring, interesting, lacks subject)
- View/Perspective (narrow, wide, shows detail)
"""

import sys
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.config import setup_logging
from sim_bench.quality_assessment.clip_aesthetic import (
    CLIPAestheticAssessor,
    LearnedCLIPAestheticAssessor
)

logger = logging.getLogger(__name__)


def compare_prompts():
    """Compare prompt sets between hardcoded and learned variants."""
    logger.info("="*80)
    logger.info("CLIP Aesthetic Prompts Comparison")
    logger.info("="*80)

    logger.info("\n" + "="*80)
    logger.info("HARDCODED PROMPTS (Manual Design)")
    logger.info("="*80)

    logger.info("\nContrastive Pairs:")
    for i, (pos, neg) in enumerate(CLIPAestheticAssessor.CONTRASTIVE_PAIRS, 1):
        logger.info(f"  {i}. '{pos}' vs '{neg}'")

    logger.info("\nPositive Attributes:")
    for i, attr in enumerate(CLIPAestheticAssessor.POSITIVE_ATTRIBUTES, 1):
        logger.info(f"  {i}. '{attr}'")

    logger.info("\nNegative Attributes:")
    for i, attr in enumerate(CLIPAestheticAssessor.NEGATIVE_ATTRIBUTES, 1):
        logger.info(f"  {i}. '{attr}'")

    logger.info(f"\nTotal prompts: {len(CLIPAestheticAssessor.CONTRASTIVE_PAIRS) * 2 + len(CLIPAestheticAssessor.POSITIVE_ATTRIBUTES) + len(CLIPAestheticAssessor.NEGATIVE_ATTRIBUTES)}")

    logger.info("\n" + "="*80)
    logger.info("LEARNED PROMPTS (From PhotoTriage Dataset)")
    logger.info("="*80)

    # Load learned prompts via assessor
    try:
        learned = LearnedCLIPAestheticAssessor(
            prompts_file="configs/learned_aesthetic_prompts.yaml",
            model_name="ViT-B-32",
            device="cpu"
        )

        logger.info("\nContrastive Pairs:")
        for i, (pos, neg) in enumerate(learned.CONTRASTIVE_PAIRS, 1):
            logger.info(f"  {i}. '{pos}' vs '{neg}'")

        logger.info(f"\nTotal prompts: {len(learned.CONTRASTIVE_PAIRS) * 2}")
        logger.info("(No separate positive/negative attributes - all from contrastive pairs)")

    except ImportError as e:
        logger.warning(f"\nCannot load learned assessor (PyTorch not available): {e}")
        logger.info("Showing prompts from YAML file instead...")

        import yaml
        with open("configs/learned_aesthetic_prompts.yaml", 'r') as f:
            prompts_data = yaml.safe_load(f)

        logger.info("\nContrastive Pairs:")
        for i, (pos, neg) in enumerate(prompts_data['contrastive_pairs'], 1):
            logger.info(f"  {i}. '{pos}' vs '{neg}'")

        logger.info(f"\nTotal prompts: {len(prompts_data['contrastive_pairs']) * 2}")

    logger.info("\n" + "="*80)
    logger.info("KEY DIFFERENCES")
    logger.info("="*80)
    logger.info("""
1. HARDCODED (4 pairs + 2 pos + 2 neg = 10 total prompts):
   - Designed based on general photography principles
   - Focuses on composition, placement, cropping, quality
   - Includes separate positive/negative attributes

2. LEARNED (9 pairs = 18 total prompts):
   - Derived from 34,827 real user feedback reasons
   - Covers broader range of quality dimensions:
     * Focus/Sharpness
     * Composition & Framing
     * Exposure/Lighting
     * Color/Clarity
     * Content/Interest
     * View/Perspective
     * Detail Visibility
   - All assessment from contrastive pairs (no separate attributes)
   - Aligned with how real users evaluate photo quality
    """)

    logger.info("="*80)


def compare_on_images():
    """Compare assessors on actual images (requires PyTorch)."""
    logger.info("\n" + "="*80)
    logger.info("IMAGE ASSESSMENT COMPARISON")
    logger.info("="*80)

    try:
        # Create both assessors
        logger.info("\nInitializing assessors...")
        hardcoded = CLIPAestheticAssessor(
            model_name="ViT-B-32",
            device="cpu",
            aggregation_method="weighted"
        )

        learned = LearnedCLIPAestheticAssessor(
            prompts_file="configs/learned_aesthetic_prompts.yaml",
            model_name="ViT-B-32",
            device="cpu",
            aggregation_method="weighted"
        )

        # Find test images
        test_dirs = [
            Path("D:/Budapest2025_Google"),
            Path("D:/Similar Images/automatic_triage_photo_series/train_val/train_val_imgs")
        ]

        test_images = []
        for test_dir in test_dirs:
            if test_dir.exists():
                images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG"))
                test_images = images[:5]
                break

        if not test_images:
            logger.warning("No test images found")
            return

        logger.info(f"\nComparing on {len(test_images)} images...")
        logger.info("-"*80)

        results = []
        for img_path in test_images:
            hardcoded_score = hardcoded.assess_image(str(img_path))
            learned_score = learned.assess_image(str(img_path))

            diff = learned_score - hardcoded_score

            results.append({
                'name': img_path.name,
                'hardcoded': hardcoded_score,
                'learned': learned_score,
                'diff': diff
            })

            logger.info(f"\n{img_path.name}:")
            logger.info(f"  Hardcoded: {hardcoded_score:+.4f}")
            logger.info(f"  Learned:   {learned_score:+.4f}")
            logger.info(f"  Diff:      {diff:+.4f}")

        # Summary
        logger.info("\n" + "-"*80)
        logger.info("SUMMARY:")
        avg_diff = sum(r['diff'] for r in results) / len(results)
        logger.info(f"  Average difference: {avg_diff:+.4f}")
        logger.info(f"  Learned scores higher: {sum(1 for r in results if r['diff'] > 0)}/{len(results)}")
        logger.info(f"  Hardcoded scores higher: {sum(1 for r in results if r['diff'] < 0)}/{len(results)}")

    except ImportError as e:
        logger.warning(f"\nCannot run image comparison (PyTorch not available): {e}")
        logger.info("Install PyTorch to enable image assessment comparison")

    logger.info("\n" + "="*80)


def main():
    """Run comparison demos."""
    setup_logging()

    try:
        compare_prompts()
    except Exception as e:
        logger.error(f"Prompt comparison failed: {e}", exc_info=True)

    try:
        compare_on_images()
    except Exception as e:
        logger.error(f"Image comparison failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
