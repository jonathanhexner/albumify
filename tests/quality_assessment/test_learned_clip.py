"""
Quick test to verify LearnedCLIPAestheticAssessor works correctly.
"""

import sys
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.config import setup_logging
from sim_bench.quality_assessment.clip_aesthetic import LearnedCLIPAestheticAssessor

logger = logging.getLogger(__name__)


def test_learned_clip():
    """Test that learned CLIP assessor loads and works."""
    logger.info("="*80)
    logger.info("Testing LearnedCLIPAestheticAssessor")
    logger.info("="*80)

    # Create assessor
    logger.info("\n1. Initializing with learned prompts...")
    assessor = LearnedCLIPAestheticAssessor(
        prompts_file="configs/learned_aesthetic_prompts.yaml",
        model_name="ViT-B-32",
        device="cpu",
        aggregation_method="weighted"
    )

    logger.info(f"   {assessor}")
    logger.info(f"   Loaded {len(assessor.CONTRASTIVE_PAIRS)} contrastive pairs")

    # Show loaded prompts
    logger.info("\n2. Learned prompt pairs:")
    for i, (pos, neg) in enumerate(assessor.CONTRASTIVE_PAIRS, 1):
        logger.info(f"   {i}. '{pos}' vs '{neg}'")

    # Find test image
    logger.info("\n3. Testing on sample image...")
    test_dirs = [
        Path("D:/Budapest2025_Google"),
        Path("D:/Similar Images/automatic_triage_photo_series/train_val/train_val_imgs")
    ]

    test_image = None
    for test_dir in test_dirs:
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG"))
            if images:
                test_image = images[0]
                break

    if not test_image:
        logger.warning("   No test images found, skipping assessment test")
        return

    logger.info(f"   Using: {test_image.name}")

    # Assess image
    overall_score = assessor.assess_image(str(test_image))
    logger.info(f"   Overall score: {overall_score:.4f}")

    # Get detailed scores
    detailed = assessor.get_detailed_scores(str(test_image))
    if detailed:
        logger.info("\n4. Detailed scores:")
        for key, value in sorted(detailed.items()):
            if key.startswith('contrast_'):
                logger.info(f"   {key}: {value:+.4f}")

    logger.info("\n" + "="*80)
    logger.info("SUCCESS: LearnedCLIPAestheticAssessor is working!")
    logger.info("="*80)


if __name__ == "__main__":
    setup_logging()
    test_learned_clip()
