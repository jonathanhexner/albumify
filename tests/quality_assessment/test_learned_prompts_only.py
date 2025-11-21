"""
Test that learned prompts are loaded correctly without needing PyTorch.
"""

from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prompt_loading():
    """Test that prompts file can be loaded and parsed correctly."""
    logger.info("="*80)
    logger.info("Testing Learned Prompts Loading")
    logger.info("="*80)

    # Load prompts from YAML
    prompts_file = Path("configs/learned_aesthetic_prompts.yaml")

    logger.info(f"\n1. Loading prompts from: {prompts_file}")
    if not prompts_file.exists():
        logger.error(f"   ERROR: File not found!")
        return False

    with open(prompts_file, 'r') as f:
        prompts_data = yaml.safe_load(f)

    # Extract contrastive pairs
    contrastive_pairs = prompts_data.get('contrastive_pairs', [])

    logger.info(f"   SUCCESS: Loaded {len(contrastive_pairs)} contrastive pairs")

    # Show loaded prompts
    logger.info("\n2. Learned prompt pairs:")
    for i, pair in enumerate(contrastive_pairs, 1):
        pos, neg = pair
        logger.info(f"   {i}. Positive: '{pos}'")
        logger.info(f"      Negative: '{neg}'")

    # Verify structure
    logger.info("\n3. Validating structure...")
    all_valid = True
    for i, pair in enumerate(contrastive_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            logger.error(f"   ERROR: Pair {i} is not a 2-element list")
            all_valid = False
        elif not all(isinstance(p, str) for p in pair):
            logger.error(f"   ERROR: Pair {i} elements are not strings")
            all_valid = False

    if all_valid:
        logger.info("   SUCCESS: All pairs are valid")

    logger.info("\n" + "="*80)
    logger.info("Prompt Loading Test Complete!")
    logger.info("="*80)

    return all_valid


if __name__ == "__main__":
    success = test_prompt_loading()
    exit(0 if success else 1)
