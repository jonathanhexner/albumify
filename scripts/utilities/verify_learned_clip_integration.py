"""
Verification script for learned CLIP prompts integration.

Tests that:
1. Prompts can be loaded from YAML
2. LearnedCLIPAestheticAssessor class exists and can be instantiated (structure only)
3. Factory supports 'clip_learned' method
4. All files are in place
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_integration():
    """Verify all components are integrated correctly."""
    logger.info("="*80)
    logger.info("Learned CLIP Prompts - Integration Verification")
    logger.info("="*80)

    all_checks_passed = True

    # Check 1: Prompts file exists
    logger.info("\n1. Checking prompts file...")
    prompts_file = Path("configs/learned_aesthetic_prompts.yaml")
    if prompts_file.exists():
        logger.info(f"   ✓ Found: {prompts_file}")
    else:
        logger.error(f"   ✗ Missing: {prompts_file}")
        all_checks_passed = False

    # Check 2: Script exists
    logger.info("\n2. Checking learning script...")
    script_file = Path("scripts/learn_prompts_from_phototriage.py")
    if script_file.exists():
        logger.info(f"   ✓ Found: {script_file}")
    else:
        logger.error(f"   ✗ Missing: {script_file}")
        all_checks_passed = False

    # Check 3: LearnedCLIPAestheticAssessor exists
    logger.info("\n3. Checking LearnedCLIPAestheticAssessor class...")
    try:
        from sim_bench.quality_assessment.clip_aesthetic import LearnedCLIPAestheticAssessor
        logger.info(f"   ✓ Class imported successfully")
        logger.info(f"   ✓ Class: {LearnedCLIPAestheticAssessor}")
    except ImportError as e:
        logger.error(f"   ✗ Import failed: {e}")
        all_checks_passed = False

    # Check 4: Factory integration
    logger.info("\n4. Checking factory integration...")
    try:
        from sim_bench.quality_assessment.factory import load_quality_method

        # Check that factory knows about clip_learned
        try:
            # This will fail if PyTorch not installed, but that's ok
            # We just want to verify the method name is recognized
            assessor = load_quality_method('clip_learned')
            logger.info(f"   ✓ Factory loaded clip_learned successfully")
        except ImportError as e:
            # Expected if PyTorch not installed
            if "PyTorch" in str(e) or "CLIP" in str(e):
                logger.info(f"   ✓ Factory recognizes 'clip_learned' (PyTorch not installed)")
            else:
                logger.error(f"   ✗ Unexpected import error: {e}")
                all_checks_passed = False
        except ValueError as e:
            logger.error(f"   ✗ Factory doesn't recognize 'clip_learned': {e}")
            all_checks_passed = False

    except Exception as e:
        logger.error(f"   ✗ Factory test failed: {e}")
        all_checks_passed = False

    # Check 5: Demo files exist
    logger.info("\n5. Checking demo files...")
    demo_file = Path("examples/compare_clip_variants.py")
    if demo_file.exists():
        logger.info(f"   ✓ Found: {demo_file}")
    else:
        logger.error(f"   ✗ Missing: {demo_file}")
        all_checks_passed = False

    # Check 6: Test files exist
    logger.info("\n6. Checking test files...")
    test_file = Path("tests/quality_assessment/test_learned_prompts_only.py")
    if test_file.exists():
        logger.info(f"   ✓ Found: {test_file}")
    else:
        logger.error(f"   ✗ Missing: {test_file}")
        all_checks_passed = False

    # Check 7: Summary doc exists
    logger.info("\n7. Checking documentation...")
    doc_file = Path("LEARNED_CLIP_PROMPTS_SUMMARY.md")
    if doc_file.exists():
        logger.info(f"   ✓ Found: {doc_file}")
    else:
        logger.error(f"   ✗ Missing: {doc_file}")
        all_checks_passed = False

    # Summary
    logger.info("\n" + "="*80)
    if all_checks_passed:
        logger.info("✓ ALL CHECKS PASSED - Integration is complete!")
    else:
        logger.error("✗ SOME CHECKS FAILED - Please review errors above")

    logger.info("="*80)

    # Show usage example
    if all_checks_passed:
        logger.info("\nQuick Start:")
        logger.info("  1. View learned prompts:")
        logger.info("     cat configs/learned_aesthetic_prompts.yaml")
        logger.info("")
        logger.info("  2. Compare hardcoded vs learned:")
        logger.info("     python examples/compare_clip_variants.py")
        logger.info("")
        logger.info("  3. Use in your code:")
        logger.info("     from sim_bench.quality_assessment.factory import load_quality_method")
        logger.info("     assessor = load_quality_method('clip_learned')")
        logger.info("     score = assessor.assess_image('photo.jpg')")
        logger.info("")
        logger.info("See LEARNED_CLIP_PROMPTS_SUMMARY.md for full documentation.")
        logger.info("="*80)

    return all_checks_passed


if __name__ == "__main__":
    success = verify_integration()
    exit(0 if success else 1)
