"""
Quick test to verify CLIP aesthetic integrates with quality benchmark.
"""

import sys
from pathlib import Path

def test_import():
    """Test that CLIP aesthetic can be imported."""
    print("="*80)
    print("TEST 1: Import CLIP Aesthetic")
    print("="*80)

    try:
        from sim_bench.quality_assessment import CLIPAestheticAssessor, load_quality_method
        print("[OK] CLIPAestheticAssessor imported successfully")
        print("[OK] load_quality_method imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_factory():
    """Test that factory function can load CLIP aesthetic."""
    print("\n" + "="*80)
    print("TEST 2: Factory Function")
    print("="*80)

    try:
        from sim_bench.quality_assessment.factory import load_quality_method

        config = {
            'model_name': 'ViT-B-32',
            'pretrained': 'laion2b_s34b_b79k',
            'device': 'cpu',
            'aggregation_method': 'weighted'
        }

        method = load_quality_method('clip_aesthetic', config)
        print(f"[OK] Method loaded: {method.__class__.__name__}")
        print(f"[OK] Model: {method.model_name}")
        print(f"[OK] Aggregation: {method.aggregation_method}")
        return True

    except Exception as e:
        print(f"[FAIL] Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_assess_image():
    """Test that CLIP aesthetic can assess a sample image."""
    print("\n" + "="*80)
    print("TEST 3: Assess Sample Image")
    print("="*80)

    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print("[SKIP] Skipping (no sample images)")
        return None

    sample_images = list(samples_dir.glob("*.jpg"))
    if not sample_images:
        print("[SKIP] Skipping (no sample images)")
        return None

    try:
        from sim_bench.quality_assessment.factory import load_quality_method

        config = {
            'model_name': 'ViT-B-32',
            'pretrained': 'laion2b_s34b_b79k',
            'device': 'cpu',
            'aggregation_method': 'weighted'
        }

        method = load_quality_method('clip_aesthetic', config)

        # Test on first image
        test_image = str(sample_images[0])
        print(f"Testing on: {sample_images[0].name}")

        score = method.assess_image(test_image)
        print(f"[OK] Score: {score:.6f}")

        # Get detailed scores
        if hasattr(method, 'get_detailed_scores'):
            detailed = method.get_detailed_scores(test_image)
            if detailed:
                print(f"[OK] Detailed scores available ({len(detailed)} prompts)")
                # Show top 3
                contrastive = {k: v for k, v in detailed.items() if k.startswith('contrast_')}
                if contrastive:
                    print("\n  Top contrastive scores:")
                    for i, (k, v) in enumerate(sorted(contrastive.items(), key=lambda x: x[1], reverse=True)[:3]):
                        print(f"    {i+1}. {k}: {v:.4f}")

        return True

    except Exception as e:
        print(f"[FAIL] Assessment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompts():
    """Test that prompts are correctly configured."""
    print("\n" + "="*80)
    print("TEST 4: Verify Prompts")
    print("="*80)

    try:
        from sim_bench.quality_assessment.clip_aesthetic import CLIPAestheticAssessor

        print("Contrastive pairs:")
        for i, (pos, neg) in enumerate(CLIPAestheticAssessor.CONTRASTIVE_PAIRS, 1):
            print(f"  {i}. '{pos}' vs '{neg}'")

        print("\nPositive attributes:")
        for i, prompt in enumerate(CLIPAestheticAssessor.POSITIVE_ATTRIBUTES, 1):
            print(f"  {i}. '{prompt}'")

        print("\nNegative attributes:")
        for i, prompt in enumerate(CLIPAestheticAssessor.NEGATIVE_ATTRIBUTES, 1):
            print(f"  {i}. '{prompt}'")

        total_prompts = (
            len(CLIPAestheticAssessor.CONTRASTIVE_PAIRS) * 2 +
            len(CLIPAestheticAssessor.POSITIVE_ATTRIBUTES) +
            len(CLIPAestheticAssessor.NEGATIVE_ATTRIBUTES)
        )
        print(f"\n[OK] Total prompts: {total_prompts}")

        return True

    except Exception as e:
        print(f"[FAIL] Prompt test failed: {e}")
        return False


def test_config():
    """Test that config file is valid."""
    print("\n" + "="*80)
    print("TEST 5: Verify Config File")
    print("="*80)

    config_path = Path("configs/quality_benchmark.deep_learning.yaml")
    if not config_path.exists():
        print(f"[FAIL] Config not found: {config_path}")
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check for clip_aesthetic method
        methods = config.get('methods', [])
        clip_method = None
        for method in methods:
            if method.get('name') == 'clip_aesthetic':
                clip_method = method
                break

        if clip_method is None:
            print("[FAIL] clip_aesthetic not found in config")
            return False

        print(f"[OK] Found clip_aesthetic in config")
        print(f"  Type: {clip_method.get('type')}")

        method_config = clip_method.get('config', {})
        print(f"  Model: {method_config.get('model_name')}")
        print(f"  Pretrained: {method_config.get('pretrained')}")
        print(f"  Device: {method_config.get('device')}")
        print(f"  Aggregation: {method_config.get('aggregation_method')}")

        return True

    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CLIP AESTHETIC INTEGRATION TEST")
    print("="*80)
    print("Testing integration with quality benchmark framework\n")

    results = []

    # Run tests
    results.append(("Import", test_import()))
    results.append(("Factory", test_factory()))
    results.append(("Assess", test_assess_image()))
    results.append(("Prompts", test_prompts()))
    results.append(("Config", test_config()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)

    for name, result in results:
        status = "[OK] PASS" if result is True else "[FAIL] FAIL" if result is False else "[SKIP] SKIP"
        print(f"{name:20s}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    print("\n" + "="*80)
    if failed == 0:
        print("[OK] ALL TESTS PASSED - Ready to run benchmark!")
        print("="*80)
        print("\nNext steps:")
        print("  python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml")
    else:
        print("[FAIL] SOME TESTS FAILED - Fix errors before running benchmark")
        print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
