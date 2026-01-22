"""
Tests for Model Hub module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that model hub modules import correctly."""
    print("\n" + "="*80)
    print("TEST: Model Hub Imports")
    print("="*80)

    from sim_bench.model_hub import ModelHub, ImageMetrics
    print("[OK] ModelHub and ImageMetrics imported")
    return True


def test_hub_initialization():
    """Test ModelHub initialization with config."""
    print("\n" + "="*80)
    print("TEST: ModelHub Initialization")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.model_hub import ModelHub

    config = get_global_config().to_dict()
    hub = ModelHub(config)
    print(f"[OK] ModelHub initialized with device={hub._device}")
    return True


def test_image_metrics():
    """Test ImageMetrics dataclass and composite scoring."""
    print("\n" + "="*80)
    print("TEST: ImageMetrics")
    print("="*80)

    from sim_bench.model_hub import ImageMetrics

    metrics = ImageMetrics(
        image_path="test.jpg",
        iqa_score=0.8,
        ava_score=7.5,
        has_face=True,
        is_portrait=True,
        eyes_open=True,
        is_smiling=True,
        smile_score=0.9
    )
    print(f"[OK] Created ImageMetrics: {metrics.image_path}")

    composite = metrics.get_composite_score()
    print(f"[OK] Composite score: {composite:.3f}")

    return composite > 0 and composite <= 1


def test_quality_scoring():
    """Test quality scoring with real image if available."""
    print("\n" + "="*80)
    print("TEST: Quality Scoring")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.model_hub import ModelHub

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True

    image_files = list(test_images.glob("*.jpg"))
    if not image_files:
        print("[SKIP] No test images found")
        return True

    test_img = image_files[0]
    print(f"[INFO] Testing with: {test_img.name}")

    config = get_global_config().to_dict()
    hub = ModelHub(config)

    scores = hub.score_quality(test_img)
    print(f"[OK] Quality scores: overall={scores['overall']:.3f}")
    print(f"     sharpness={scores['sharpness']:.3f}, exposure={scores['exposure']:.3f}")

    return scores['overall'] >= 0 and scores['overall'] <= 1


def test_portrait_analysis():
    """Test portrait analysis with real image if available."""
    print("\n" + "="*80)
    print("TEST: Portrait Analysis")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.model_hub import ModelHub

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True

    image_files = list(test_images.glob("*.jpg"))
    if not image_files:
        print("[SKIP] No test images found")
        return True

    test_img = image_files[0]
    print(f"[INFO] Testing with: {test_img.name}")

    config = get_global_config().to_dict()
    hub = ModelHub(config)

    portrait_metrics = hub.analyze_portrait(test_img)
    print(f"[OK] Portrait analysis complete")
    print(f"     has_face={portrait_metrics.has_face}, is_portrait={portrait_metrics.is_portrait}")

    return True


def test_unified_analysis():
    """Test unified image analysis."""
    print("\n" + "="*80)
    print("TEST: Unified Image Analysis")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.model_hub import ModelHub

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True

    image_files = list(test_images.glob("*.jpg"))
    if not image_files:
        print("[SKIP] No test images found")
        return True

    test_img = image_files[0]
    print(f"[INFO] Testing with: {test_img.name}")

    config = get_global_config().to_dict()
    hub = ModelHub(config)

    metrics = hub.analyze_image(test_img)
    print(f"[OK] Unified analysis complete")
    print(f"     IQA={metrics.iqa_score:.3f}, portrait={metrics.is_portrait}")

    return metrics.image_path == str(test_img)


def test_batch_analysis():
    """Test batch analysis of multiple images."""
    print("\n" + "="*80)
    print("TEST: Batch Analysis")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.model_hub import ModelHub

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True

    image_files = list(test_images.glob("*.jpg"))[:3]
    if len(image_files) < 2:
        print("[SKIP] Need at least 2 test images")
        return True

    print(f"[INFO] Testing with {len(image_files)} images")

    config = get_global_config().to_dict()
    hub = ModelHub(config)

    def progress(current, total):
        print(f"     Progress: {current}/{total}")

    results = hub.analyze_batch(
        image_files,
        include_quality=True,
        include_portrait=True,
        progress_callback=progress
    )

    print(f"[OK] Batch analysis complete: {len(results)} images")

    return len(results) == len(image_files)


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("MODEL HUB TEST SUITE")
    print("="*80)

    tests = [
        test_imports,
        test_hub_initialization,
        test_image_metrics,
        test_quality_scoring,
        test_portrait_analysis,
        test_unified_analysis,
        test_batch_analysis,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
