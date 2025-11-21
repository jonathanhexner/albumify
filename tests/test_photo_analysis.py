"""
Tests for photo analysis module.

Tests thumbnail generation, CLIP tagging, and batch processing.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules import correctly."""
    print("\n" + "="*80)
    print("TEST: Module Imports")
    print("="*80)

    try:
        from sim_bench.config import get_global_config, setup_logging
        print("[OK] Config module imports")

        from sim_bench.image_processing import ThumbnailGenerator, create_image_processor
        print("[OK] Image processing module imports")

        from sim_bench.photo_analysis import CLIPTagger, create_photo_analyzer
        print("[OK] Photo analysis module imports")

        return True

    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_global_config():
    """Test global configuration system."""
    print("\n" + "="*80)
    print("TEST: Global Configuration")
    print("="*80)

    from sim_bench.config import get_global_config

    config = get_global_config()

    # Test basic gets
    device = config.get('device', 'cpu')
    print(f"[OK] Device: {device}")

    cache_dir = config.get_path('cache_dir')
    print(f"[OK] Cache dir: {cache_dir}")

    clip_model = config.get('clip.model_name', 'ViT-B-32')
    print(f"[OK] CLIP model: {clip_model}")

    # Test type conversions
    num_workers = config.get_int('num_workers', 4)
    print(f"[OK] Num workers: {num_workers}")

    enable_cache = config.get_bool('enable_embedding_cache', True)
    print(f"[OK] Enable cache: {enable_cache}")

    # Test nested config
    thumbnail_sizes = config.get('thumbnail_sizes', {})
    print(f"[OK] Thumbnail sizes: {list(thumbnail_sizes.keys())}")

    return True


def test_thumbnail_generator():
    """Test thumbnail generation (without actual images)."""
    print("\n" + "="*80)
    print("TEST: Thumbnail Generator")
    print("="*80)

    from sim_bench.image_processing import ThumbnailGenerator

    # Create generator
    generator = ThumbnailGenerator(cache_dir=".cache/test_thumbnails")
    print(f"[OK] Created generator: {generator}")

    # Check sizes
    assert 'tiny' in generator.sizes
    assert 'small' in generator.sizes
    assert 'medium' in generator.sizes
    assert 'large' in generator.sizes
    print(f"[OK] All expected sizes present: {list(generator.sizes.keys())}")

    # Check cache stats
    stats = generator.get_cache_stats()
    assert 'enabled' in stats
    assert 'cache_dir' in stats
    print(f"[OK] Cache stats: {stats}")

    return True


def test_clip_tagger():
    """Test CLIP tagger initialization (without actual inference)."""
    print("\n" + "="*80)
    print("TEST: CLIP Tagger")
    print("="*80)

    try:
        from sim_bench.photo_analysis import CLIPTagger

        print("Loading CLIP tagger...")
        tagger = CLIPTagger(device='cpu')
        print(f"[OK] Created tagger: {tagger}")

        # Check prompt loading
        summary = tagger.get_prompt_summary()
        print(f"[OK] Loaded {summary['total_prompts']} prompts")

        assert summary['total_prompts'] == 55, f"Expected 55 prompts, got {summary['total_prompts']}"
        print("[OK] Correct number of prompts")

        # Check categories
        expected_categories = ['scene_content', 'quality_technical', 'composition_aesthetic', 'human_focused']
        for category in expected_categories:
            assert category in summary['categories'], f"Missing category: {category}"
        print(f"[OK] All expected categories present: {list(summary['categories'].keys())}")

        # Check thresholds
        assert 'face_detection' in summary['routing_thresholds']
        print(f"[OK] Routing thresholds configured: {summary['routing_thresholds']}")

        return True

    except ImportError as e:
        print(f"[SKIP] PyTorch/OpenCLIP not available: {e}")
        return True  # Not a failure, just not installed


def test_factory_functions():
    """Test factory functions."""
    print("\n" + "="*80)
    print("TEST: Factory Functions")
    print("="*80)

    from sim_bench.image_processing.factory import create_image_processor

    # Test image processor factory
    generator = create_image_processor('thumbnail', cache_dir='.cache/test')
    print(f"[OK] Image processor factory: {type(generator).__name__}")

    try:
        from sim_bench.photo_analysis.factory import create_photo_analyzer

        # Test photo analyzer factory
        tagger = create_photo_analyzer('clip', device='cpu')
        print(f"[OK] Photo analyzer factory: {type(tagger).__name__}")

    except ImportError:
        print("[SKIP] Photo analyzer factory (PyTorch not available)")

    return True


def test_config_file_exists():
    """Test that config files exist."""
    print("\n" + "="*80)
    print("TEST: Configuration Files")
    print("="*80)

    project_root = Path(__file__).parent.parent

    # Check global config
    global_config = project_root / 'configs' / 'global_config.yaml'
    assert global_config.exists(), f"Global config not found: {global_config}"
    print(f"[OK] Global config exists: {global_config}")

    # Check prompts config
    prompts_config = project_root / 'configs' / 'photo_analysis_prompts.yaml'
    assert prompts_config.exists(), f"Prompts config not found: {prompts_config}"
    print(f"[OK] Prompts config exists: {prompts_config}")

    # Validate prompts config structure
    import yaml
    with open(prompts_config, 'r') as f:
        prompts = yaml.safe_load(f)

    assert 'scene_content' in prompts
    assert 'quality_technical' in prompts
    assert 'composition_aesthetic' in prompts
    assert 'human_focused' in prompts
    assert 'routing_thresholds' in prompts
    print("[OK] Prompts config has correct structure")

    # Count prompts
    total = (
        len(prompts.get('scene_content', [])) +
        len(prompts.get('quality_technical', [])) +
        len(prompts.get('composition_aesthetic', [])) +
        len(prompts.get('human_focused', []))
    )
    assert total == 55, f"Expected 55 prompts, got {total}"
    print(f"[OK] Prompts config has 55 prompts")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHOTO ANALYSIS MODULE TESTS")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Global Config", test_global_config),
        ("Config Files", test_config_file_exists),
        ("Thumbnail Generator", test_thumbnail_generator),
        ("CLIP Tagger", test_clip_tagger),
        ("Factory Functions", test_factory_functions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)

    return passed == total


if __name__ == "__main__":
    # Setup logging
    from sim_bench.config import setup_logging
    setup_logging()

    success = main()
    sys.exit(0 if success else 1)
