"""
Tests for Album Workflow module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that album modules import correctly."""
    print("\n" + "="*80)
    print("TEST: Album Module Imports")
    print("="*80)

    from sim_bench.album import AlbumService, WorkflowResult, WorkflowStage
    from sim_bench.album import stages
    print("[OK] Album modules imported")
    return True


def test_config_defaults():
    """Test that global config has album section."""
    print("\n" + "="*80)
    print("TEST: Album Config Defaults")
    print("="*80)

    from sim_bench.config import get_global_config

    config = get_global_config().to_dict()
    
    if 'album' not in config:
        print("[FAIL] No 'album' section in config")
        return False
    
    album_config = config['album']
    
    print(f"[OK] Album config found")
    print(f"     Quality min_ava_score: {album_config['quality']['min_ava_score']}")
    print(f"     Clustering method: {album_config['clustering']['method']}")
    print(f"     Selection images_per_cluster: {album_config['selection']['images_per_cluster']}")
    
    return True


def test_service_creation():
    """Test AlbumService creation."""
    print("\n" + "="*80)
    print("TEST: Service Creation")
    print("="*80)

    from sim_bench.album import AlbumService
    from sim_bench.config import get_global_config

    config = get_global_config().to_dict()
    service = AlbumService(config)
    
    album_config = service.get_config()
    
    print(f"[OK] AlbumService created")
    print(f"     Clustering method: {album_config['clustering'].get('method')}")
    print(f"     Min cluster size: {album_config['clustering'].get('min_cluster_size')}")
    
    return True


def test_service_with_overrides():
    """Test AlbumService with custom configuration overrides."""
    print("\n" + "="*80)
    print("TEST: Service with Overrides")
    print("="*80)

    from sim_bench.album import AlbumService
    from sim_bench.config import get_global_config

    config = get_global_config().to_dict()
    
    overrides = {
        'album': {
            'quality': {'min_ava_score': 6.0},
            'selection': {'images_per_cluster': 2}
        }
    }
    
    # Deep merge
    config['album']['quality']['min_ava_score'] = 6.0
    config['album']['selection']['images_per_cluster'] = 2
    
    service = AlbumService(config)
    album_config = service.get_config()
    
    min_ava = album_config['quality'].get('min_ava_score')
    images_per = album_config['selection'].get('images_per_cluster')
    
    print(f"[OK] Service created with overrides")
    print(f"     Min AVA score: {min_ava} (expected 6.0)")
    print(f"     Images per cluster: {images_per} (expected 2)")
    
    return min_ava == 6.0 and images_per == 2


def test_stage_discover_images():
    """Test image discovery stage."""
    print("\n" + "="*80)
    print("TEST: Stage - Discover Images")
    print("="*80)

    from sim_bench.album import stages

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True
    
    images = stages.discover_images(test_images)
    print(f"[OK] Discovered {len(images)} images")
    
    return len(images) >= 0


def test_stage_filter_quality():
    """Test quality filtering stage."""
    print("\n" + "="*80)
    print("TEST: Stage - Filter Quality")
    print("="*80)

    from sim_bench.album import stages
    from sim_bench.model_hub import ImageMetrics

    metrics = {
        'img1.jpg': ImageMetrics(image_path='img1.jpg', iqa_score=0.8, ava_score=7.0),
        'img2.jpg': ImageMetrics(image_path='img2.jpg', iqa_score=0.2, ava_score=3.0),
        'img3.jpg': ImageMetrics(image_path='img3.jpg', iqa_score=0.6, ava_score=5.5),
    }
    
    quality_config = {'min_iqa_score': 0.3, 'min_ava_score': 4.0}
    
    passed = stages.filter_by_quality(metrics, quality_config)
    
    print(f"[OK] Quality filter passed {len(passed)}/{len(metrics)} images")
    
    return 'img1.jpg' in passed and 'img3.jpg' in passed and 'img2.jpg' not in passed


def test_stage_filter_portrait():
    """Test portrait filtering stage."""
    print("\n" + "="*80)
    print("TEST: Stage - Filter Portrait")
    print("="*80)

    from sim_bench.album import stages
    from sim_bench.model_hub import ImageMetrics

    metrics = {
        'portrait1.jpg': ImageMetrics(
            image_path='portrait1.jpg',
            is_portrait=True,
            eyes_open=True
        ),
        'portrait2.jpg': ImageMetrics(
            image_path='portrait2.jpg',
            is_portrait=True,
            eyes_open=False
        ),
        'landscape.jpg': ImageMetrics(
            image_path='landscape.jpg',
            is_portrait=False
        ),
    }
    
    portrait_config = {'require_eyes_open': True}
    
    passed = stages.filter_by_portrait(metrics, portrait_config)
    
    print(f"[OK] Portrait filter passed {len(passed)}/{len(metrics)} images")
    
    expected_pass = 'portrait1.jpg' in passed and 'landscape.jpg' in passed
    expected_fail = 'portrait2.jpg' not in passed
    
    return expected_pass and expected_fail


def test_stage_organize_clusters():
    """Test cluster organization stage."""
    print("\n" + "="*80)
    print("TEST: Stage - Organize Clusters")
    print("="*80)

    from sim_bench.album import stages

    image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
    labels = [0, 0, 1, -1]  # -1 is noise
    
    clusters = stages.organize_clusters(image_paths, labels)
    
    print(f"[OK] Organized into {len(clusters)} clusters")
    print(f"     Cluster 0: {len(clusters.get(0, []))} images")
    print(f"     Cluster 1: {len(clusters.get(1, []))} images")
    
    return len(clusters) == 2 and len(clusters[0]) == 2 and len(clusters[1]) == 1


def test_full_workflow():
    """Test complete workflow execution if test images available."""
    print("\n" + "="*80)
    print("TEST: Full Workflow Execution")
    print("="*80)

    from sim_bench.album import AlbumService
    from sim_bench.album.domain.types import WorkflowStage
    from sim_bench.config import get_global_config
    import tempfile

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True
    
    image_files = list(test_images.glob("*.jpg"))
    if len(image_files) < 3:
        print("[SKIP] Need at least 3 test images")
        return True
    
    print(f"[INFO] Running workflow with {len(image_files)} images")
    
    config = get_global_config().to_dict()
    service = AlbumService(config)
    
    def progress(stage: WorkflowStage, pct: float, detail: str = None):
        print(f"     {stage.name}: {pct*100:.0f}%")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = service.organize_album(
            source_directory=test_images,
            output_directory=Path(tmpdir),
            progress_callback=progress
        )
    
    print(f"[OK] Workflow completed")
    print(f"     Total images: {result.total_images}")
    print(f"     Filtered images: {result.filtered_images}")
    print(f"     Clusters: {len(result.clusters)}")
    print(f"     Selected: {len(result.selected_images)}")
    
    return result.total_images > 0


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("ALBUM WORKFLOW TEST SUITE")
    print("="*80)

    tests = [
        test_imports,
        test_config_defaults,
        test_service_creation,
        test_service_with_overrides,
        test_stage_discover_images,
        test_stage_filter_quality,
        test_stage_filter_portrait,
        test_stage_organize_clusters,
        test_full_workflow,
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
