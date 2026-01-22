"""
Tests for Selection and Export modules.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that selection and export modules import correctly."""
    print("\n" + "="*80)
    print("TEST: Selection/Export Imports")
    print("="*80)

    from sim_bench.album.selection import BestImageSelector
    from sim_bench.album.export import (
        BaseExporter,
        FolderExporter,
        ZipExporter,
        create_exporter
    )
    print("[OK] Selection and export modules imported")
    return True


def test_selector_initialization():
    """Test BestImageSelector initialization."""
    print("\n" + "="*80)
    print("TEST: Selector Initialization")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.album.selection import BestImageSelector

    config = get_global_config().to_dict()
    selector = BestImageSelector(config)
    
    print(f"[OK] Selector initialized")
    print(f"     Images per cluster: {selector._images_per_cluster}")
    print(f"     AVA weight: {selector._ava_weight}")
    print(f"     IQA weight: {selector._iqa_weight}")
    
    return True


def test_selector_compute_score():
    """Test score computation."""
    print("\n" + "="*80)
    print("TEST: Selector Compute Score")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.album.selection import BestImageSelector
    from sim_bench.model_hub import ImageMetrics

    config = get_global_config().to_dict()
    selector = BestImageSelector(config)
    
    metrics = ImageMetrics(
        image_path="test.jpg",
        iqa_score=0.8,
        ava_score=7.5,
        is_portrait=True,
        eyes_open=True,
        is_smiling=True,
        smile_score=0.9
    )
    
    score = selector.compute_score(metrics)
    print(f"[OK] Computed score: {score:.3f}")
    
    return score > 0 and score <= 1


def test_selector_best_selection():
    """Test best image selection from clusters."""
    print("\n" + "="*80)
    print("TEST: Selector Best Selection")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.album.selection import BestImageSelector
    from sim_bench.model_hub import ImageMetrics

    config = get_global_config().to_dict()
    selector = BestImageSelector(config)
    
    clusters = {
        0: ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        1: ['img4.jpg', 'img5.jpg'],
    }
    
    metrics = {
        'img1.jpg': ImageMetrics(image_path='img1.jpg', iqa_score=0.9, ava_score=8.0),
        'img2.jpg': ImageMetrics(image_path='img2.jpg', iqa_score=0.5, ava_score=5.0),
        'img3.jpg': ImageMetrics(image_path='img3.jpg', iqa_score=0.7, ava_score=6.5),
        'img4.jpg': ImageMetrics(image_path='img4.jpg', iqa_score=0.6, ava_score=6.0),
        'img5.jpg': ImageMetrics(image_path='img5.jpg', iqa_score=0.8, ava_score=7.0),
    }
    
    selected = selector.select_from_clusters(clusters, metrics)
    
    print(f"[OK] Selected {len(selected)} images from {len(clusters)} clusters")
    print(f"     Selected: {selected}")
    
    return len(selected) == 2 and 'img1.jpg' in selected and 'img5.jpg' in selected


def test_exporter_factory():
    """Test exporter factory function."""
    print("\n" + "="*80)
    print("TEST: Exporter Factory")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.album.export import create_exporter, FolderExporter, ZipExporter

    config = get_global_config().to_dict()
    
    config['album']['export']['format'] = 'folder'
    exporter = create_exporter(config)
    print(f"[OK] Created folder exporter: {type(exporter).__name__}")
    
    folder_ok = isinstance(exporter, FolderExporter)
    
    config['album']['export']['format'] = 'zip'
    exporter = create_exporter(config)
    print(f"[OK] Created zip exporter: {type(exporter).__name__}")
    
    zip_ok = isinstance(exporter, ZipExporter)
    
    return folder_ok and zip_ok


def test_folder_export():
    """Test folder exporter."""
    print("\n" + "="*80)
    print("TEST: Folder Export")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.album.export import FolderExporter
    from sim_bench.model_hub import ImageMetrics
    import shutil

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True
    
    image_files = list(test_images.glob("*.jpg"))[:3]
    if len(image_files) < 2:
        print("[SKIP] Need at least 2 test images")
        return True
    
    config = get_global_config().to_dict()
    config['album']['export']['organize_by_cluster'] = True
    config['album']['export']['include_thumbnails'] = False
    
    exporter = FolderExporter(config)
    
    selected = [str(image_files[0]), str(image_files[1])]
    clusters = {
        0: [str(image_files[0])],
        1: [str(image_files[1])],
    }
    
    metrics = {
        str(image_files[0]): ImageMetrics(
            image_path=str(image_files[0]),
            iqa_score=0.8,
            cluster_id=0
        ),
        str(image_files[1]): ImageMetrics(
            image_path=str(image_files[1]),
            iqa_score=0.7,
            cluster_id=1
        ),
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output = exporter.export(selected, clusters, Path(tmpdir), metrics)
        
        cluster_0 = output / "cluster_0"
        cluster_1 = output / "cluster_1"
        
        print(f"[OK] Exported to {output}")
        print(f"     Cluster 0 exists: {cluster_0.exists()}")
        print(f"     Cluster 1 exists: {cluster_1.exists()}")
        
        return cluster_0.exists() and cluster_1.exists()


def test_zip_export():
    """Test ZIP exporter."""
    print("\n" + "="*80)
    print("TEST: ZIP Export")
    print("="*80)

    from sim_bench.config import get_global_config
    from sim_bench.album.export import ZipExporter
    from sim_bench.model_hub import ImageMetrics

    test_images = Path(__file__).parent.parent / "test_images"
    if not test_images.exists():
        print("[SKIP] No test_images directory found")
        return True
    
    image_files = list(test_images.glob("*.jpg"))[:2]
    if len(image_files) < 2:
        print("[SKIP] Need at least 2 test images")
        return True
    
    config = get_global_config().to_dict()
    config['album']['export']['organize_by_cluster'] = False
    config['album']['export']['include_thumbnails'] = False
    
    exporter = ZipExporter(config)
    
    selected = [str(image_files[0]), str(image_files[1])]
    clusters = {0: selected}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output = exporter.export(selected, clusters, Path(tmpdir))
        
        print(f"[OK] Created ZIP: {output}")
        print(f"     ZIP exists: {output.exists()}")
        print(f"     ZIP size: {output.stat().st_size} bytes")
        
        return output.exists() and output.suffix == '.zip'


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("SELECTION & EXPORT TEST SUITE")
    print("="*80)

    tests = [
        test_imports,
        test_selector_initialization,
        test_selector_compute_score,
        test_selector_best_selection,
        test_exporter_factory,
        test_folder_export,
        test_zip_export,
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
