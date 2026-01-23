"""
Smoke test for Phase 1.5 refactoring - verify nothing broke.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_backend_imports():
    """Test backend module imports."""
    print("Testing backend imports...")
    from sim_bench.album import AlbumService, SelectionService, WorkflowResult, WorkflowStage, ClusterInfo
    from sim_bench.album.domain import WorkflowResult as DomainResult
    from sim_bench.album.services import AlbumService as Service
    print("  [OK] Backend imports OK")
    return True


def test_ui_imports():
    """Test UI module imports."""
    print("Testing UI imports...")
    from app.album.session import AlbumSession
    from app.album.components import (
        render_config_panel,
        render_workflow_form,
        render_workflow_runner,
        render_results,
        render_gallery,
        render_metrics
    )
    print("  [OK] UI imports OK")
    return True


def test_service_instantiation():
    """Test service can be created."""
    print("Testing service instantiation...")
    from sim_bench.album import AlbumService
    from sim_bench.config import get_global_config

    config = get_global_config().to_dict()
    service = AlbumService(config)
    assert service is not None
    print("  [OK] Service instantiation OK")
    return True


def test_domain_models():
    """Test domain models work."""
    print("Testing domain models...")
    from sim_bench.album.domain import WorkflowResult, ClusterInfo, WorkflowStage
    from pathlib import Path

    result = WorkflowResult(
        source_directory=Path("/test"),
        total_images=10,
        filtered_images=8,
        clusters={0: ["img1.jpg", "img2.jpg"]},
        selected_images=["img1.jpg"],
        metrics={}
    )
    assert result.total_images == 10
    assert len(result.clusters) == 1

    cluster_info = result.get_cluster_info(0)
    assert cluster_info is not None
    assert cluster_info.size == 2
    print("  [OK] Domain models OK")
    return True


def test_no_legacy_imports():
    """Verify legacy imports are gone."""
    print("Testing legacy code removal...")
    import importlib

    # These should fail
    legacy_modules = [
        'sim_bench.album.workflow',
        'sim_bench.album.selection',
        'app.album.config_panel',
        'app.album.workflow_runner',
        'app.album.results_viewer'
    ]

    for module_name in legacy_modules:
        try:
            importlib.import_module(module_name)
            print(f"  [FAIL] {module_name} still exists!")
            return False
        except ModuleNotFoundError:
            pass  # Expected - module should not exist

    print("  [OK] Legacy modules removed")
    return True


def run_smoke_tests():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("PHASE 1.5 SMOKE TESTS")
    print("="*60 + "\n")

    tests = [
        test_backend_imports,
        test_ui_imports,
        test_service_instantiation,
        test_domain_models,
        test_no_legacy_imports,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [FAIL] {test.__name__} - {e}")
            results.append(False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All smoke tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
