"""Album organization service - main business logic.

Pure business logic - no UI dependencies.
Can be called from Streamlit, FastAPI, CLI, or tests.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from uuid import uuid4

from sim_bench.model_hub import ModelHub, ImageMetrics
from sim_bench.album import stages
from sim_bench.album.domain.models import WorkflowResult
from sim_bench.album.domain.types import WorkflowStage, ProgressCallback
from sim_bench.album.preprocessor import ImagePreprocessor
from sim_bench.album.services.selection_service import SelectionService
from sim_bench.album.telemetry import WorkflowTelemetry, TimingTracker

logger = logging.getLogger(__name__)


class AlbumService:
    """
    Album organization service.
    
    Orchestrates the complete album organization pipeline.
    This is the main entry point for business logic.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self._config = config
        self._album_config = config.get('album', {})
        self._quality_config = self._album_config.get('quality', {})
        self._portrait_config = self._album_config.get('portrait', {})
        self._clustering_config = self._album_config.get('clustering', {})
        self._export_config = self._album_config.get('export', {})

        self._hub = ModelHub(config)
        self._preprocessor = ImagePreprocessor(config)
        self._selector = SelectionService(config)

        logger.info("AlbumService initialized")

    def organize_album(
        self,
        source_directory: Path,
        output_directory: Optional[Path] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> WorkflowResult:
        """
        Execute album organization workflow.
        
        Args:
            source_directory: Directory containing images
            output_directory: Where to export results (optional)
            progress_callback: Optional progress callback
        
        Returns:
            WorkflowResult with all pipeline outputs
        """
        source_directory = Path(source_directory)
        if not source_directory.exists():
            raise FileNotFoundError(f"Source directory not found: {source_directory}")

        run_id = uuid4().hex[:8]
        telemetry = WorkflowTelemetry(run_id=run_id)
        start_time = time.time()

        self._notify(progress_callback, WorkflowStage.DISCOVER, 0.0)

        # Stage 1: Discover
        with TimingTracker(telemetry, "discover_images"):
            images = stages.discover_images(source_directory)

        if not images:
            logger.warning("No images found")
            return self._empty_result(source_directory)

        # Stage 2: Preprocess
        self._notify(progress_callback, WorkflowStage.PREPROCESS, 0.05)
        preprocessing_enabled = self._album_config.get('preprocessing', {}).get('enabled', True)
        thumbnails = None

        if preprocessing_enabled:
            with TimingTracker(telemetry, "preprocess_thumbnails", len(images)):
                thumbnails = self._preprocessor.preprocess_batch(images)

        # Stage 3: Analyze
        self._notify(progress_callback, WorkflowStage.ANALYZE, 0.15)
        with TimingTracker(telemetry, "analyze_images", len(images)):
            metrics = self._analyze_images(images, thumbnails, progress_callback)

        # Stage 4: Filter quality
        self._notify(progress_callback, WorkflowStage.FILTER_QUALITY, 0.6)
        with TimingTracker(telemetry, "filter_quality"):
            quality_passed = stages.filter_by_quality(metrics, self._quality_config)

        # Stage 5: Filter portraits
        self._notify(progress_callback, WorkflowStage.FILTER_PORTRAIT, 0.65)
        with TimingTracker(telemetry, "filter_portrait"):
            filtered_metrics = {k: v for k, v in metrics.items() if k in quality_passed}
            all_passed = stages.filter_by_portrait(filtered_metrics, self._portrait_config)

        if not all_passed:
            logger.warning("No images passed filters")
            return self._empty_result(source_directory)

        # Stage 6: Extract features
        self._notify(progress_callback, WorkflowStage.EXTRACT_FEATURES, 0.7)
        passed_paths = [Path(p) for p in all_passed]
        with TimingTracker(telemetry, "extract_features", len(passed_paths)):
            features = self._hub.extract_features(passed_paths)

        # Stage 7: Cluster
        self._notify(progress_callback, WorkflowStage.CLUSTER, 0.85)
        with TimingTracker(telemetry, "cluster_images", len(passed_paths)):
            labels, _ = self._hub.cluster_images(features)

        for path, label in zip(all_passed, labels):
            metrics[path].cluster_id = int(label)

        clusters = stages.organize_clusters(all_passed, labels)
        cluster_stats = stages.compute_cluster_stats(clusters)

        # Stage 8: Select best
        self._notify(progress_callback, WorkflowStage.SELECT, 0.92)
        with TimingTracker(telemetry, "select_best", len(clusters)):
            selected = self._selector.select_best(clusters, metrics, self._hub)

        # Stage 9: Export
        self._notify(progress_callback, WorkflowStage.EXPORT, 0.96)
        export_path = None
        if output_directory:
            with TimingTracker(telemetry, "export_results", len(selected)):
                export_path = self._export_results(selected, clusters, output_directory, metrics)

        # Finalize telemetry
        telemetry.total_duration_sec = time.time() - start_time
        telemetry.metadata = {
            'total_images': len(images),
            'filtered_images': len(all_passed),
            'num_clusters': len(clusters),
            'selected_images': len(selected),
            'preprocessing_enabled': preprocessing_enabled
        }

        if output_directory:
            telemetry.export_json(output_directory / f"telemetry_{run_id}.json")

        self._notify(progress_callback, WorkflowStage.COMPLETE, 1.0)

        all_image_paths = [str(p) for p in images]
        filtered_out = [p for p in all_image_paths if p not in all_passed]

        return WorkflowResult(
            source_directory=source_directory,
            total_images=len(images),
            filtered_images=len(all_passed),
            clusters=clusters,
            selected_images=selected,
            metrics=metrics,
            all_images=all_image_paths,
            filtered_out=filtered_out,
            cluster_stats=cluster_stats,
            export_path=export_path,
            run_id=run_id,
            telemetry=telemetry
        )

    def get_config(self) -> Dict[str, Any]:
        """Return current album configuration."""
        return self._album_config.copy()

    def _analyze_images(
        self,
        images: List[Path],
        thumbnails: Optional[Dict] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Dict[str, ImageMetrics]:
        """Analyze quality and portrait metrics for all images."""
        def inner_progress(operation, current, total, image_name):
            if progress_callback:
                pct = 0.15 + (current / total) * 0.45
                progress_callback(WorkflowStage.ANALYZE, pct, image_name)

        return self._hub.analyze_batch(
            images,
            thumbnails=thumbnails,
            include_quality=True,
            include_portrait=True,
            progress_callback=inner_progress
        )

    def _export_results(
        self,
        selected: List[str],
        clusters: Dict[int, List[str]],
        output_directory: Path,
        metrics: Dict[str, ImageMetrics]
    ) -> Path:
        """Export selected images to output directory."""
        import shutil

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        organize_by_cluster = self._export_config.get('organize_by_cluster', True)

        if organize_by_cluster:
            for cluster_id, cluster_images in clusters.items():
                cluster_dir = output_directory / f"cluster_{cluster_id}"
                cluster_dir.mkdir(exist_ok=True)
                for img_path in cluster_images:
                    if img_path in selected:
                        shutil.copy2(img_path, cluster_dir / Path(img_path).name)
        else:
            for img_path in selected:
                shutil.copy2(img_path, output_directory / Path(img_path).name)

        logger.info(f"Exported {len(selected)} images to {output_directory}")
        return output_directory

    def _notify(
        self,
        callback: Optional[ProgressCallback],
        stage: WorkflowStage,
        progress: float,
        detail: str = None
    ):
        """Helper to call progress callback."""
        if callback:
            callback(stage, progress, detail)

    def _empty_result(self, source_directory: Path) -> WorkflowResult:
        """Create empty result for failed workflow."""
        return WorkflowResult(
            source_directory=source_directory,
            total_images=0,
            filtered_images=0,
            clusters={},
            selected_images=[],
            metrics={}
        )
