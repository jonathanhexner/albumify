"""
Album organization workflow pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from uuid import uuid4

from sim_bench.model_hub import ModelHub, ImageMetrics
from sim_bench.album import stages
from sim_bench.album.preprocessor import ImagePreprocessor
from sim_bench.album.telemetry import WorkflowTelemetry, TimingTracker

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """
    Results from album workflow execution.
    """
    source_directory: Path
    total_images: int
    filtered_images: int
    clusters: Dict[int, List[str]]
    selected_images: List[str]
    metrics: Dict[str, ImageMetrics]
    cluster_stats: Dict[str, Any] = field(default_factory=dict)
    export_path: Optional[Path] = None
    run_id: Optional[str] = None
    telemetry: Optional[WorkflowTelemetry] = None


class AlbumWorkflow:
    """
    Album organization workflow pipeline.
    
    Config-only constructor - reads settings from config['album'].
    
    Pipeline stages:
    1. discover_images - Find all images in source directory
    2. analyze_quality - Run quality and portrait analysis
    3. filter_quality - Apply quality thresholds
    4. filter_portrait - Apply portrait requirements
    5. extract_features - Extract embeddings for clustering
    6. cluster_images - Group similar images
    7. select_best - Choose best images per cluster
    8. export_results - Export selected images
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize workflow with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self._config = config
        self._album_config = config.get('album', {})
        self._quality_config = self._album_config.get('quality', {})
        self._portrait_config = self._album_config.get('portrait', {})
        self._clustering_config = self._album_config.get('clustering', {})
        self._selection_config = self._album_config.get('selection', {})
        self._export_config = self._album_config.get('export', {})
        
        self._hub = ModelHub(config)
        self._preprocessor = ImagePreprocessor(config)
        
        logger.info("AlbumWorkflow initialized")
    
    def run(
        self,
        source_directory: Path,
        output_directory: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> WorkflowResult:
        """
        Execute complete workflow pipeline.
        
        Args:
            source_directory: Directory containing images
            output_directory: Where to export results
            progress_callback: Optional callback(stage_name, progress)
        
        Returns:
            WorkflowResult with all pipeline outputs
        """
        source_directory = Path(source_directory)
        
        if not source_directory.exists():
            raise FileNotFoundError(f"Source directory not found: {source_directory}")
        
        # Initialize telemetry
        run_id = uuid4().hex[:8]
        telemetry = WorkflowTelemetry(run_id=run_id)
        start_time = time.time()
        
        self._progress(progress_callback, "discover_images", 0.0)
        
        # Stage 1: Discover images
        with TimingTracker(telemetry, "discover_images"):
            images = stages.discover_images(source_directory)
        
        if not images:
            logger.warning("No images found in source directory")
            return self._empty_result(source_directory)
        
        self._progress(progress_callback, "preprocess", 0.05)
        
        # Stage 2: Preprocess (generate thumbnails)
        preprocessing_enabled = self._album_config.get('preprocessing', {}).get('enabled', True)
        thumbnails = None
        
        if preprocessing_enabled:
            with TimingTracker(telemetry, "preprocess_thumbnails", len(images)):
                thumbnails = self._preprocessor.preprocess_batch(images, progress_callback)
        
        self._progress(progress_callback, "analyze_quality", 0.15)
        
        # Stage 3: Analyze all images
        with TimingTracker(telemetry, "analyze_images", len(images)):
            metrics = self._analyze_images(images, thumbnails, progress_callback)
        
        self._progress(progress_callback, "filter_quality", 0.6)
        
        # Stage 4: Filter by quality
        with TimingTracker(telemetry, "filter_quality"):
            quality_passed = stages.filter_by_quality(metrics, self._quality_config)
        
        self._progress(progress_callback, "filter_portrait", 0.65)
        
        # Stage 5: Filter by portrait requirements
        with TimingTracker(telemetry, "filter_portrait"):
            all_passed = stages.filter_by_portrait(
                {k: v for k, v in metrics.items() if k in quality_passed},
                self._portrait_config
            )
        
        if not all_passed:
            logger.warning("No images passed quality and portrait filters")
            return self._empty_result(source_directory)
        
        self._progress(progress_callback, "extract_features", 0.7)
        
        # Stage 6: Extract features
        passed_paths = [Path(p) for p in all_passed]
        with TimingTracker(telemetry, "extract_features", len(passed_paths)):
            features = self._hub.extract_features(passed_paths)
        
        self._progress(progress_callback, "cluster_images", 0.85)
        
        # Stage 7: Cluster images
        with TimingTracker(telemetry, "cluster_images", len(passed_paths)):
            labels, cluster_info = self._hub.cluster_images(features)
        
        # Update metrics with cluster assignments
        for path, label in zip(all_passed, labels):
            metrics[path].cluster_id = int(label)
        
        clusters = stages.organize_clusters(all_passed, labels)
        cluster_stats = stages.compute_cluster_stats(clusters)
        
        self._progress(progress_callback, "select_best", 0.92)
        
        # Stage 8: Select best images
        with TimingTracker(telemetry, "select_best", len(clusters)):
            selected = self._select_best_images(clusters, metrics)
        
        self._progress(progress_callback, "export_results", 0.96)
        
        # Stage 9: Export (if output directory provided)
        export_path = None
        if output_directory:
            with TimingTracker(telemetry, "export_results", len(selected)):
                export_path = self._export_results(
                    selected,
                    clusters,
                    output_directory,
                    metrics
                )
        
        # Finalize telemetry
        telemetry.total_duration_sec = time.time() - start_time
        telemetry.metadata = {
            'total_images': len(images),
            'filtered_images': len(all_passed),
            'num_clusters': len(clusters),
            'selected_images': len(selected),
            'preprocessing_enabled': preprocessing_enabled
        }
        
        # Export telemetry
        if output_directory:
            telemetry.export_json(output_directory / f"telemetry_{run_id}.json")
        
        self._progress(progress_callback, "complete", 1.0)
        
        return WorkflowResult(
            source_directory=source_directory,
            total_images=len(images),
            filtered_images=len(all_passed),
            clusters=clusters,
            selected_images=selected,
            metrics=metrics,
            cluster_stats=cluster_stats,
            export_path=export_path,
            run_id=run_id,
            telemetry=telemetry
        )
    
    def _analyze_images(
        self,
        images: List[Path],
        thumbnails: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, ImageMetrics]:
        """
        Analyze quality and portrait metrics for all images.
        """
        def progress(operation, current, total, image_name):
            if progress_callback:
                # Map to enhanced progress callback with operation details
                pct = 0.15 + (current / total) * 0.45
                progress_callback("analyze_quality", pct, operation, image_name)
        
        return self._hub.analyze_batch(
            images,
            thumbnails=thumbnails,
            include_quality=True,
            include_portrait=True,
            progress_callback=progress
        )
    
    def _select_best_images(
        self,
        clusters: Dict[int, List[str]],
        metrics: Dict[str, ImageMetrics]
    ) -> List[str]:
        """
        Select best images from each cluster.
        """
        images_per_cluster = self._selection_config.get('images_per_cluster', 1)
        ava_weight = self._selection_config.get('ava_weight', 0.5)
        iqa_weight = self._selection_config.get('iqa_weight', 0.2)
        portrait_weight = self._selection_config.get('portrait_weight', 0.3)
        
        selected = []
        for cluster_id, cluster_images in clusters.items():
            scored = []
            for img_path in cluster_images:
                metric = metrics[img_path]
                score = metric.get_composite_score(ava_weight, iqa_weight, portrait_weight)
                scored.append((img_path, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            selected.extend([path for path, _ in scored[:images_per_cluster]])
        
        logger.info(f"Selected {len(selected)} best images from {len(clusters)} clusters")
        return selected
    
    def _export_results(
        self,
        selected_images: List[str],
        clusters: Dict[int, List[str]],
        output_directory: Path,
        metrics: Dict[str, ImageMetrics]
    ) -> Path:
        """
        Export selected images to output directory.
        """
        import shutil
        
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        organize_by_cluster = self._export_config.get('organize_by_cluster', True)
        
        if organize_by_cluster:
            for cluster_id, cluster_images in clusters.items():
                cluster_dir = output_directory / f"cluster_{cluster_id}"
                cluster_dir.mkdir(exist_ok=True)
                
                for img_path in cluster_images:
                    if img_path in selected_images:
                        dest = cluster_dir / Path(img_path).name
                        shutil.copy2(img_path, dest)
        else:
            for img_path in selected_images:
                dest = output_directory / Path(img_path).name
                shutil.copy2(img_path, dest)
        
        logger.info(f"Exported {len(selected_images)} images to {output_directory}")
        return output_directory
    
    def _progress(
        self,
        callback: Optional[Callable],
        stage: str,
        progress: float
    ):
        """Helper to call progress callback."""
        if callback:
            callback(stage, progress)
    
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


def create_album_workflow(
    source_directory: Path,
    album_name: str,
    overrides: Optional[Dict[str, Any]] = None
) -> AlbumWorkflow:
    """
    Factory function to create workflow with album-specific overrides.
    
    Args:
        source_directory: Directory containing images
        album_name: Name of the album
        overrides: Optional configuration overrides
    
    Returns:
        Configured AlbumWorkflow instance
    """
    from sim_bench.config import get_global_config
    
    config = get_global_config().to_dict()
    
    config.setdefault('album', {})
    config['album']['source_directory'] = str(source_directory)
    config['album']['name'] = album_name
    
    if overrides:
        config = _deep_merge(config, overrides)
    
    return AlbumWorkflow(config)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result
