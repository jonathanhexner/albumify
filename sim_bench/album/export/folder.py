"""
Folder-based exporter.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from sim_bench.album.export.base import BaseExporter
from sim_bench.model_hub import ImageMetrics

logger = logging.getLogger(__name__)


class FolderExporter(BaseExporter):
    """
    Exports images to folder structure.
    
    Can organize by cluster or flat structure.
    Optionally includes thumbnails and metadata.
    """
    
    def export(
        self,
        selected_images: List[str],
        clusters: Dict[int, List[str]],
        output_path: Path,
        metrics: Optional[Dict[str, ImageMetrics]] = None
    ) -> Path:
        """
        Export images to folder structure.
        
        Creates:
        - output_path/cluster_N/ (if organize_by_cluster=True)
        - output_path/thumbnails/ (if include_thumbnails=True)
        - output_path/metadata.json (if metrics provided)
        
        Args:
            selected_images: List of selected image paths
            clusters: Cluster assignments
            output_path: Output directory
            metrics: Optional image metrics
        
        Returns:
            Path to output directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self._organize_by_cluster:
            self._export_by_cluster(selected_images, clusters, output_path)
        else:
            self._export_flat(selected_images, output_path)
        
        if self._include_thumbnails:
            self._create_thumbnails(selected_images, output_path)
        
        if metrics:
            self._export_metadata(selected_images, output_path, metrics)
        
        logger.info(f"Exported {len(selected_images)} images to {output_path}")
        return output_path
    
    def _export_by_cluster(
        self,
        selected_images: List[str],
        clusters: Dict[int, List[str]],
        output_path: Path
    ):
        """Export images organized by cluster."""
        for cluster_id, cluster_images in clusters.items():
            cluster_dir = output_path / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            for img_path in cluster_images:
                if img_path in selected_images:
                    src = Path(img_path)
                    dst = cluster_dir / src.name
                    shutil.copy2(src, dst)
    
    def _export_flat(self, selected_images: List[str], output_path: Path):
        """Export images to flat directory."""
        for img_path in selected_images:
            src = Path(img_path)
            dst = output_path / src.name
            shutil.copy2(src, dst)
    
    def _create_thumbnails(self, selected_images: List[str], output_path: Path):
        """Create thumbnails for selected images."""
        from PIL import Image, ImageOps

        thumb_dir = output_path / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)

        thumb_size = self._config.get('thumbnail_sizes', {}).get('small', 512)

        for img_path in selected_images:
            src = Path(img_path)
            dst = thumb_dir / src.name

            with Image.open(src) as img:
                # Apply EXIF orientation (fixes rotation issues)
                img = ImageOps.exif_transpose(img)
                img.thumbnail((thumb_size, thumb_size))
                img.save(dst, quality=85)
    
    def _export_metadata(
        self,
        selected_images: List[str],
        output_path: Path,
        metrics: Dict[str, ImageMetrics]
    ):
        """Export metadata JSON file."""
        import json
        
        metadata = {}
        for img_path in selected_images:
            metric = metrics.get(img_path)
            if not metric:
                continue
            
            metadata[Path(img_path).name] = {
                'iqa_score': metric.iqa_score,
                'ava_score': metric.ava_score,
                'sharpness': metric.sharpness,
                'is_portrait': metric.is_portrait,
                'eyes_open': metric.eyes_open,
                'is_smiling': metric.is_smiling,
                'cluster_id': metric.cluster_id,
            }
        
        meta_file = output_path / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
