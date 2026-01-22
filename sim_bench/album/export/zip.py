"""
ZIP archive exporter.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

from sim_bench.album.export.base import BaseExporter
from sim_bench.album.export.folder import FolderExporter
from sim_bench.model_hub import ImageMetrics

logger = logging.getLogger(__name__)


class ZipExporter(BaseExporter):
    """
    Exports images as ZIP archive.
    
    Uses FolderExporter internally, then zips the result.
    """
    
    def export(
        self,
        selected_images: List[str],
        clusters: Dict[int, List[str]],
        output_path: Path,
        metrics: Optional[Dict[str, ImageMetrics]] = None
    ) -> Path:
        """
        Export images as ZIP archive.
        
        Args:
            selected_images: List of selected image paths
            clusters: Cluster assignments
            output_path: Output ZIP file path (or directory)
            metrics: Optional image metrics
        
        Returns:
            Path to created ZIP file
        """
        output_path = Path(output_path)
        
        if output_path.is_dir():
            zip_path = output_path / "album_export.zip"
        else:
            zip_path = output_path.with_suffix('.zip')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            
            folder_exporter = FolderExporter(self._config)
            folder_exporter.export(selected_images, clusters, temp_dir, metrics)
            
            self._create_zip(temp_dir, zip_path)
        
        logger.info(f"Exported {len(selected_images)} images to {zip_path}")
        return zip_path
    
    def _create_zip(self, source_dir: Path, zip_path: Path):
        """Create ZIP archive from directory."""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zf.write(file_path, arcname)
