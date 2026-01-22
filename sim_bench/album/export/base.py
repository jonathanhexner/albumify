"""
Base exporter interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

from sim_bench.model_hub import ImageMetrics


class BaseExporter(ABC):
    """
    Abstract base class for album exporters.
    
    Config-only constructor - reads from config['album']['export'].
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exporter with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self._config = config
        exp = config.get('album', {}).get('export', {})
        
        self._organize_by_cluster = exp.get('organize_by_cluster', True)
        self._include_thumbnails = exp.get('include_thumbnails', True)
    
    @abstractmethod
    def export(
        self,
        selected_images: List[str],
        clusters: Dict[int, List[str]],
        output_path: Path,
        metrics: Optional[Dict[str, ImageMetrics]] = None
    ) -> Path:
        """
        Export selected images to output location.
        
        Args:
            selected_images: List of selected image paths
            clusters: Cluster assignments
            output_path: Where to export
            metrics: Optional image metrics for metadata
        
        Returns:
            Path to exported output
        """
        pass
