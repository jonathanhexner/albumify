"""
Export functionality for album organization.
"""

from sim_bench.album.export.base import BaseExporter
from sim_bench.album.export.folder import FolderExporter
from sim_bench.album.export.zip import ZipExporter

__all__ = ['BaseExporter', 'FolderExporter', 'ZipExporter', 'create_exporter']


def create_exporter(config: dict) -> BaseExporter:
    """
    Create exporter based on configuration.
    
    Args:
        config: Full configuration dictionary
    
    Returns:
        Configured exporter instance
    """
    format_type = config.get('album', {}).get('export', {}).get('format', 'folder')
    
    exporters = {
        'folder': FolderExporter,
        'zip': ZipExporter,
    }
    
    exporter_class = exporters.get(format_type, FolderExporter)
    return exporter_class(config)
