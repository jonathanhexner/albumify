"""
Album Organization Module.

Provides workflow pipeline for organizing photo albums:
- Image discovery and quality filtering
- Portrait analysis and clustering
- Best image selection and export

Usage:
    from sim_bench.album import AlbumService
    
    service = AlbumService(config)
    result = service.organize_album(source_dir, output_dir)
"""

from sim_bench.album.domain import WorkflowResult, ClusterInfo, WorkflowStage
from sim_bench.album.services import AlbumService, SelectionService

__all__ = [
    'AlbumService',
    'SelectionService',
    'WorkflowResult',
    'ClusterInfo',
    'WorkflowStage',
]
