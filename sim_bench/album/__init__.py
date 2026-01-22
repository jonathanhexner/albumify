"""
Album Organization Module.

Provides workflow pipeline for organizing photo albums:
- Image discovery and quality filtering
- Portrait analysis and clustering
- Best image selection and export
"""

from sim_bench.album.workflow import AlbumWorkflow, WorkflowResult, create_album_workflow

__all__ = [
    'AlbumWorkflow',
    'WorkflowResult',
    'create_album_workflow',
]
