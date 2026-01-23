"""Domain models for album organization.

This layer contains pure data structures with no business logic dependencies.
"""

from sim_bench.album.domain.models import WorkflowResult, ClusterInfo
from sim_bench.album.domain.types import WorkflowStage

__all__ = ['WorkflowResult', 'ClusterInfo', 'WorkflowStage']
