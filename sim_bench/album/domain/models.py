"""Core domain models for album organization.

These are pure data structures - no business logic, no external dependencies
except for ImageMetrics and WorkflowTelemetry which are also data structures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

from sim_bench.model_hub import ImageMetrics
from sim_bench.album.telemetry import WorkflowTelemetry


@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    cluster_id: int
    image_paths: List[str]
    best_image: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.image_paths)


@dataclass
class WorkflowResult:
    """
    Immutable result from album workflow execution.
    
    This is the primary output of the album organization pipeline.
    """
    source_directory: Path
    total_images: int
    filtered_images: int
    clusters: Dict[int, List[str]]
    selected_images: List[str]
    metrics: Dict[str, ImageMetrics]
    all_images: List[str] = field(default_factory=list)
    filtered_out: List[str] = field(default_factory=list)
    cluster_stats: Dict[str, Any] = field(default_factory=dict)
    export_path: Optional[Path] = None
    run_id: Optional[str] = None
    telemetry: Optional[WorkflowTelemetry] = None

    def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Get info for a specific cluster."""
        if cluster_id not in self.clusters:
            return None
        images = self.clusters[cluster_id]
        best = next((img for img in self.selected_images if img in images), None)
        return ClusterInfo(cluster_id=cluster_id, image_paths=images, best_image=best)

    def get_all_cluster_info(self) -> List[ClusterInfo]:
        """Get ClusterInfo for all clusters."""
        return [self.get_cluster_info(cid) for cid in sorted(self.clusters.keys())]
