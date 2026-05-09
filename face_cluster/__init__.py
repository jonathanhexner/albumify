"""Face clustering library for small batches using mutual kNN graphs.

High-precision clustering pipeline with quality gating and interpretable debugging.
"""

from face_cluster.types import (
    FaceRecord, GraphResult, ClusterResult,
    GateResult, QualityVerdict, ClusterOrigin, ClusterMetadata,
)
from face_cluster.config import PipelineConfig
from face_cluster.embedding import InsightFaceEmbedder
from face_cluster.quality import QualityGater
from face_cluster.knn_graph import KNNGraphBuilder
from face_cluster.clustering import ConnectedComponentsClusterer
from face_cluster.exemplars import D10ExemplarSelector
from face_cluster.merge import ConservativeMerger, apply_manual_merges, propose_merge_candidates
from face_cluster.profile_store import ProfileStore
from face_cluster.attach import HoldoutAttacher
from face_cluster.analysis import ClusterSnapshot
from face_cluster.features import FeatureComputer, ClusterPairFeatures, MergeFeatureContext
from face_cluster.pipeline import FaceClusteringPipeline, PipelineResult, PipelineStageError
from face_cluster.analysis_views import MergeAnalysisView, MergeComparisonView
from face_cluster.manual_merge_snapshot import save_manual_merge_snapshot
from face_cluster.run_naming import RunDirSpec, allocate_run_dir
from face_cluster.config_diff import ConfigDelta, compute as compute_config_diff
from face_cluster.run_history import RunRow, HistoryFilters

__all__ = [
    'FaceRecord',
    'GraphResult',
    'ClusterResult',
    'GateResult',
    'QualityVerdict',
    'ClusterOrigin',
    'ClusterMetadata',
    'PipelineConfig',
    'InsightFaceEmbedder',
    'QualityGater',
    'KNNGraphBuilder',
    'ConnectedComponentsClusterer',
    'D10ExemplarSelector',
    'ConservativeMerger',
    'apply_manual_merges',
    'propose_merge_candidates',
    'ProfileStore',
    'HoldoutAttacher',
    'ClusterSnapshot',
    'FeatureComputer',
    'ClusterPairFeatures',
    'MergeFeatureContext',
    'FaceClusteringPipeline',
    'PipelineResult',
    'PipelineStageError',
    'MergeAnalysisView',
    'MergeComparisonView',
    'save_manual_merge_snapshot',
    'RunDirSpec',
    'allocate_run_dir',
    'ConfigDelta',
    'compute_config_diff',
    'RunRow',
    'HistoryFilters',
]
