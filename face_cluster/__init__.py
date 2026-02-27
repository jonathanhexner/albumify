"""Face clustering library for small batches using mutual kNN graphs.

High-precision clustering pipeline with quality gating and interpretable debugging.
"""

from face_cluster.types import FaceRecord, GraphResult, ClusterResult
from face_cluster.config import PipelineConfig
from face_cluster.embedding import InsightFaceEmbedder
from face_cluster.quality import QualityGater
from face_cluster.knn_graph import KNNGraphBuilder
from face_cluster.clustering import ConnectedComponentsClusterer
from face_cluster.exemplars import D10ExemplarSelector
from face_cluster.merge import ConservativeMerger
from face_cluster.attach import HoldoutAttacher
from face_cluster.analysis import ClusterSnapshot
from face_cluster.features import FeatureComputer, ClusterPairFeatures

__all__ = [
    'FaceRecord',
    'GraphResult',
    'ClusterResult',
    'PipelineConfig',
    'InsightFaceEmbedder',
    'QualityGater',
    'KNNGraphBuilder',
    'ConnectedComponentsClusterer',
    'D10ExemplarSelector',
    'ConservativeMerger',
    'HoldoutAttacher',
    'ClusterSnapshot',
    'FeatureComputer',
    'ClusterPairFeatures',
]
