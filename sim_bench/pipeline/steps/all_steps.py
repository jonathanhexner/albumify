"""Import all steps to register them with the global registry."""

# Discovery
from sim_bench.pipeline.steps.discover_images import DiscoverImagesStep

# Geo-temporal (spec-022)
from sim_bench.pipeline.steps.extract_geo_metadata import ExtractGeoMetadataStep
from sim_bench.pipeline.steps.geo_temporal_segment import GeoTemporalSegmentStep
from sim_bench.pipeline.steps.infer_geo_clip import InferGeoClipStep
from sim_bench.pipeline.steps.infer_geo_coords import InferGeoCoordsStep
from sim_bench.pipeline.steps.caption_images import CaptionImagesStep

# Scoring
from sim_bench.pipeline.steps.score_iqa import ScoreIQAStep
from sim_bench.pipeline.steps.score_ava import ScoreAVAStep
from sim_bench.pipeline.steps.score_face_quality import ScoreFaceQualityStep
from sim_bench.pipeline.steps.score_quality import ScoreQualityStep  # spec-093
from sim_bench.pipeline.steps.classify_scene import ClassifySceneStep  # spec-094 scene tags
from sim_bench.pipeline.steps.score_occlusion import ScoreOcclusionStep  # spec-097 Stage 1

# Face Scoring (individual steps for flexible pipelines)
from sim_bench.pipeline.steps.score_face_pose import ScoreFacePoseStep
from sim_bench.pipeline.steps.score_face_eyes import ScoreFaceEyesStep
from sim_bench.pipeline.steps.score_face_smile import ScoreFaceSmileStep

# Filtering
from sim_bench.pipeline.steps.filter_quality import FilterQualityStep
from sim_bench.pipeline.steps.filter_portraits import FilterPortraitsStep
from sim_bench.pipeline.steps.filter_best_faces import FilterBestFacesStep
from sim_bench.pipeline.steps.filter_faces import FilterFacesStep

# Face Analysis
from sim_bench.pipeline.steps.score_face_frontal import ScoreFaceFrontalStep

# Face Detection & Embedding (MediaPipe)
from sim_bench.pipeline.steps.detect_faces import DetectFacesStep
from sim_bench.pipeline.steps.extract_face_embeddings import ExtractFaceEmbeddingsStep

# Face Alignment Pipeline (single-responsibility steps)
from sim_bench.pipeline.steps.detect_face_orientation import DetectFaceOrientationStep
from sim_bench.pipeline.steps.align_faces import AlignFacesStep
from sim_bench.pipeline.steps.validate_alignment import ValidateAlignmentStep
from sim_bench.pipeline.steps.crop_faces import CropFacesStep
from sim_bench.pipeline.steps.save_face_debug_artifacts import SaveFaceDebugArtifactsStep

# InsightFace Pipeline Steps
from sim_bench.pipeline.steps.detect_persons import DetectPersonsStep
from sim_bench.pipeline.steps.insightface_detect_faces import InsightFaceDetectFacesStep
from sim_bench.pipeline.steps.insightface_score_expression import InsightFaceScoreExpressionStep
from sim_bench.pipeline.steps.insightface_score_eyes import InsightFaceScoreEyesStep
from sim_bench.pipeline.steps.insightface_score_pose import InsightFaceScorePoseStep

# Scene Embedding & Clustering
from sim_bench.pipeline.steps.extract_scene_embedding import ExtractSceneEmbeddingStep
from sim_bench.pipeline.steps.cluster_scenes import ClusterScenesStep

# People Clustering
from sim_bench.pipeline.steps.cluster_people import ClusterPeopleStep
from sim_bench.pipeline.steps.identity_refinement import IdentityRefinementStep
from sim_bench.pipeline.steps.cluster_by_identity import ClusterByIdentityStep

# spec-040 Phase 3: unified face-clustering steps (operate on context.face_records)
from sim_bench.pipeline.steps.face_clustering_steps import (  # noqa: F401
    BuildFaceKNNGraphStep,
    ClusterFaceComponentsStep,
    SelectFaceExemplarsStep,
    MergeFaceClustersStep,
    AttachHoldoutFacesStep,
    ApplyDiameterCapStep,
    AssignPeopleClustersStep,
)

# Face Clustering Experimentation Pipeline
# spec-053: filter_quality_gate + quality_gate_faces consolidated into quality_gate.
from sim_bench.pipeline.steps.quality_gate import QualityGateStep  # noqa: F401
from sim_bench.pipeline.steps.build_knn_graph import BuildKNNGraphStep
from sim_bench.pipeline.steps.cluster_connected_components import ClusterConnectedComponentsStep
from sim_bench.pipeline.steps.select_exemplars import SelectExemplarsStep
from sim_bench.pipeline.steps.compute_debug_distances import ComputeDebugDistancesStep

# Selection
from sim_bench.pipeline.steps.select_best import SelectBestStep
from sim_bench.pipeline.steps.select_best_per_person import SelectBestPerPersonStep

# Export
from sim_bench.pipeline.steps.export_for_labeling import ExportForLabelingStep
# spec-088: FC-app analysis export for the unified clustering chain (SIGHTING-107)
from sim_bench.pipeline.steps.face_cluster_analysis_export import FaceClusterAnalysisExportStep  # noqa: F401

__all__ = [
    # Discovery
    "DiscoverImagesStep",
    # Geo-temporal (spec-022)
    "ExtractGeoMetadataStep",
    "GeoTemporalSegmentStep",
    "InferGeoClipStep",
    "InferGeoCoordsStep",
    "CaptionImagesStep",
    # Scoring
    "ScoreIQAStep",
    "ScoreAVAStep",
    "ScoreFaceQualityStep",
    # Face Scoring (individual)
    "ScoreFacePoseStep",
    "ScoreFaceEyesStep",
    "ScoreFaceSmileStep",
    # Filtering
    "FilterQualityStep",
    "FilterPortraitsStep",
    "FilterBestFacesStep",
    "FilterFacesStep",
    # Face Analysis
    "ScoreFaceFrontalStep",
    # Face Detection & Embedding
    "DetectFacesStep",
    "ExtractFaceEmbeddingsStep",
    # Face Alignment Pipeline (single-responsibility)
    "DetectFaceOrientationStep",
    "AlignFacesStep",
    "ValidateAlignmentStep",
    "CropFacesStep",
    "SaveFaceDebugArtifactsStep",
    # InsightFace Pipeline
    "DetectPersonsStep",
    "InsightFaceDetectFacesStep",
    "InsightFaceScoreExpressionStep",
    "InsightFaceScoreEyesStep",
    "InsightFaceScorePoseStep",
    # Scene Embedding & Clustering
    "ExtractSceneEmbeddingStep",
    "ClusterScenesStep",
    # People Clustering
    "ClusterPeopleStep",
    "IdentityRefinementStep",
    "ClusterByIdentityStep",
    # Face Clustering (consolidated quality_gate per spec-053)
    "QualityGateStep",
    "BuildKNNGraphStep",
    "ClusterConnectedComponentsStep",
    "SelectExemplarsStep",
    "ComputeDebugDistancesStep",
    # Selection
    "SelectBestStep",
    "SelectBestPerPersonStep",
    # Export
    "ExportForLabelingStep",
    "FaceClusterAnalysisExportStep",
]
