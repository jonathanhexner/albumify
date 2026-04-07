"""Import all steps to register them with the global registry."""

# Discovery
from sim_bench.pipeline.steps.discover_images import DiscoverImagesStep

# Scoring
from sim_bench.pipeline.steps.score_iqa import ScoreIQAStep
from sim_bench.pipeline.steps.score_ava import ScoreAVAStep
from sim_bench.pipeline.steps.score_face_quality import ScoreFaceQualityStep

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

# Face Clustering Experimentation Pipeline
from sim_bench.pipeline.steps.filter_quality_gate import FilterQualityGateStep
from sim_bench.pipeline.steps.build_knn_graph import BuildKNNGraphStep
from sim_bench.pipeline.steps.cluster_connected_components import ClusterConnectedComponentsStep
from sim_bench.pipeline.steps.select_exemplars import SelectExemplarsStep
from sim_bench.pipeline.steps.compute_debug_distances import ComputeDebugDistancesStep

# Selection
from sim_bench.pipeline.steps.select_best import SelectBestStep
from sim_bench.pipeline.steps.select_best_per_person import SelectBestPerPersonStep

# Export
from sim_bench.pipeline.steps.export_for_labeling import ExportForLabelingStep

__all__ = [
    # Discovery
    "DiscoverImagesStep",
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
    # Face Clustering Experimentation
    "FilterQualityGateStep",
    "BuildKNNGraphStep",
    "ClusterConnectedComponentsStep",
    "SelectExemplarsStep",
    "ComputeDebugDistancesStep",
    # Selection
    "SelectBestStep",
    "SelectBestPerPersonStep",
    # Export
    "ExportForLabelingStep",
]
