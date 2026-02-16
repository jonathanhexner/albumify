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

# Selection
from sim_bench.pipeline.steps.select_best import SelectBestStep
from sim_bench.pipeline.steps.select_best_per_person import SelectBestPerPersonStep

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
    # Selection
    "SelectBestStep",
    "SelectBestPerPersonStep",
]
