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

# Face Detection & Embedding
from sim_bench.pipeline.steps.detect_faces import DetectFacesStep
from sim_bench.pipeline.steps.extract_face_embeddings import ExtractFaceEmbeddingsStep

# Scene Embedding & Clustering
from sim_bench.pipeline.steps.extract_scene_embedding import ExtractSceneEmbeddingStep
from sim_bench.pipeline.steps.cluster_scenes import ClusterScenesStep

# People Clustering
from sim_bench.pipeline.steps.cluster_people import ClusterPeopleStep
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
    # Face Detection & Embedding
    "DetectFacesStep",
    "ExtractFaceEmbeddingsStep",
    # Scene Embedding & Clustering
    "ExtractSceneEmbeddingStep",
    "ClusterScenesStep",
    # People Clustering
    "ClusterPeopleStep",
    "ClusterByIdentityStep",
    # Selection
    "SelectBestStep",
    "SelectBestPerPersonStep",
]
