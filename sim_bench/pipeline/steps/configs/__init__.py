"""spec-033 P-G + spec-040 Phase 2: typed Pydantic step configs.

One BaseModel per face-clustering step. Misspelled or unknown keys at the
UI→step boundary raise ValidationError instead of silently defaulting.

Each model declares ``model_config = ConfigDict(extra="forbid")`` so the
contract is bidirectional: the UI can't write a typo'd key, and the step
can't read a typo'd attribute either.

``Field(description=...)`` is the SINGLE SOURCE OF TRUTH for the UI ``help=``
text — eliminating the label-vs-reality drift class that motivated spec-033 P-A.
"""
from sim_bench.pipeline.steps.configs.align_faces import AlignFacesConfig
from sim_bench.pipeline.steps.configs.cluster_people import ClusterPeopleConfig
from sim_bench.pipeline.steps.configs.cluster_scenes import ClusterScenesConfig
from sim_bench.pipeline.steps.configs.detect_face_orientation import DetectFaceOrientationConfig
from sim_bench.pipeline.steps.configs.detect_persons import DetectPersonsConfig
from sim_bench.pipeline.steps.configs.extract_face_embeddings import ExtractFaceEmbeddingsConfig
from sim_bench.pipeline.steps.configs.extract_scene_embedding import ExtractSceneEmbeddingConfig
from sim_bench.pipeline.steps.configs.filter_faces import FilterFacesConfig
from sim_bench.pipeline.steps.configs.filter_quality import FilterQualityConfig
from sim_bench.pipeline.steps.configs.insightface_detect_faces import InsightFaceDetectFacesConfig
from sim_bench.pipeline.steps.configs.insightface_score_expression import InsightFaceScoreExpressionConfig
from sim_bench.pipeline.steps.configs.insightface_score_eyes import InsightFaceScoreEyesConfig
from sim_bench.pipeline.steps.configs.insightface_score_pose import InsightFaceScorePoseConfig
from sim_bench.pipeline.steps.configs.score_ava import ScoreAVAConfig
from sim_bench.pipeline.steps.configs.score_face_frontal import ScoreFaceFrontalConfig
from sim_bench.pipeline.steps.configs.score_iqa import ScoreIQAConfig

# Registry: step_name → config model.  Used by the architecture test and by
# any tooling that wants to introspect "what config does step X take?".
STEP_CONFIG_MODELS = {
    # Original 5 (spec-033 P-G):
    "cluster_people":               ClusterPeopleConfig,
    "extract_face_embeddings":      ExtractFaceEmbeddingsConfig,
    "filter_faces":                 FilterFacesConfig,
    "filter_quality":               FilterQualityConfig,
    "insightface_detect_faces":     InsightFaceDetectFacesConfig,
    # Added in spec-040 Phase 2:
    "align_faces":                  AlignFacesConfig,
    "cluster_scenes":               ClusterScenesConfig,
    "detect_face_orientation":      DetectFaceOrientationConfig,
    "detect_persons":               DetectPersonsConfig,
    "extract_scene_embedding":      ExtractSceneEmbeddingConfig,
    "insightface_score_expression": InsightFaceScoreExpressionConfig,
    "insightface_score_eyes":       InsightFaceScoreEyesConfig,
    "insightface_score_pose":       InsightFaceScorePoseConfig,
    "score_ava":                    ScoreAVAConfig,
    "score_face_frontal":           ScoreFaceFrontalConfig,
    "score_iqa":                    ScoreIQAConfig,
}

__all__ = [
    "AlignFacesConfig",
    "ClusterPeopleConfig",
    "ClusterScenesConfig",
    "DetectFaceOrientationConfig",
    "DetectPersonsConfig",
    "ExtractFaceEmbeddingsConfig",
    "ExtractSceneEmbeddingConfig",
    "FilterFacesConfig",
    "FilterQualityConfig",
    "InsightFaceDetectFacesConfig",
    "InsightFaceScoreExpressionConfig",
    "InsightFaceScoreEyesConfig",
    "InsightFaceScorePoseConfig",
    "ScoreAVAConfig",
    "ScoreFaceFrontalConfig",
    "ScoreIQAConfig",
    "STEP_CONFIG_MODELS",
]
