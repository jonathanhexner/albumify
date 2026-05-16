"""spec-033 P-G: typed Pydantic step configs.

One BaseModel per face-clustering step. Misspelled or unknown keys at the
UI→step boundary raise ValidationError instead of silently defaulting.

Each model declares ``model_config = ConfigDict(extra="forbid")`` so the
contract is bidirectional: the UI can't write a typo'd key, and the step
can't read a typo'd attribute either.

``Field(description=...)`` is the SINGLE SOURCE OF TRUTH for the UI ``help=``
text — eliminating the label-vs-reality drift class that motivated spec-033 P-A.
"""
from sim_bench.pipeline.steps.configs.cluster_people import ClusterPeopleConfig
from sim_bench.pipeline.steps.configs.extract_face_embeddings import ExtractFaceEmbeddingsConfig
from sim_bench.pipeline.steps.configs.filter_faces import FilterFacesConfig
from sim_bench.pipeline.steps.configs.filter_quality import FilterQualityConfig
from sim_bench.pipeline.steps.configs.insightface_detect_faces import InsightFaceDetectFacesConfig

# Registry: step_name → config model.  Used by the architecture test and by
# any tooling that wants to introspect "what config does step X take?".
STEP_CONFIG_MODELS = {
    "cluster_people":             ClusterPeopleConfig,
    "extract_face_embeddings":    ExtractFaceEmbeddingsConfig,
    "filter_faces":               FilterFacesConfig,
    "filter_quality":             FilterQualityConfig,
    "insightface_detect_faces":   InsightFaceDetectFacesConfig,
}

__all__ = [
    "ClusterPeopleConfig",
    "ExtractFaceEmbeddingsConfig",
    "FilterFacesConfig",
    "FilterQualityConfig",
    "InsightFaceDetectFacesConfig",
    "STEP_CONFIG_MODELS",
]
