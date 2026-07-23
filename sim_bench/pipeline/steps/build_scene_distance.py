"""Build scene distance step (spec-103) - fuse visual similarity with a short-range capture-time boost.

Thin translator (spec-053): reads context.scene_embeddings (+ optional geo_metadata for EXIF timestamps),
calls the framework-agnostic SceneDistanceBuilder (Path A), writes the precomputed NxN distance to
context.scene_distance. cluster_scenes then clusters that distance instead of the raw embeddings.

Gated by PRESENCE, not a flag: this step is NOT in default_pipeline. When it is absent, cluster_scenes
sees no scene_distance and clusters embeddings exactly as before (byte-identical). Opt-in pipelines insert
build_scene_distance between extract_scene_embedding and cluster_scenes.
"""

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step


@register_step
class BuildSceneDistanceStep(BaseStep):
    """Fuse DINOv2 visual distance with a one-sided short-range (<=~1 min) capture-time boost."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="build_scene_distance",
            display_name="Build Scene Distance",
            description=("Fuse visual similarity with a short-range capture-time boost into a "
                         "precomputed scene distance (spec-103 Path A). Gated by presence; default off."),
            category="clustering",
            requires={"scene_embeddings"},
            produces={"scene_distance", "scene_distance_signal"},
            depends_on=["extract_scene_embedding"],
            config_schema={
                "type": "object",
                "properties": {
                    "boost": {
                        "type": "number", "default": 0.6, "minimum": 0.0, "maximum": 1.0,
                        "description": "Strength of the short-range time pull (0=off, 1=collapse "
                                       "near-simultaneous photos)."
                    },
                    "tau_sec": {
                        "type": "number", "default": 60.0, "minimum": 1.0,
                        "description": "Time constant (s): photos within ~this gap get the boost, "
                                       "beyond ~3x it is negligible."
                    },
                },
            },
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if not context.scene_embeddings:
            context.report_progress("build_scene_distance", 1.0, "No embeddings to fuse")
            return

        from sim_bench.scene_cluster.geo_time_fusion import SceneDistanceBuilder, SceneDistanceInputs

        image_ids = list(context.scene_embeddings.keys())
        builder = SceneDistanceBuilder(
            boost=config.get("boost", 0.6),
            tau_sec=config.get("tau_sec", 60.0),
        )
        result = builder.calc(SceneDistanceInputs(
            embeddings=context.scene_embeddings,
            geo_metadata=context.geo_metadata or {},
            image_ids=image_ids,
        ))
        context.scene_distance = result
        context.scene_distance_signal = result.per_image_signal_used
        n_time = sum(1 for sig in result.per_image_signal_used.values() if "time" in sig)
        context.report_progress(
            "build_scene_distance", 1.0,
            f"Built {len(image_ids)}x{len(image_ids)} scene distance ({n_time} with capture time)",
        )
