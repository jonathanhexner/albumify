"""Filter quality step - filter images by quality thresholds."""

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step


@register_step
class FilterQualityStep(BaseStep):
    """Filter images based on quality score thresholds."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="filter_quality",
            display_name="Filter by Quality",
            description="Filter out images that don't meet quality thresholds (IQA, sharpness).",
            category="filtering",
            requires={"iqa_scores"},
            produces={"quality_passed", "active_images"},
            depends_on=["score_iqa"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_iqa_score": {
                        "type": "number",
                        "default": 0.3,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum IQA score (0-1)"
                    },
                    "min_sharpness": {
                        "type": "number",
                        "default": 0.2,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum sharpness score (0-1)"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        min_iqa = config.get("min_iqa_score", 0.3)
        min_sharpness = config.get("min_sharpness", 0.2)

        passed = set()
        total = len(context.iqa_scores)

        for path_str, iqa_score in context.iqa_scores.items():
            sharpness = context.sharpness_scores.get(path_str, 1.0)

            iqa_ok = iqa_score >= min_iqa
            sharpness_ok = sharpness >= min_sharpness

            if iqa_ok and sharpness_ok:
                passed.add(path_str)

        context.quality_passed = passed
        context.active_images = passed.copy()

        context.report_progress(
            "filter_quality",
            1.0,
            f"Passed {len(passed)}/{total} images (IQA>={min_iqa}, sharpness>={min_sharpness})"
        )
