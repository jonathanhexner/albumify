"""Filter quality step - filter images by quality thresholds."""

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext, StepDecision
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

        cfg = {"min_iqa_score": min_iqa, "min_sharpness": min_sharpness}

        for path_str, iqa_score in context.iqa_scores.items():
            sharpness = context.sharpness_scores.get(path_str, 1.0)

            iqa_ok = iqa_score >= min_iqa
            sharpness_ok = sharpness >= min_sharpness

            if iqa_ok and sharpness_ok:
                passed.add(path_str)
                reason = f"Passed (IQA {iqa_score:.2f}, sharpness {sharpness:.2f})"
                decision = "passed"
            elif not iqa_ok and not sharpness_ok:
                reason = f"IQA {iqa_score:.2f} < {min_iqa} AND sharpness {sharpness:.2f} < {min_sharpness}"
                decision = "rejected"
            elif not iqa_ok:
                reason = f"IQA {iqa_score:.2f} < threshold {min_iqa}"
                decision = "rejected"
            else:
                reason = f"Sharpness {sharpness:.2f} < threshold {min_sharpness}"
                decision = "rejected"

            context.step_decisions.append(StepDecision(
                item_id=path_str, item_type="image", step="filter_quality",
                decision=decision, reason=reason, config_used=cfg,
                metrics={"iqa_score": round(iqa_score, 3), "sharpness": round(sharpness, 3)},
            ))

        context.quality_passed = passed
        context.active_images = passed.copy()

        context.report_progress(
            "filter_quality",
            1.0,
            f"Passed {len(passed)}/{total} images (IQA>={min_iqa}, sharpness>={min_sharpness})"
        )
