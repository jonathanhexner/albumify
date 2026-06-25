"""spec-085: ImageMetrics is the single source of truth for the image API contract.

These tests fail loudly if the producer (`_build_image_metrics`) ever emits a field
that the contract (`ImageMetrics`) doesn't declare — the exact drift that silently
blanked the Results columns (spec-084).
"""

from sim_bench.pipeline.context import PipelineContext
from sim_bench.api.services.pipeline_service import PipelineService
from sim_bench.api.services.result_service import ResultService
from sim_bench.api.schemas.result import ImageMetrics


class ut_ImageMetricsContract:

    def test_producer_emits_exactly_the_contract(self):
        """spec-085 C-lite: the producer builds ImageMetrics directly, so its
        output is EXACTLY the contract field set — not a subset, not a superset."""
        ctx = PipelineContext()
        path = "a.jpg"
        produced = PipelineService(None)._build_image_metrics(ctx, path, {})

        assert set(produced.keys()) == set(ImageMetrics.model_fields)

    def test_build_image_dict_matches_contract_exactly(self):
        """_build_image_dict output keys == ImageMetrics fields (no hand list) (AC2)."""
        d = ResultService(None)._build_image_dict("a.jpg", {})
        assert set(d.keys()) == set(ImageMetrics.model_fields)

    def test_previously_dropped_fields_now_survive(self):
        """The fields that used to vanish at the API boundary now round-trip (AC3)."""
        metrics = {
            "person_detected": True, "best_frontal_score": 0.8,
            "roll_angles": [3.2], "filter_reason": "Best in cluster (0.8)",
            "quality_score": 0.8, "person_penalty": -0.15,
        }
        d = ResultService(None)._build_image_dict("a.jpg", metrics)
        for k, v in metrics.items():
            assert d[k] == v, f"{k} did not survive _build_image_dict"
