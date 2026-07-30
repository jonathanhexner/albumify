"""Tests for spec-084 Results metrics transparency.

Covers: (AC1) the filter reason is joined per image; (AC2) the composite
breakdown (quality_score + person_penalty == composite_score) is persisted and
exposed; (AC3) every metrics-table column has a tooltip in METRIC_HELP.
"""

import pytest

from sim_bench.pipeline.context import PipelineContext, StepDecision
from sim_bench.api.services.pipeline_service import PipelineService, _build_reason_by_path
from sim_bench.api.services.result_service import ResultService
from sim_bench.pipeline.steps.select_best import SelectBestStep
from app.streamlit.components.metrics import METRIC_HELP, _build_metric_row
from app.streamlit.models import ImageInfo


class ut_ImageMetricsReasonAndBreakdown:
    """_build_image_metrics surfaces the reason + composite breakdown (AC1/AC2)."""

    def _ctx(self, path):
        ctx = PipelineContext()
        ctx.composite_scores[path] = 0.65
        ctx.quality_scores[path] = 0.80
        ctx.person_penalties[path] = -0.15
        ctx.selected_images = [path]
        return ctx

    def test_filter_reason_joined(self):
        path = "D:/x/img.jpg"
        ctx = self._ctx(path)
        reason_by_path = {path: "Best in cluster (score 0.65)"}
        svc = PipelineService(None)

        m = svc._build_image_metrics(ctx, path, reason_by_path)

        assert m["filter_reason"] == "Best in cluster (score 0.65)"

    def test_missing_reason_is_none_not_crash(self):
        path = "D:/x/img.jpg"
        svc = PipelineService(None)
        m = svc._build_image_metrics(self._ctx(path), path, {})  # no entry
        assert m["filter_reason"] is None

    def test_composite_breakdown_exposed(self):
        path = "D:/x/img.jpg"
        svc = PipelineService(None)
        m = svc._build_image_metrics(self._ctx(path), path, {})

        assert m["quality_score"] == 0.80
        assert m["person_penalty"] == -0.15
        assert m["composite_score"] == 0.65
        assert m["quality_score"] + m["person_penalty"] == pytest.approx(m["composite_score"])


class ut_ReasonByPath:
    """Reason aggregation: filtered-early images still get a reason (AC1)."""

    def _d(self, item_id, step, decision, reason, item_type="image"):
        return StepDecision(item_id=item_id, item_type=item_type, step=step,
                            decision=decision, reason=reason)

    def test_select_best_overrides_earlier_step(self):
        # Same image: filter_quality passes it, then select_best selects it.
        decisions = [
            self._d("a.jpg", "filter_quality", "passed", "IQA 0.5 ok"),
            self._d("a.jpg", "select_best", "selected", "Best in cluster (0.8)"),
        ]
        m = _build_reason_by_path(decisions)
        assert m["a.jpg"] == "Best in cluster (0.8)"  # final word wins

    def test_early_filtered_image_keeps_its_reason(self):
        # Image rejected at filter_quality never reaches select_best.
        decisions = [self._d("b.jpg", "filter_quality", "rejected", "IQA 0.08 < 0.20")]
        m = _build_reason_by_path(decisions)
        assert m["b.jpg"] == "IQA 0.08 < 0.20"

    def test_face_decisions_ignored(self):
        decisions = [self._d("f:face_0", "filter_faces", "rejected",
                             "confidence low", item_type="face")]
        assert _build_reason_by_path(decisions) == {}


class ut_CompositeBreakdownPersisted:
    """select_best persists both halves and they sum to the composite (AC2)."""

    def test_quality_and_penalty_persisted_and_sum(self):
        class _FakeQuality:
            def compute_quality(self, image_path, context, siamese_model, image_paths):
                return 0.8

        class _FakePenalty:
            def compute_penalty(self, image_path, context):
                return -0.15

        step = SelectBestStep()
        step._quality_strategy = _FakeQuality()
        step._penalty_computer = _FakePenalty()
        ctx = PipelineContext()
        path = "D:/x/img.jpg"

        scored = step._compute_composite_scores(ctx, [path], None)

        assert ctx.quality_scores[path] == 0.8
        assert ctx.person_penalties[path] == -0.15
        # The composite returned is exactly quality + penalty.
        assert scored[0][1] == pytest.approx(0.65)
        assert (ctx.quality_scores[path] + ctx.person_penalties[path]
                == pytest.approx(scored[0][1]))


class ut_ResultDictForwardsAllMetrics:
    """result_service must forward the full metric set to the UI (no silent drops).

    Regression: Body/Frontal/Central/Roll/BodyPose rendered blank because
    `_build_image_dict` dropped person/frontal/filter fields that were stored.
    """

    def test_person_and_frontal_fields_forwarded(self):
        metrics = {
            "iqa_score": 0.5, "composite_score": 0.65,
            "person_detected": True, "body_facing_score": 0.7,
            "person_confidence": 0.9,
            "best_frontal_score": 0.8, "best_centrality": 0.6,
            "roll_angles": [3.2], "frontal_stats": {"clusterable": 2},
            "filter_stats": {"passed": 2, "filtered": 1},
        }
        svc = ResultService(None)
        d = svc._build_image_dict("img.jpg", metrics)

        # Every column-backing field the table reads must survive the boundary.
        for key in ("person_detected", "body_facing_score", "best_frontal_score",
                    "best_centrality", "roll_angles", "frontal_stats", "filter_stats"):
            assert d[key] == metrics[key], f"{key} dropped by _build_image_dict"


class ut_MetricTooltips:
    """Every column the table builds must have a tooltip (AC3)."""

    def test_every_column_has_help_entry(self):
        img = ImageInfo(
            path="x.jpg", filename="x.jpg",
            composite_score=0.65, quality_score=0.80, person_penalty=-0.15,
            filter_reason="Best in cluster", is_selected=True,
        )
        row = _build_metric_row(img, is_sel=True)

        missing = set(row.keys()) - set(METRIC_HELP.keys())
        assert not missing, f"columns without a tooltip: {missing}"

    def test_help_texts_are_non_empty(self):
        assert all(isinstance(v, str) and v.strip() for v in METRIC_HELP.values())
