"""Tests for pipeline step decision records (StepDecision).

Verifies that pipeline steps emit structured decision records that the UI
can display without reimplementing pipeline logic.
"""
import pytest
from unittest.mock import MagicMock
from sim_bench.pipeline.context import PipelineContext, StepDecision


class ut_FilterQualityDecisions:
    """filter_quality step must emit per-image decisions with actual thresholds."""

    def test_emits_decision_for_every_image(self):
        """Every image gets a decision record, whether passed or rejected."""
        from sim_bench.pipeline.steps.filter_quality import FilterQualityStep

        ctx = PipelineContext()
        ctx.iqa_scores = {"a.jpg": 0.5, "b.jpg": 0.1, "c.jpg": 0.8}
        ctx.sharpness_scores = {"a.jpg": 0.3, "b.jpg": 0.5, "c.jpg": 0.05}

        step = FilterQualityStep()
        step.process(ctx, {"min_iqa_score": 0.2, "min_sharpness": 0.1})

        assert len(ctx.step_decisions) == 3, f"Expected 3 decisions, got {len(ctx.step_decisions)}"

    def test_passed_images_have_passed_decision(self):
        from sim_bench.pipeline.steps.filter_quality import FilterQualityStep

        ctx = PipelineContext()
        ctx.iqa_scores = {"good.jpg": 0.8}
        ctx.sharpness_scores = {"good.jpg": 0.5}

        step = FilterQualityStep()
        step.process(ctx, {"min_iqa_score": 0.2, "min_sharpness": 0.1})

        d = ctx.step_decisions[0]
        assert d.decision == "passed"
        assert d.step == "filter_quality"
        assert d.item_id == "good.jpg"

    def test_rejected_iqa_shows_actual_threshold(self):
        """Reason must contain the actual threshold value, not a hardcoded constant."""
        from sim_bench.pipeline.steps.filter_quality import FilterQualityStep

        ctx = PipelineContext()
        ctx.iqa_scores = {"bad.jpg": 0.08}
        ctx.sharpness_scores = {"bad.jpg": 0.5}

        step = FilterQualityStep()
        step.process(ctx, {"min_iqa_score": 0.25, "min_sharpness": 0.1})

        d = ctx.step_decisions[0]
        assert d.decision == "rejected"
        assert "0.08" in d.reason, f"Reason should contain actual IQA value: {d.reason}"
        assert "0.25" in d.reason, f"Reason should contain threshold used: {d.reason}"

    def test_config_used_contains_actual_thresholds(self):
        """config_used must reflect the ACTUAL config, not defaults."""
        from sim_bench.pipeline.steps.filter_quality import FilterQualityStep

        ctx = PipelineContext()
        ctx.iqa_scores = {"img.jpg": 0.5}
        ctx.sharpness_scores = {"img.jpg": 0.5}

        step = FilterQualityStep()
        step.process(ctx, {"min_iqa_score": 0.35, "min_sharpness": 0.15})

        d = ctx.step_decisions[0]
        assert d.config_used["min_iqa_score"] == 0.35
        assert d.config_used["min_sharpness"] == 0.15

    def test_metrics_contain_measured_values(self):
        from sim_bench.pipeline.steps.filter_quality import FilterQualityStep

        ctx = PipelineContext()
        ctx.iqa_scores = {"img.jpg": 0.42}
        ctx.sharpness_scores = {"img.jpg": 0.67}

        step = FilterQualityStep()
        step.process(ctx, {"min_iqa_score": 0.2, "min_sharpness": 0.1})

        d = ctx.step_decisions[0]
        assert d.metrics["iqa_score"] == 0.42
        assert d.metrics["sharpness"] == 0.67


class ut_SelectBestDecisions:
    """select_best step must emit per-image decisions with score breakdown."""

    def _make_context_with_cluster(self):
        """Create a context with one scene cluster of 3 images."""
        import numpy as np
        ctx = PipelineContext()
        ctx.scene_clusters = {0: ["a.jpg", "b.jpg", "c.jpg"]}
        ctx.iqa_scores = {"a.jpg": 0.8, "b.jpg": 0.5, "c.jpg": 0.3}
        ctx.ava_scores = {"a.jpg": 0.9, "b.jpg": 0.6, "c.jpg": 0.4}
        ctx.sharpness_scores = {"a.jpg": 0.7, "b.jpg": 0.5, "c.jpg": 0.3}
        ctx.scene_embeddings = {
            "a.jpg": np.random.randn(512).astype(np.float32),
            "b.jpg": np.random.randn(512).astype(np.float32),
            "c.jpg": np.random.randn(512).astype(np.float32),
        }
        ctx.persons = {}
        ctx.insightface_faces = {}
        ctx.face_clusters = {}
        return ctx

    def test_emits_decision_for_every_image_in_cluster(self):
        from sim_bench.pipeline.steps.select_best import SelectBestStep

        ctx = self._make_context_with_cluster()
        step = SelectBestStep()
        config = {
            "max_images_per_cluster": 1,
            "min_score_threshold": 0.0,
            "quality_strategy": "weighted_average",
            "siamese": {"enabled": False},
        }
        step.process(ctx, config)

        select_decisions = [d for d in ctx.step_decisions if d.step == "select_best"]
        assert len(select_decisions) == 3, f"Expected 3 decisions, got {len(select_decisions)}"

    def test_best_image_has_selected_decision(self):
        from sim_bench.pipeline.steps.select_best import SelectBestStep

        ctx = self._make_context_with_cluster()
        step = SelectBestStep()
        config = {
            "max_images_per_cluster": 1,
            "min_score_threshold": 0.0,
            "quality_strategy": "weighted_average",
            "siamese": {"enabled": False},
        }
        step.process(ctx, config)

        select_decisions = [d for d in ctx.step_decisions if d.step == "select_best"]
        selected = [d for d in select_decisions if d.decision == "selected"]
        assert len(selected) == 1
        assert "Best" in selected[0].reason or "Rank 1" in selected[0].reason

    def test_rejected_shows_rank_and_score(self):
        from sim_bench.pipeline.steps.select_best import SelectBestStep

        ctx = self._make_context_with_cluster()
        step = SelectBestStep()
        config = {
            "max_images_per_cluster": 1,
            "min_score_threshold": 0.0,
            "quality_strategy": "weighted_average",
            "siamese": {"enabled": False},
        }
        step.process(ctx, config)

        select_decisions = [d for d in ctx.step_decisions if d.step == "select_best"]
        rejected = [d for d in select_decisions if d.decision == "rejected"]
        assert len(rejected) >= 1
        # Reason should mention rank, not a hardcoded threshold
        for d in rejected:
            assert "rank" in d.reason.lower() or "outranked" in d.reason.lower() or "score" in d.reason.lower(), \
                f"Rejected reason should explain ranking: {d.reason}"

    def test_no_hardcoded_thresholds_in_reasons(self):
        """Decision reasons must never contain hardcoded values — only actual config values."""
        from sim_bench.pipeline.steps.select_best import SelectBestStep

        ctx = self._make_context_with_cluster()
        step = SelectBestStep()
        config = {
            "max_images_per_cluster": 1,
            "min_score_threshold": 0.55,  # Non-default threshold
            "quality_strategy": "weighted_average",
            "siamese": {"enabled": False},
        }
        step.process(ctx, config)

        select_decisions = [d for d in ctx.step_decisions if d.step == "select_best"]
        for d in select_decisions:
            if "threshold" in d.reason.lower():
                # If threshold is mentioned, it must be the actual one (0.55), not 0.4
                assert "0.55" in d.reason, \
                    f"Reason mentions threshold but uses wrong value: {d.reason}"
