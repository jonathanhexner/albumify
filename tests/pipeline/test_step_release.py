"""SIGHTING-117: model-owning steps must free their model after running, and
the executor must call release() after EVERY step (success or failure) so peak
RSS is one model, not the sum of all.

Covers:
  ut_BaseStep._release_models        — nulls handles + no-op when nothing held
  ut_ModelStep.release               — a step's release() drops its model handle
  ut_PipelineExecutor.release_called — executor calls release() on success AND failure
"""

import pytest

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import StepRegistry


class _Sentinel:
    """Stand-in for a heavy model handle."""


def _meta(name: str) -> StepMetadata:
    return StepMetadata(name=name, display_name=name, description="test",
                        category="scoring", requires=set(), produces=set())


class _FakeModelStep(BaseStep):
    """Minimal step that lazy-holds a 'model' and frees it in release()."""

    def __init__(self, name="fake_model_step", raises=False):
        self._metadata = _meta(name)
        self._model = None
        self._config_tracker = "keep-me"
        self._raises = raises
        self.release_calls = 0

    def process(self, context, config):
        self._model = _Sentinel()          # simulate lazy model load
        if self._raises:
            raise RuntimeError("boom")

    def release(self):
        self.release_calls += 1
        self._release_models("_model")


def test_release_models_nulls_handles_and_is_noop_when_empty():
    """ut_BaseStep._release_models: nulls named attrs; safe when already None."""
    step = _FakeModelStep()
    step._model = _Sentinel()
    step._release_models("_model")
    assert step._model is None
    # config tracker is deliberately NOT cleared (reload guard uses `model is None`)
    assert step._config_tracker == "keep-me"
    # second call with nothing held must not raise
    step._release_models("_model")
    assert step._model is None


def test_step_release_drops_model_handle():
    """ut_ModelStep.release: after release() the model handle is gone."""
    step = _FakeModelStep()
    step.process(PipelineContext(), {})
    assert step._model is not None
    step.release()
    assert step._model is None


def test_executor_calls_release_on_success():
    """ut_PipelineExecutor.release_called: release() runs after a good step."""
    step = _FakeModelStep()
    executor = PipelineExecutor(StepRegistry())
    result = executor._execute_step(step, PipelineContext(), PipelineConfig())
    assert result.success is True
    assert step.release_calls == 1
    assert step._model is None            # freed, not left resident


def test_executor_calls_release_on_failure():
    """release() must still run when the step raises (finally-block)."""
    step = _FakeModelStep(raises=True)
    executor = PipelineExecutor(StepRegistry())
    result = executor._execute_step(step, PipelineContext(), PipelineConfig())
    assert result.success is False
    assert step.release_calls == 1        # released despite the exception
    assert step._model is None


def test_real_occlusion_step_release_frees_scorer_without_loading():
    """The production occlusion step frees its CLIP-backed scorer on release()."""
    from sim_bench.pipeline.steps.score_occlusion import ScoreOcclusionStep
    step = ScoreOcclusionStep()
    step._scorer = _Sentinel()            # pretend it was lazy-loaded
    step.release()
    assert step._scorer is None


def test_base_release_default_is_noop():
    """A step that holds no model inherits a harmless no-op release()."""
    class _NoModel(BaseStep):
        def __init__(self):
            self._metadata = _meta("no_model")
        def process(self, context, config):
            pass
    _NoModel().release()                  # must not raise
