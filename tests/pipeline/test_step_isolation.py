"""spec-104 v1 — per-step process isolation (executor-owned, default-off flag).

Proves the executor can run a step in a subprocess that exits (freeing model
memory), applying the child's context mutations back onto the parent, and that a
crash/exception in the child is contained (parent survives with a failed result).
"""
import os

from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import StepRegistry

from tests.pipeline import _isolation_fixtures as fx


def _run(step_cls, isolate: bool):
    registry = StepRegistry()
    registry.register(step_cls)
    executor = PipelineExecutor(registry)
    ctx = PipelineContext()
    cfg = PipelineConfig(isolate_steps=isolate)
    name = step_cls().metadata.name
    result = executor.execute(ctx, [name], config=cfg)
    return ctx, result


def test_isolated_step_applies_produces_back_to_parent():
    ctx, result = _run(fx.WriteValueStep, isolate=True)
    assert result.success, result.error_message
    # the mutation happened in the child but is visible on the parent context
    assert ctx.iqa_scores == {"a.jpg": 0.9, "b.jpg": 0.1}


def test_isolated_step_runs_in_a_child_process():
    ctx, result = _run(fx.ChildPidStep, isolate=True)
    assert result.success, result.error_message
    assert ctx.metrics["child_pid"] != os.getpid()  # ran in a separate process


def test_default_off_is_identical_to_in_process():
    ctx_in, res_in = _run(fx.WriteValueStep, isolate=False)
    ctx_iso, res_iso = _run(fx.WriteValueStep, isolate=True)
    assert res_in.success and res_iso.success
    assert ctx_in.iqa_scores == ctx_iso.iqa_scores  # same observable result


def test_in_process_runs_in_same_process():
    ctx, result = _run(fx.ChildPidStep, isolate=False)
    assert result.success
    assert ctx.metrics["child_pid"] == os.getpid()  # sanity: flag OFF stays in-process


def test_isolated_exception_becomes_failed_result():
    ctx, result = _run(fx.RaiseStep, isolate=True)
    assert not result.success
    assert "boom in child" in result.error_message


def test_isolated_crash_parent_survives():
    ctx, result = _run(fx.HardCrashStep, isolate=True)
    assert not result.success                       # parent SURVIVED and reported it
    assert "crash" in result.error_message.lower()
