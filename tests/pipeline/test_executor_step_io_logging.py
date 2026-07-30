"""spec-041 follow-up — executor emits per-step in/out telemetry.

The executor logs an INFO line after every successful step:

    <step_name>: in[<requires_summary>] -> out[<produces_summary>] (<duration>ms)

And an ERROR line on validation failure with the input shape, so the
post-mortem of "step X aborted because Y was empty" is one grep, not a
walk back through prior steps.
"""
from __future__ import annotations

import logging
from typing import Set

import pytest

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import (
    PipelineExecutor,
    _summarize_keys,
    _summarize_one,
)
from sim_bench.pipeline.registry import StepRegistry, register_step


def _isolated_registry() -> StepRegistry:
    """Make a fresh registry so test-only steps don't leak into others."""
    return StepRegistry()


class _PopulateStep(BaseStep):
    """Test step: writes `out_list` to the context."""
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="t_populate",
            display_name="Populate",
            description="",
            category="test",
            requires=set(),
            produces={"out_list"},
            depends_on=[],
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        context.out_list = [1, 2, 3]


class _ConsumeStep(BaseStep):
    """Test step: requires `out_list`."""
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="t_consume",
            display_name="Consume",
            description="",
            category="test",
            requires={"out_list"},
            produces={"result"},
            depends_on=["t_populate"],
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        context.result = sum(context.out_list)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_summarize_one_list_shows_length():
    ctx = PipelineContext()
    ctx.face_records = [1, 2, 3, 4, 5]
    assert _summarize_one(ctx, "face_records") == "face_records=5"


def test_summarize_one_none_is_explicit():
    ctx = PipelineContext()
    ctx.face_records = None
    assert _summarize_one(ctx, "face_records") == "face_records=None"


def test_summarize_one_missing_is_explicit():
    ctx = PipelineContext()
    assert _summarize_one(ctx, "never_set") == "never_set=missing"


def test_summarize_one_scalar():
    ctx = PipelineContext()
    ctx.some_count = 42
    assert _summarize_one(ctx, "some_count") == "some_count=<scalar>"


def test_summarize_keys_empty_uses_dash():
    ctx = PipelineContext()
    assert _summarize_keys(ctx, set()) == "—"


def test_summarize_keys_orders_alphabetically():
    ctx = PipelineContext()
    ctx.b_key = [1]
    ctx.a_key = [1, 2]
    out = _summarize_keys(ctx, {"b_key", "a_key"})
    # alphabetical ordering means a_key shows first
    assert out.startswith("a_key=2")


# ---------------------------------------------------------------------------
# Integration: real executor + capturing logs
# ---------------------------------------------------------------------------

def _run_two_step_chain(caplog) -> None:
    registry = _isolated_registry()
    registry.register(_PopulateStep)
    registry.register(_ConsumeStep)
    executor = PipelineExecutor(registry)
    ctx = PipelineContext()
    cfg = PipelineConfig(fail_fast=True)
    with caplog.at_level(logging.INFO, logger="sim_bench.pipeline.executor"):
        result = executor.execute(ctx, ["t_populate", "t_consume"], config=cfg)
    assert result.success, result.error_message


def test_executor_logs_in_out_for_each_successful_step(caplog):
    _run_two_step_chain(caplog)
    messages = [r.message for r in caplog.records]

    # t_populate has no requires → "—"; produces out_list (length 3 after run)
    assert any(
        "t_populate: in[—] -> out[out_list=3]" in m for m in messages
    ), f"missing t_populate telemetry line. captured: {messages}"

    # t_consume requires out_list (length 3); produces result (scalar after run)
    assert any(
        "t_consume: in[out_list=3] -> out[result=<scalar>]" in m for m in messages
    ), f"missing t_consume telemetry line. captured: {messages}"


def test_executor_logs_validation_failure_with_input_shape(caplog):
    """When a step's validator rejects, the log line names the input shape
    that triggered the rejection — so a post-mortem of "X was empty" doesn't
    need to scroll through earlier steps."""
    registry = _isolated_registry()
    registry.register(_ConsumeStep)  # requires out_list
    executor = PipelineExecutor(registry)
    ctx = PipelineContext()  # out_list never set
    cfg = PipelineConfig(fail_fast=True)
    with caplog.at_level(logging.ERROR, logger="sim_bench.pipeline.executor"):
        result = executor.execute(ctx, ["t_consume"], config=cfg)
    assert not result.success
    error_lines = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "t_consume: validation failed (in[out_list=missing])" in m for m in error_lines
    ), f"missing input-shape in validation error. captured: {error_lines}"
