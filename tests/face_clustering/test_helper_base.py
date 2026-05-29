"""spec-053 Phase 0 — PipelineHelper protocol imports + is runtime-checkable."""
from __future__ import annotations

from dataclasses import dataclass

from face_cluster._helper_base import PipelineHelper


def test_protocol_importable() -> None:
    """The protocol exists and is generic over (Inputs, Result)."""
    assert PipelineHelper is not None


def test_protocol_is_structural() -> None:
    """Any class with a calc(inputs) method matches the Protocol via
    structural typing — no inheritance required."""

    @dataclass
    class FooInputs:
        x: int

    @dataclass
    class FooResult:
        y: int

    class FooHelper:
        def calc(self, inputs: FooInputs) -> FooResult:
            return FooResult(y=inputs.x * 2)

    helper = FooHelper()
    assert isinstance(helper, PipelineHelper)
    assert helper.calc(FooInputs(x=3)).y == 6


def test_class_without_calc_does_not_match() -> None:
    class NoCalc:
        def something_else(self):  # pragma: no cover
            pass

    assert not isinstance(NoCalc(), PipelineHelper)
