"""ut for OcclusionPenaltyComputer (spec-097 Stage 1) — pure math, no models."""

import pytest

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.scoring.occlusion_penalty import OcclusionPenaltyFactory


def _ctx(scores=None, tiles=None):
    ctx = PipelineContext(image_paths=[])
    ctx.occlusion_scores = scores or {}
    ctx.occlusion_tiles = tiles or {}
    return ctx


def test_below_gate_is_exactly_zero_bokeh_safety():
    """The bokeh/night-shot guarantee: any P below the gate -> NO penalty.

    (Empirical half of this guarantee: 0/60 motion-blurred and 0/60 defocused
    clean images crossed the gate — scripts/train_occlusion_artifact.py.)"""
    comp = OcclusionPenaltyFactory.create({})
    ctx = _ctx({"a.jpg": 0.79}, {"a.jpg": [0.9] * 9})
    assert comp.compute_penalty("a.jpg", ctx) == 0.0


def test_missing_scores_mean_zero():
    """Pipelines without score_occlusion upstream are unaffected."""
    comp = OcclusionPenaltyFactory.create({})
    assert comp.compute_penalty("a.jpg", _ctx()) == 0.0


def test_gated_penalty_scales_with_area():
    comp = OcclusionPenaltyFactory.create({})
    one_tile = _ctx({"a.jpg": 0.9}, {"a.jpg": [0.9] + [0.0] * 8})
    all_tiles = _ctx({"a.jpg": 0.9}, {"a.jpg": [0.9] * 9})
    p_small = comp.compute_penalty("a.jpg", one_tile)
    p_large = comp.compute_penalty("a.jpg", all_tiles)
    assert p_small == pytest.approx(-0.35 * 0.9 * (0.4 + 0.6 / 9))
    assert p_large == pytest.approx(-0.35 * 0.9 * 1.0)  # floor (-0.5) not reached
    assert p_large < p_small  # bigger occlusion hurts more


def test_floor_is_respected():
    comp = OcclusionPenaltyFactory.create({"weight": -2.0})
    ctx = _ctx({"a.jpg": 1.0}, {"a.jpg": [1.0] * 9})
    assert comp.compute_penalty("a.jpg", ctx) == pytest.approx(-0.5)


def test_disabled_kills_the_term():
    comp = OcclusionPenaltyFactory.create({"enabled": False})
    ctx = _ctx({"a.jpg": 1.0}, {"a.jpg": [1.0] * 9})
    assert comp.compute_penalty("a.jpg", ctx) == 0.0


def test_path_separator_tolerance():
    """Same rule as person_penalty: context may key by forward slashes."""
    comp = OcclusionPenaltyFactory.create({})
    ctx = _ctx({"C:/x/a.jpg": 0.9}, {"C:/x/a.jpg": [0.9] * 9})
    assert comp.compute_penalty("C:\\x\\a.jpg", ctx) < 0.0
