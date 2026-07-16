"""ut for TiltPenaltyComputer (spec-099 Phase 2 / spec-100) — pure math, no models."""

import pytest

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.scoring.tilt_penalty import TiltPenaltyFactory


def _ctx(angles=None, confs=None):
    ctx = PipelineContext(image_paths=[])
    ctx.tilt_angles = angles or {}
    ctx.tilt_confidences = confs or {}
    return ctx


def test_low_confidence_is_exactly_zero():
    """A guessed tilt never moves a score: below conf_gate -> no penalty.

    This is what keeps GeoCalib safe on structureless scenes (kaleidoscope /
    mirror shots come back with huge uncertainty -> low confidence)."""
    comp = TiltPenaltyFactory.create({})
    ctx = _ctx({"a.jpg": 20.0}, {"a.jpg": 0.2})  # very tilted but unsure
    assert comp.compute_penalty("a.jpg", ctx) == 0.0


def test_below_angle_gate_is_zero():
    """< gate_deg is imperceptible AND within detector noise."""
    comp = TiltPenaltyFactory.create({})
    ctx = _ctx({"a.jpg": 2.0}, {"a.jpg": 0.9})
    assert comp.compute_penalty("a.jpg", ctx) == 0.0


def test_missing_scores_mean_zero_composite_regression():
    """A4: pipelines without score_tilt upstream are bit-identical to pre-099."""
    comp = TiltPenaltyFactory.create({})
    assert comp.compute_penalty("a.jpg", _ctx()) == 0.0


def test_linear_region():
    comp = TiltPenaltyFactory.create({})
    ctx = _ctx({"a.jpg": 6.0}, {"a.jpg": 0.9})  # excess 3 deg
    assert comp.compute_penalty("a.jpg", ctx) == pytest.approx(-0.02 * 3.0)


def test_cap_is_respected():
    comp = TiltPenaltyFactory.create({})
    ctx = _ctx({"a.jpg": 20.0}, {"a.jpg": 0.9})  # excess 17 deg -> 0.34 > cap
    assert comp.compute_penalty("a.jpg", ctx) == pytest.approx(-0.15)


def test_sign_independence():
    """A crooked photo is equally penalized clockwise or counter-clockwise."""
    comp = TiltPenaltyFactory.create({})
    cw = _ctx({"a.jpg": 8.0}, {"a.jpg": 0.9})
    ccw = _ctx({"a.jpg": -8.0}, {"a.jpg": 0.9})
    assert comp.compute_penalty("a.jpg", cw) == comp.compute_penalty("a.jpg", ccw)


def test_disabled_kills_the_term():
    comp = TiltPenaltyFactory.create({"enabled": False})
    ctx = _ctx({"a.jpg": 20.0}, {"a.jpg": 0.9})
    assert comp.compute_penalty("a.jpg", ctx) == 0.0


def test_stays_subordinate_to_occlusion():
    """The cap (-0.15) must be gentler than occlusion's floor (-0.5): tilt is a
    tie-breaker, not a disqualifier (blur > noise > exposure > tilt)."""
    comp = TiltPenaltyFactory.create({})
    ctx = _ctx({"a.jpg": 45.0}, {"a.jpg": 1.0})
    assert comp.compute_penalty("a.jpg", ctx) >= -0.15


def test_path_separator_tolerance():
    comp = TiltPenaltyFactory.create({})
    ctx = _ctx({"C:/x/a.jpg": 8.0}, {"C:/x/a.jpg": 0.9})
    assert comp.compute_penalty("C:\\x\\a.jpg", ctx) < 0.0


# --- fixability scaling (spec-101 option A) — needs real image dims -----------

def _real_ctx(tmp_path, w, h, angle, conf, person=None):
    import numpy as np
    from PIL import Image
    p = tmp_path / "w.jpg"
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(p, "JPEG")
    ctx = _ctx({str(p): angle}, {str(p): conf})
    ctx.persons = {str(p): person} if person else {}
    return str(p), ctx


def test_fixable_tilt_gets_small_fov_penalty(tmp_path):
    """Cleanly-straightenable landscape tilt -> penalty = fov_weight*(1-retained), small."""
    from sim_bench.quality_assessment.straighten import retained_area_fraction
    p, ctx = _real_ctx(tmp_path, 1500, 1000, 6.0, 0.9)
    comp = TiltPenaltyFactory.create({})
    retained = retained_area_fraction(1500, 1000, 6.0)
    # penalty is exactly the FOV cost of the crop (not the angle-based term)
    assert comp.compute_penalty(p, ctx) == pytest.approx(-0.4 * (1 - retained), abs=1e-6)
    assert -0.15 <= comp.compute_penalty(p, ctx) < 0.0


def test_unfixable_person_clip_gets_full_angle_penalty(tmp_path):
    """A prominent person the crop would clip -> not fixable -> full angle penalty."""
    person = {"person_detected": True, "bbox": {"x": 0.66, "y": 0.25, "w": 0.30, "h": 0.55}}
    p, ctx = _real_ctx(tmp_path, 1500, 1000, 6.0, 0.9, person)
    comp = TiltPenaltyFactory.create({})
    assert comp.compute_penalty(p, ctx) == pytest.approx(-min(0.02 * (6 - 3), 0.15))
