"""ut for StraightenImagesStep (spec-101 option A) — TERMINAL: straightens winners."""

import numpy as np
from PIL import Image

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.straighten_images import StraightenImagesStep


def _img(tmp_path, name, w, h):
    p = tmp_path / name
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (h, w, 3), np.uint8)).save(p, "JPEG")
    return str(p)


def _ctx(winners, angles, confs, persons=None):
    c = PipelineContext(image_paths=list(winners))
    c.selected_images = list(winners)
    c.tilt_angles = angles
    c.tilt_confidences = confs
    c.persons = persons or {}
    return c


def test_landscape_winner_straightened_in_place(tmp_path):
    a = _img(tmp_path, "a.jpg", 1500, 1000)     # landscape, 6 deg -> straighten
    b = _img(tmp_path, "b.jpg", 1500, 1000)     # ~level -> passthrough
    ctx = _ctx([a, b], {a: 6.0, b: 1.0}, {a: 0.9, b: 0.9})

    StraightenImagesStep().process(ctx, {})

    assert ctx.selected_images[0] != a and ctx.selected_images[0].endswith(".jpg")  # derivative
    assert ctx.selected_images[1] == b                                              # level -> unchanged
    assert ctx.straightened_from.get(ctx.selected_images[0]) == a                   # provenance
    assert ctx.image_paths == [a, b]  # the scoring set is NOT touched (terminal)


def test_prominent_person_at_edge_declines(tmp_path):
    a = _img(tmp_path, "a.jpg", 1500, 1000)
    person = {"person_detected": True, "bbox": {"x": 0.66, "y": 0.25, "w": 0.30, "h": 0.55}}
    ctx = _ctx([a], {a: 5.0}, {a: 0.9}, {a: person})

    StraightenImagesStep().process(ctx, {})

    assert ctx.selected_images[0] == a          # declined -> winner kept as-shot
    assert not ctx.straightened_from


def test_disabled_is_passthrough(tmp_path):
    a = _img(tmp_path, "a.jpg", 1500, 1000)
    ctx = _ctx([a], {a: 8.0}, {a: 0.9})
    StraightenImagesStep().process(ctx, {"enabled": False})
    assert ctx.selected_images == [a] and not ctx.straightened_from


def test_derived_file_is_cached_second_run_reuses(tmp_path):
    import os
    a = _img(tmp_path, "a.jpg", 1500, 1000)
    ctx1 = _ctx([a], {a: 6.0}, {a: 0.9})
    StraightenImagesStep().process(ctx1, {})
    derived = ctx1.selected_images[0]
    mtime = os.path.getmtime(derived)
    ctx2 = _ctx([a], {a: 6.0}, {a: 0.9})
    StraightenImagesStep().process(ctx2, {})
    assert ctx2.selected_images[0] == derived and os.path.getmtime(derived) == mtime  # reused
