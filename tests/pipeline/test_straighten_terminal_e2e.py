"""Real-data E2E for the terminal straighten + fixability penalty (spec-101 A).

Closes the REVIEW.md §5 blocker: exercises the actual step on real Budapest
photos — a straightenable landscape and a decline-on-area portrait — and proves
the straightened winner is really leveled (GeoCalib round-trip). Marked slow
(loads GeoCalib) and skipped when the album is absent.
"""

from pathlib import Path

import pytest

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.scoring.tilt_penalty import TiltPenaltyFactory
from sim_bench.pipeline.steps.straighten_images import StraightenImagesStep

pytestmark = [pytest.mark.slow, pytest.mark.needs_data]

BUD = Path(r"D:/Budapest2025_Google")
LANDSCAPE = "20250822_194201"   # roll -4.3, confident, 4:3 -> straightens
PORTRAIT = "20250822_123528"    # roll +16.5 -> ~39% area -> declines


def _path(stem):
    return str(next(BUD.rglob(stem + ".jp*")))


def _roll(rgb_path):
    import torch
    from geocalib import GeoCalib
    from geocalib.utils import numpy_image_to_torch
    import numpy as np
    from PIL import Image, ImageOps
    m = GeoCalib().to("cpu").eval()
    rgb = np.array(ImageOps.exif_transpose(Image.open(rgb_path)).convert("RGB"))
    with torch.no_grad():
        r = m.calibrate(numpy_image_to_torch(rgb).to("cpu"))
    return -1.0 * float(torch.rad2deg(r["gravity"].roll))


@pytest.mark.skipif(not BUD.exists(), reason="Budapest album not on this machine")
def test_terminal_straighten_and_penalty_on_real_winners():
    land, port = _path(LANDSCAPE), _path(PORTRAIT)
    ctx = PipelineContext(image_paths=[land, port])
    ctx.selected_images = [land, port]
    ctx.tilt_angles = {land: -4.3, port: 16.5}
    ctx.tilt_confidences = {land: 0.9, port: 0.9}
    ctx.persons = {}

    StraightenImagesStep().process(ctx, {})

    # landscape winner straightened (repointed to a derivative); portrait declined
    assert ctx.selected_images[0] != land and ctx.selected_images[0] in ctx.straightened_from
    assert ctx.selected_images[1] == port
    # the straightened output is actually level
    assert abs(_roll(ctx.selected_images[0])) < 1.5

    # penalty: landscape is cleanly FIXABLE -> small FOV cost; portrait UNFIXABLE (area) -> full angle
    pen = TiltPenaltyFactory.create({})
    p_land = pen.compute_penalty(land, ctx)
    p_port = pen.compute_penalty(port, ctx)
    assert -0.15 <= p_land < 0.0
    assert p_port == pytest.approx(-min(0.02 * (16.5 - 3), 0.15))  # capped angle penalty
