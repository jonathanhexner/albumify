"""ut for ScoreTiltStep (spec-099 Phase 1 / spec-100).

Offline tests use a fake scorer (fast, deterministic). One slow test runs the
real GeoCalib backend end-to-end (step -> cache -> penalty) on a tilted vs a
straight Budapest photo to prove the integration, when the album is present.
"""

from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import sim_bench.quality_assessment.tilt_geocalib as tilt_mod
from sim_bench.api.database.models import Base
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.scoring.tilt_penalty import TiltPenaltyFactory
from sim_bench.pipeline.steps.score_tilt import ScoreTiltStep

CALLS = {"n": 0}


class _FakeScorer:
    version = "fake-v1"

    def calc(self, inputs):
        CALLS["n"] += len(inputs.image_paths)
        return tilt_mod.TiltScoreResult(
            angles={p: 8.0 for p in inputs.image_paths},
            confidences={p: 0.9 for p in inputs.image_paths}, skipped=[])


@pytest.fixture
def handler():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return UniversalCacheHandler(sessionmaker(bind=engine)())


def _imgs(tmp_path, n=2):
    out = []
    for i in range(n):
        p = tmp_path / f"i{i}.jpg"
        Image.new("RGB", (16, 16), (i * 9, 40, 40)).save(p, "JPEG")
        out.append(str(p))
    return out


def test_angles_and_confidences_stored_then_cache_hit(tmp_path, handler, monkeypatch):
    monkeypatch.setattr(tilt_mod, "TiltScorer", _FakeScorer)
    CALLS["n"] = 0
    paths = _imgs(tmp_path)

    ctx1 = PipelineContext(image_paths=paths, cache_handler=handler)
    ScoreTiltStep().process(ctx1, {})
    assert CALLS["n"] == len(paths)
    assert ctx1.tilt_angles[paths[0]] == pytest.approx(8.0)
    assert ctx1.tilt_confidences[paths[0]] == pytest.approx(0.9)

    # fresh context, same handler: pure cache hit — scorer never called again
    ctx2 = PipelineContext(image_paths=paths, cache_handler=handler)
    ScoreTiltStep().process(ctx2, {})
    assert CALLS["n"] == len(paths)
    assert ctx2.tilt_angles[paths[1]] == pytest.approx(8.0)


def test_model_version_invalidates_cache(tmp_path, handler, monkeypatch):
    monkeypatch.setattr(tilt_mod, "TiltScorer", _FakeScorer)
    CALLS["n"] = 0
    paths = _imgs(tmp_path, 1)
    ScoreTiltStep().process(PipelineContext(image_paths=paths, cache_handler=handler), {})
    n1 = CALLS["n"]

    class _V2(_FakeScorer):
        version = "fake-v2"

    monkeypatch.setattr(tilt_mod, "TiltScorer", _V2)
    ScoreTiltStep().process(PipelineContext(image_paths=paths, cache_handler=handler), {})
    assert CALLS["n"] == n1 + 1  # new version -> recompute, not cache hit


_BUDAPEST = Path(r"D:\Budapest2025_Google")


@pytest.mark.slow
@pytest.mark.skipif(not _BUDAPEST.exists(), reason="Budapest album not on this machine")
def test_real_geocalib_end_to_end_penalizes_tilted(handler):
    """Full path with the real backend: a confidently-tilted photo is penalized,
    a level one is not. 123528 = +16.5 deg (conf), 112359 = ~0 deg."""
    tilted = str(_BUDAPEST / "20250822_123528.jpg")
    straight = str(_BUDAPEST / "20250822_112359.jpg")
    ctx = PipelineContext(image_paths=[tilted, straight], cache_handler=handler)
    ScoreTiltStep().process(ctx, {})

    assert abs(ctx.tilt_angles[tilted]) > 10.0
    assert abs(ctx.tilt_angles[straight]) < 3.0

    pen = TiltPenaltyFactory.create({})
    assert pen.compute_penalty(tilted, ctx) < 0.0     # tilted -> penalized
    assert pen.compute_penalty(straight, ctx) == 0.0  # level -> untouched
