"""ut for ScoreOcclusionStep (spec-097 Stage 1) — offline, fake scorer."""

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import sim_bench.occlusion_bench.scorer as scorer_mod
from sim_bench.api.database.models import Base
from sim_bench.occlusion_bench.scorer import OcclusionInputs, OcclusionResult
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.score_occlusion import ScoreOcclusionStep

CALLS = {"n": 0}


class _FakeScorer:
    version = "fake-v1"

    def __init__(self, artifact_path=None):
        pass

    def calc(self, inputs: OcclusionInputs) -> OcclusionResult:
        CALLS["n"] += len(inputs.image_paths)
        return OcclusionResult(
            scores={p: 0.9 for p in inputs.image_paths},
            tiles={p: [0.9] + [0.1] * 8 for p in inputs.image_paths})


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


def test_scores_and_tiles_stored_then_cache_hit(tmp_path, handler, monkeypatch):
    monkeypatch.setattr(scorer_mod, "OcclusionScorer", _FakeScorer)
    CALLS["n"] = 0
    paths = _imgs(tmp_path)

    ctx1 = PipelineContext(image_paths=paths, cache_handler=handler)
    ScoreOcclusionStep().process(ctx1, {})
    assert CALLS["n"] == len(paths)
    assert ctx1.occlusion_scores[paths[0]] == pytest.approx(0.9)
    assert len(ctx1.occlusion_tiles[paths[0]]) == 9

    # fresh context, same handler: pure cache hit — the scorer is never called
    ctx2 = PipelineContext(image_paths=paths, cache_handler=handler)
    ScoreOcclusionStep().process(ctx2, {})
    assert CALLS["n"] == len(paths)
    assert ctx2.occlusion_scores[paths[1]] == pytest.approx(0.9)


def test_artifact_version_invalidates_cache(tmp_path, handler, monkeypatch):
    monkeypatch.setattr(scorer_mod, "OcclusionScorer", _FakeScorer)
    CALLS["n"] = 0
    paths = _imgs(tmp_path, 1)
    ScoreOcclusionStep().process(
        PipelineContext(image_paths=paths, cache_handler=handler), {})
    n1 = CALLS["n"]

    class _V2(_FakeScorer):
        version = "fake-v2"

    monkeypatch.setattr(scorer_mod, "OcclusionScorer", _V2)
    ScoreOcclusionStep().process(
        PipelineContext(image_paths=paths, cache_handler=handler), {})
    assert CALLS["n"] == n1 + 1  # new model version -> recompute, not cache hit
