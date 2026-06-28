"""test_score_quality_step (spec-093) - multi-method dispatch + per-method cache.

Uses a deterministic fake model registered in MODEL_REGISTRY (offline, no weight
download). Verifies: dispatch writes context.method_scores; a second run on the
same cache is a pure hit (score_image not called again); the no-cache path still
computes.
"""

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import Base
from sim_bench.image_quality_models import model_factory
from sim_bench.image_quality_models.base_model import BaseQualityModel
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.score_quality import ScoreQualityStep


class _CountingModel(BaseQualityModel):
    calls = 0

    def __init__(self, device="cpu"):
        super().__init__(name="ut-fake", device=device)

    def score_image(self, image_path):
        _CountingModel.calls += 1
        return 0.5

    def raw_score(self, image_path):
        return 0.5

    @classmethod
    def from_config(cls, config):
        return cls(device=config.get("device", "cpu"))


@pytest.fixture(autouse=True)
def register_fake():
    model_factory.MODEL_REGISTRY["ut_fake"] = _CountingModel
    _CountingModel.calls = 0
    yield
    model_factory.MODEL_REGISTRY.pop("ut_fake", None)


@pytest.fixture
def handler():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return UniversalCacheHandler(sessionmaker(bind=engine)())


def _imgs(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(np.full((16, 16, 3), i * 5, dtype=np.uint8)).save(p, "JPEG")
        paths.append(str(p))
    return paths


def test_dispatch_writes_method_scores(tmp_path):
    paths = _imgs(tmp_path)
    ctx = PipelineContext(image_paths=paths)  # no cache handler
    ScoreQualityStep().process(ctx, {"methods": ["ut_fake"]})

    assert set(ctx.method_scores) == set(paths)
    for path in paths:
        assert ctx.method_scores[path]["ut_fake"] == pytest.approx(0.5)
    assert _CountingModel.calls == len(paths)


def test_second_run_hits_cache(tmp_path, handler):
    paths = _imgs(tmp_path)
    step = ScoreQualityStep()

    step.process(PipelineContext(image_paths=paths, cache_handler=handler), {"methods": ["ut_fake"]})
    assert _CountingModel.calls == len(paths)

    # fresh context, same cache -> no recompute
    ctx2 = PipelineContext(image_paths=paths, cache_handler=handler)
    step.process(ctx2, {"methods": ["ut_fake"]})
    assert _CountingModel.calls == len(paths)  # unchanged
    for path in paths:
        assert ctx2.method_scores[path]["ut_fake"] == pytest.approx(0.5)


def test_unknown_method_is_skipped_not_fatal(tmp_path):
    paths = _imgs(tmp_path)
    ctx = PipelineContext(image_paths=paths)
    ScoreQualityStep().process(ctx, {"methods": ["ut_fake", "does_not_exist"]})
    # fake still scored; unknown method simply absent
    for path in paths:
        assert ctx.method_scores[path] == {"ut_fake": pytest.approx(0.5)}


def test_empty_album_no_error():
    ctx = PipelineContext(image_paths=[])
    ScoreQualityStep().process(ctx, {"methods": ["ut_fake"]})
    assert ctx.method_scores == {}
