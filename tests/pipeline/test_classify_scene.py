"""ut for ClassifySceneStep (spec-094 scene tags) — offline, fake CLIP."""

import numpy as np
import pytest
import torch
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import clip as clip_mod
from sim_bench.api.database.models import Base
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.classify_scene import ClassifySceneStep

CALLS = {"img": 0}


class _FakeClip:
    def eval(self):
        return self

    def encode_text(self, toks):
        n = toks.shape[0]
        t = torch.eye(n, 8)  # n categories -> orthogonal 8-d vectors
        return t

    def encode_image(self, batch):
        CALLS["img"] += 1
        v = torch.zeros((batch.shape[0], 8))
        v[:, 1] = 1.0  # always most similar to category index 1
        return v


def _fake_load(name, device="cpu"):
    return _FakeClip(), (lambda img: torch.zeros(3, 224, 224))


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


def test_tags_full_ranking_and_cache_hit(tmp_path, handler, monkeypatch):
    monkeypatch.setattr(clip_mod, "load", _fake_load)
    CALLS["img"] = 0
    paths = _imgs(tmp_path)
    cats = ["cat A", "cat B", "cat C"]
    step = ClassifySceneStep()

    ctx1 = PipelineContext(image_paths=paths, cache_handler=handler)
    step.process(ctx1, {"categories": cats})
    assert CALLS["img"] == len(paths)
    tags = ctx1.scene_tags[paths[0]]
    assert [t["label"] for t in tags][0] == "cat B"          # index-1 wins by construction
    assert len(tags) == 3                                     # FULL ranked list stored
    assert abs(sum(t["score"] for t in tags) - 1.0) < 1e-5    # softmax

    # second run, fresh context: pure cache hit, no image encodes
    ctx2 = PipelineContext(image_paths=paths, cache_handler=handler)
    ClassifySceneStep().process(ctx2, {"categories": cats})
    assert CALLS["img"] == len(paths)                         # unchanged
    assert ctx2.scene_tags[paths[0]][0]["label"] == "cat B"


def test_category_change_invalidates_cache(tmp_path, handler, monkeypatch):
    monkeypatch.setattr(clip_mod, "load", _fake_load)
    CALLS["img"] = 0
    paths = _imgs(tmp_path, 1)
    ClassifySceneStep().process(PipelineContext(image_paths=paths, cache_handler=handler),
                                {"categories": ["a", "b"]})
    n1 = CALLS["img"]
    ClassifySceneStep().process(PipelineContext(image_paths=paths, cache_handler=handler),
                                {"categories": ["a", "b", "c"]})  # different set -> recompute
    assert CALLS["img"] == n1 + 1


def test_in_studio_geo_family():
    from app.image_studio import engine
    m = {x["key"]: x for x in engine.available_methods()}
    assert "scene_tag" in m and m["scene_tag"]["category"] == engine.CATEGORY_GEO
