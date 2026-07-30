"""spec-094 Slice 1 — geo/vision steps persist through universal_cache.

Verifies the storage refactor: each step writes its output to the universal_cache
DB via the BaseStep cache hooks, a second run is a pure cache hit (the helper /
`_process_uncached` is not called again), and the full top-k round-trips.

No network / model download: EXIF uses real (EXIF-less) tmp images; the vision
steps monkeypatch their helper's ``calc`` with canned predictions.
"""

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import Base
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext
import sim_bench.pipeline.steps.extract_geo_metadata as exif_mod
import sim_bench.pipeline.steps.infer_geo_clip as sclip_mod
import sim_bench.pipeline.steps.infer_geo_coords as gclip_mod
import sim_bench.pipeline.steps.caption_images as blip_mod
from sim_bench.pipeline.steps.extract_geo_metadata import ExtractGeoMetadataStep
from sim_bench.pipeline.steps.infer_geo_clip import InferGeoClipStep
from sim_bench.pipeline.steps.infer_geo_coords import InferGeoCoordsStep
from sim_bench.pipeline.steps.caption_images import CaptionImagesStep


@pytest.fixture
def handler():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    return UniversalCacheHandler(session)


def _imgs(tmp_path, n=2):
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(p, "JPEG")
        paths.append(str(p))
    return paths


def _ctx(paths, handler):
    return PipelineContext(image_paths=paths, cache_handler=handler)


def _count_calls(monkeypatch, module, attr, return_value):
    """Patch ``module.attr`` (a Locator/Captioner class) so its instance's
    ``calc`` returns ``return_value`` and counts invocations."""
    calls = {"n": 0}

    class Fake:
        def __init__(self, *a, **k):
            pass

        def calc(self, inputs):
            calls["n"] += 1
            return return_value

    monkeypatch.setattr(module, attr, Fake)
    return calls


# --------------------------------------------------------------------------- #
# EXIF — real round-trip (no model), proves serialize/deserialize + cache hit
# --------------------------------------------------------------------------- #
def test_exif_step_caches_and_roundtrips(tmp_path, handler):
    paths = _imgs(tmp_path)
    step = ExtractGeoMetadataStep()

    step.process(_ctx(paths, handler), {})  # first run: compute + store

    # second run on a fresh context: must be a pure cache hit
    seen = {"items": None}
    orig = step._process_uncached

    def spy(items, context, config):
        seen["items"] = list(items)
        return orig(items, context, config)

    step._process_uncached = spy
    ctx2 = _ctx(paths, handler)
    step.process(ctx2, {})

    assert seen["items"] is None  # _process_uncached never called -> all cached
    assert set(ctx2.geo_metadata) == set(paths)
    # EXIF-less tmp images -> all fields None, but objects round-trip cleanly
    for m in ctx2.geo_metadata.values():
        assert m.lat is None and m.lon is None and m.timestamp is None


# --------------------------------------------------------------------------- #
# StreetCLIP — full top-k persisted; 2nd run hits cache (helper not called)
# --------------------------------------------------------------------------- #
def test_streetclip_step_persists_full_topk(tmp_path, handler, monkeypatch):
    paths = _imgs(tmp_path)
    topk = [{"label": "Budapest, Hungary", "score": 0.87},
            {"label": "Vienna, Austria", "score": 0.07},
            {"label": "Prague, Czechia", "score": 0.03}]
    canned = type("R", (), {"predictions": {p: topk for p in paths}})()
    calls = _count_calls(
        monkeypatch,
        sclip_mod, "StreetCLIPLocator", canned)

    step = InferGeoClipStep()
    ctx1 = _ctx(paths, handler)
    step.process(ctx1, {})
    assert calls["n"] == 1
    assert ctx1.geo_clip_predictions[paths[0]] == topk          # full top-k, 3 entries

    ctx2 = _ctx(paths, handler)
    step.process(ctx2, {})
    assert calls["n"] == 1                                       # cache hit, no recompute
    assert ctx2.geo_clip_predictions[paths[0]] == topk          # round-tripped from DB


# --------------------------------------------------------------------------- #
# GeoCLIP — full top-k (coords + prob + place) persisted; 2nd run cache hit
# --------------------------------------------------------------------------- #
def test_geoclip_step_persists_full_topk(tmp_path, handler, monkeypatch):
    paths = _imgs(tmp_path)
    topk = [{"lat": 47.5, "lon": 19.05, "prob": 0.41, "place": "Budapest, HU"},
            {"lat": 48.2, "lon": 16.37, "prob": 0.12, "place": "Vienna, AT"},
            {"lat": 50.08, "lon": 14.43, "prob": 0.05, "place": "Prague, CZ"}]
    canned = type("R", (), {"predictions": {p: topk for p in paths}})()
    calls = _count_calls(
        monkeypatch,
        gclip_mod, "GeoCLIPLocator", canned)

    step = InferGeoCoordsStep()
    ctx1 = _ctx(paths, handler)
    step.process(ctx1, {})
    ctx2 = _ctx(paths, handler)
    step.process(ctx2, {})

    assert calls["n"] == 1
    assert ctx2.geo_coord_predictions[paths[0]] == topk


# --------------------------------------------------------------------------- #
# BLIP — caption persisted; 2nd run cache hit
# --------------------------------------------------------------------------- #
def test_blip_step_caches_caption(tmp_path, handler, monkeypatch):
    paths = _imgs(tmp_path)
    caps = {p: "a group of people on a bridge" for p in paths}
    canned = type("R", (), {"captions": caps})()
    calls = _count_calls(
        monkeypatch,
        blip_mod, "BlipCaptioner", canned)

    step = CaptionImagesStep()
    ctx1 = _ctx(paths, handler)
    step.process(ctx1, {})
    ctx2 = _ctx(paths, handler)
    step.process(ctx2, {})

    assert calls["n"] == 1
    assert ctx2.image_captions[paths[0]] == "a group of people on a bridge"


# --------------------------------------------------------------------------- #
# No cache handler -> step still works (computes, no persistence)
# --------------------------------------------------------------------------- #
def test_exif_step_works_without_cache_handler(tmp_path):
    paths = _imgs(tmp_path)
    ctx = PipelineContext(image_paths=paths)  # cache_handler=None
    ExtractGeoMetadataStep().process(ctx, {})
    assert set(ctx.geo_metadata) == set(paths)
