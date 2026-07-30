"""spec-094 Slice 2 — Image Analysis Studio engine (pure, no UI).

Covers discover_images, the AnalysisColumn mappers, and run_methods. No model
load: the backing steps are monkeypatched with fakes that populate the context.
"""

from datetime import datetime

import pytest
from PIL import Image

from app.image_studio import engine
from geo_cluster.types import GeoMetadata


# --------------------------------------------------------------------------- #
# ut_DiscoverImages
# --------------------------------------------------------------------------- #
def test_discover_filters_sorts_and_caps(tmp_path):
    for name in ["b.jpg", "a.png", "c.heic", "notes.txt", "d.jpeg"]:
        (tmp_path / name).write_bytes(b"x")
    got = [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in engine.discover_images(str(tmp_path))]
    assert got == ["a.png", "b.jpg", "c.heic", "d.jpeg"]  # sorted, .txt excluded


def test_discover_cap(tmp_path):
    for i in range(5):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")
    assert len(engine.discover_images(str(tmp_path), limit=2)) == 2


def test_discover_missing_or_empty(tmp_path):
    assert engine.discover_images(str(tmp_path / "nope")) == []
    assert engine.discover_images("") == []
    assert engine.discover_images(str(tmp_path)) == []  # empty dir


# --------------------------------------------------------------------------- #
# ut_AnalysisColumn mappers
# --------------------------------------------------------------------------- #
def test_map_streetclip_kind_and_sort():
    ctx = engine.PipelineContext(image_paths=["a.jpg"])
    ctx.geo_clip_predictions = {"a.jpg": [{"label": "Budapest, Hungary", "score": 0.87},
                                          {"label": "Vienna, Austria", "score": 0.07}]}
    col = engine._map_streetclip(ctx)["a.jpg"]
    assert col.kind == engine.LABEL_CONF
    assert col.sort_value == pytest.approx(0.87)        # sorts by top-1 confidence
    assert "Budapest" in col.display and "0.87" in col.display
    assert len(col.topk) == 2                            # full top-k retained


def test_map_geoclip_uses_place_and_prob():
    ctx = engine.PipelineContext(image_paths=["a.jpg"])
    ctx.geo_coord_predictions = {"a.jpg": [{"lat": 47.5, "lon": 19.0, "prob": 0.4, "place": "Budapest, HU"}]}
    col = engine._map_geoclip(ctx)["a.jpg"]
    assert col.sort_value == pytest.approx(0.4)
    assert "Budapest, HU" in col.display


def test_map_blip_is_text_unsortable():
    ctx = engine.PipelineContext(image_paths=["a.jpg"])
    ctx.image_captions = {"a.jpg": "a bridge over a river"}
    col = engine._map_blip(ctx)["a.jpg"]
    assert col.kind == engine.TEXT and col.sort_value is None
    assert col.display == "a bridge over a river"


def test_map_exif_coord_display():
    ctx = engine.PipelineContext(image_paths=["a.jpg"])
    ctx.geo_metadata = {"a.jpg": GeoMetadata("a.jpg", datetime(2025, 8, 22, 12, 5), 47.5, 19.0)}
    col = engine._map_exif(ctx)["a.jpg"]
    assert col.kind == engine.COORD
    assert "47.5000" in col.display and "2025-08-22 12:05" in col.display


def test_map_streetclip_empty_predictions():
    ctx = engine.PipelineContext(image_paths=["a.jpg"])
    ctx.geo_clip_predictions = {"a.jpg": []}
    col = engine._map_streetclip(ctx)["a.jpg"]
    assert col.sort_value is None and col.display == "-"


# --------------------------------------------------------------------------- #
# ut_RunMethods
# --------------------------------------------------------------------------- #
def _fake_step(setter):
    """Build a fake step class whose process() runs ``setter(ctx)``."""
    class Fake:
        calls = 0

        def process(self, ctx, config):
            type(self).calls += 1
            setter(ctx)
    return Fake


def test_run_methods_only_selected(monkeypatch):
    paths = ["a.jpg", "b.jpg"]

    def set_sclip(ctx):
        ctx.geo_clip_predictions = {p: [{"label": "X", "score": 0.5}] for p in paths}

    def set_blip(ctx):
        ctx.image_captions = {p: "cap" for p in paths}

    monkeypatch.setattr(engine, "InferGeoClipStep", _fake_step(set_sclip))
    monkeypatch.setattr(engine, "CaptionImagesStep", _fake_step(set_blip))

    cols = engine.run_methods(paths, ["streetclip"])  # blip NOT selected
    assert set(cols) == {"a.jpg", "b.jpg"}
    assert set(cols["a.jpg"]) == {"streetclip"}        # only the selected method
    assert cols["a.jpg"]["streetclip"].sort_value == pytest.approx(0.5)


def test_run_methods_shared_step_runs_once(monkeypatch):
    paths = ["a.jpg"]

    def set_iqa(ctx):
        ctx.iqa_scores = {"a.jpg": 0.7}
        ctx.sharpness_scores = {"a.jpg": 0.9}

    Fake = _fake_step(set_iqa)
    monkeypatch.setattr(engine, "ScoreIQAStep", Fake)

    cols = engine.run_methods(paths, ["iqa", "sharpness"])  # both back onto score_iqa
    assert Fake.calls == 1                                   # step ran once, not twice
    assert cols["a.jpg"]["iqa"].sort_value == pytest.approx(0.7)
    assert cols["a.jpg"]["sharpness"].sort_value == pytest.approx(0.9)


def test_run_methods_failing_method_is_skipped(monkeypatch):
    paths = ["a.jpg"]

    class Boom:
        def process(self, ctx, config):
            raise RuntimeError("model exploded")

    def set_blip(ctx):
        ctx.image_captions = {"a.jpg": "cap"}

    monkeypatch.setattr(engine, "InferGeoClipStep", Boom)
    monkeypatch.setattr(engine, "CaptionImagesStep", _fake_step(set_blip))

    cols = engine.run_methods(paths, ["streetclip", "blip"])  # streetclip raises
    assert "streetclip" not in cols["a.jpg"]                  # skipped, not fatal
    assert cols["a.jpg"]["blip"].display == "cap"             # other method still ran


def test_available_methods_and_categories():
    keys = {m["key"] for m in engine.available_methods()}
    assert {"exif", "streetclip", "geoclip", "blip", "iqa"} <= keys
    assert {"maniqa", "niqe", "brisque"} <= keys        # spec-093 pyiqa metrics registered
    assert engine.categories()[0] == engine.CATEGORY_GEO


def test_merge_configs_unions_lists():
    m = engine._merge_configs([{"methods": ["a"], "device": "cpu"}, {"methods": ["b"]}])
    assert m["methods"] == ["a", "b"] and m["device"] == "cpu"


def test_run_methods_pyiqa_merges_into_one_run(monkeypatch):
    """maniqa + niqe -> a SINGLE ScoreQualityStep run scoring both (spec-093 wiring)."""
    paths = ["a.jpg"]
    seen = {"cfg": None, "calls": 0}

    class Fake:
        def process(self, ctx, config):
            seen["cfg"] = config
            seen["calls"] += 1
            ctx.method_scores = {"a.jpg": {mth: 0.5 for mth in config.get("methods", [])}}

    monkeypatch.setattr(engine, "ScoreQualityStep", Fake)
    cols = engine.run_methods(paths, ["maniqa", "niqe"])

    assert seen["calls"] == 1                                    # one run, not two
    assert set(seen["cfg"]["methods"]) == {"maniqa", "niqe"}     # merged method list
    assert cols["a.jpg"]["maniqa"].sort_value == pytest.approx(0.5)
    assert cols["a.jpg"]["niqe"].kind == engine.NUMERIC
