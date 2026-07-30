"""ut for app/image_studio/run_folder.py (spec-094 Slice 5) — save/list/load."""

import os

import pytest

from app.image_studio.engine import AnalysisColumn, CATEGORY_QUALITY, CATEGORY_GEO, NUMERIC, LABEL_CONF
from app.image_studio import run_folder


def _columns(folder):
    a = os.path.join(folder, "a.jpg")
    b = os.path.join(folder, "b.jpg")
    return {
        a: {"brisque": AnalysisColumn("brisque", CATEGORY_QUALITY, NUMERIC, -19.7, "-19.700", []),
            "streetclip": AnalysisColumn("streetclip", CATEGORY_GEO, LABEL_CONF, 0.87,
                                         "Budapest (0.87)", [{"label": "Budapest", "score": 0.87}])},
        b: {"brisque": AnalysisColumn("brisque", CATEGORY_QUALITY, NUMERIC, -30.2, "-30.200", [])},
    }


def test_save_list_load_roundtrip(tmp_path):
    folder = str(tmp_path)
    cols = _columns(folder)
    paths = list(cols)
    rid = run_folder.make_run_id("2026-07-03_143000", ["brisque", "streetclip"])
    run_folder.save(folder, rid, ["brisque", "streetclip"], paths, cols, "2026-07-03_143000")

    # files exist
    d = os.path.join(folder, run_folder.RUNS_DIRNAME, rid)
    for fn in ("run.json", "columns.json", "results.csv"):
        assert os.path.isfile(os.path.join(d, fn))

    # list
    runs = run_folder.list_runs(folder)
    assert len(runs) == 1 and runs[0]["run_id"] == rid
    assert runs[0]["n_images"] == 2 and runs[0]["methods"] == ["brisque", "streetclip"]

    # load — columns reconstructed with full attrs
    loaded = run_folder.load(folder, rid)
    assert set(loaded["paths"]) == set(paths)
    a = os.path.join(folder, "a.jpg")
    ac = loaded["columns"][a]["brisque"]
    assert ac.kind == NUMERIC and ac.sort_value == pytest.approx(-19.7) and ac.display == "-19.700"
    sc = loaded["columns"][a]["streetclip"]
    assert sc.topk == [{"label": "Budapest", "score": 0.87}]


def test_list_newest_first(tmp_path):
    folder = str(tmp_path)
    cols = _columns(folder)
    paths = list(cols)
    for ts in ["2026-07-01_100000", "2026-07-03_120000", "2026-07-02_090000"]:
        rid = run_folder.make_run_id(ts, ["brisque"])
        run_folder.save(folder, rid, ["brisque"], paths, cols, ts)
    order = [m["created_ts"] for m in run_folder.list_runs(folder)]
    assert order == ["2026-07-03_120000", "2026-07-02_090000", "2026-07-01_100000"]


def test_malformed_run_dir_is_skipped(tmp_path):
    folder = str(tmp_path)
    cols = _columns(folder)
    run_folder.save(folder, "good", ["brisque"], list(cols), cols, "2026-07-03_120000")
    # a junk dir with a broken run.json
    bad = os.path.join(folder, run_folder.RUNS_DIRNAME, "broken")
    os.makedirs(bad)
    with open(os.path.join(bad, "run.json"), "w") as f:
        f.write("{ not json")
    runs = run_folder.list_runs(folder)
    assert [m["run_id"] for m in runs] == ["good"]  # broken skipped, no raise


def test_empty_and_missing_folder(tmp_path):
    assert run_folder.list_runs(str(tmp_path)) == []          # no .studio_runs yet
    assert run_folder.list_runs(str(tmp_path / "nope")) == []  # missing folder
    assert run_folder.load(str(tmp_path), "nonexistent") is None
