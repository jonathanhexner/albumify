"""spec-110 T2.1 - JobStore: the worker's storage layout (a job is a directory)."""
from __future__ import annotations

import json

import pytest

from sim_bench.serving.job_store import JobStore


class ut_JobPaths:
    def test_layout_is_rooted_at_the_job_id(self, tmp_path):
        p = JobStore(tmp_path).paths("trip42")
        assert p.root == tmp_path / "trip42"
        assert p.input_dir == p.root / "input"
        assert p.display_dir == p.root / "display"
        assert p.album_path == p.root / "album.json"
        assert p.run_path == p.root / "run.json"

    def test_building_paths_creates_nothing(self, tmp_path):
        JobStore(tmp_path).paths("trip42")
        assert not (tmp_path / "trip42").exists()

    @pytest.mark.parametrize("bad", ["", "../evil", "a/b", ".hidden", "with space", "x" * 65, "_lead"])
    def test_a_job_id_that_could_escape_the_root_is_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError, match="invalid job_id"):
            JobStore(tmp_path).paths(bad)


class ut_JobStoreInputs:
    def test_stage_then_list_is_sorted_by_name(self, tmp_path):
        store = JobStore(tmp_path)
        store.stage_inputs("j1", [("b.jpg", b"B"), ("a.jpg", b"A")])
        assert [p.name for p in store.input_images("j1")] == ["a.jpg", "b.jpg"]

    def test_restaging_replaces_the_previous_input_set(self, tmp_path):
        store = JobStore(tmp_path)
        store.stage_inputs("j1", [("a.jpg", b"A"), ("b.jpg", b"B")])
        store.stage_inputs("j1", [("c.jpg", b"C")])
        assert [p.name for p in store.input_images("j1")] == ["c.jpg"]

    def test_non_image_files_are_not_listed_as_inputs(self, tmp_path):
        store = JobStore(tmp_path)
        store.stage_inputs("j1", [("a.jpg", b"A")])
        (store.paths("j1").input_dir / "notes.txt").write_text("x", encoding="utf-8")
        assert [p.name for p in store.input_images("j1")] == ["a.jpg"]

    def test_image_name_cannot_escape_the_job_dir(self, tmp_path):
        store = JobStore(tmp_path)
        store.stage_inputs("j1", [("../../escaped.jpg", b"A")])   # path part is stripped
        assert [p.name for p in store.input_images("j1")] == ["escaped.jpg"]
        assert not (tmp_path.parent / "escaped.jpg").exists()

    def test_a_job_with_nothing_staged_lists_no_inputs(self, tmp_path):
        assert JobStore(tmp_path).input_images("ghost") == []


class ut_JobStoreResults:
    def test_album_round_trips(self, tmp_path):
        store = JobStore(tmp_path)
        path = store.write_album("j1", {"job_id": "j1", "picks": []})
        assert json.loads(path.read_text(encoding="utf-8"))["job_id"] == "j1"
        assert store.read_album("j1") == {"job_id": "j1", "picks": []}

    def test_reading_a_missing_album_is_none_not_an_error(self, tmp_path):
        assert JobStore(tmp_path).read_album("j1") is None

    def test_displays_are_copies_and_replace_the_previous_set(self, tmp_path):
        store = JobStore(tmp_path)
        staged = store.stage_inputs("j1", [("a.jpg", b"A"), ("b.jpg", b"B")])
        store.write_displays("j1", staged)
        assert sorted(p.name for p in store.paths("j1").display_dir.iterdir()) == ["a.jpg", "b.jpg"]
        store.write_displays("j1", staged[:1])
        assert [p.name for p in store.paths("j1").display_dir.iterdir()] == ["a.jpg"]

    def test_a_missing_display_source_is_skipped_not_fatal(self, tmp_path):
        store = JobStore(tmp_path)
        staged = store.stage_inputs("j1", [("a.jpg", b"A")])
        out = store.write_displays("j1", staged + [tmp_path / "gone.jpg"])
        assert [p.name for p in out] == ["a.jpg"]

    def test_run_record_is_written_as_json(self, tmp_path):
        store = JobStore(tmp_path)
        path = store.write_run_record("j1", {"ok": True, "device": "cuda"})
        assert json.loads(path.read_text(encoding="utf-8"))["device"] == "cuda"


class ut_JobStoreResults:
    """spec-112: results bundle + face crops beside the album."""

    def test_round_trip_and_rewrite(self, tmp_path):
        import numpy as np
        store = JobStore(tmp_path)
        store.write_results("j1", {"photos": [1]}, np.ones((2, 3)), [4, 9], {4: b"jpg4", 9: b"jpg9"})
        p = store.paths("j1")
        assert store.read_results("j1") == {"photos": [1]}
        assert np.load(p.results_dir / "embedding_face_ids.npy").tolist() == [4, 9]
        assert sorted(f.name for f in p.faces_dir.iterdir()) == ["4.jpg", "9.jpg"]
        store.write_results("j1", {"photos": []}, crops={1: b"x"})
        assert sorted(f.name for f in p.faces_dir.iterdir()) == ["1.jpg"]
        assert not (p.results_dir / "embeddings.npy").exists()

    def test_missing_bundle_reads_none(self, tmp_path):
        assert JobStore(tmp_path).read_results("nope") is None
