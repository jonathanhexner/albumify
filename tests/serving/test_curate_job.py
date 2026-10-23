"""spec-110 T2.1/AC3/AC4 - JobCurator: storage in, normalized album.json out, idempotent.

The real pipeline is replaced by an injected runner (what is under test is the orchestration and the
storage contract, not the models). The runner is handed the job's input paths; spec-112 A2 makes those
the shared input store's, identical across jobs, which is what lets the feature cache carry over.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim_bench.serving.curate_job import CurateJobInputs, JobCurator, stable_album
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder


def _context(images):
    """A finished-looking PipelineContext: 2 scenes, 2 picks, 1 near-dup, 1 screenshot set aside."""
    img = {Path(p).name: str(p) for p in images}
    return SimpleNamespace(
        source_directory=None,
        selected_images=[img["a.jpg"], img["c.jpg"]],
        scene_cluster_labels={img["a.jpg"]: 0, img["b.jpg"]: 0, img["c.jpg"]: 1},
        set_aside=[{"image": img["s.jpg"], "reason": "screenshot (filename)"}],
        composite_scores={img["a.jpg"]: 0.8, img["b.jpg"]: 0.5, img["c.jpg"]: 0.7},
        iqa_scores={img["a.jpg"]: 0.6}, ava_scores={img["a.jpg"]: 5.1},
        occlusion_scores={img["a.jpg"]: 0.01}, tilt_angles={img["a.jpg"]: 1.2},
        image_captions={img["a.jpg"]: "a bridge at dusk"},
        straightened_from={}, people_clusters={1: ["f1", "f2"]},
        album_theme={"label": "city trip"},
    )


def _ok_result():
    return SimpleNamespace(success=True, total_duration_ms=1234.0, failed_step=None,
                           error_message=None,
                           step_results=[SimpleNamespace(step_name="discover_images",
                                                         duration_ms=12.0)])


def fake_runner(images, plan):
    return _context(images), _ok_result()


def failing_runner(images, plan):
    return _context(images), SimpleNamespace(success=False, failed_step="score_ava",
                                           error_message="boom", total_duration_ms=9.0,
                                           step_results=[])


NAMES = ("a.jpg", "b.jpg", "c.jpg", "s.jpg")


def _stage_shared(s: JobStore, job_id: str) -> list:
    """Stage NAMES through the shared input store (fixed fake hashes) and point ``job_id`` at them."""
    keys = [f"{i:016x}/{n}" for i, n in enumerate(NAMES)]
    for key in keys:
        if not s.inputs.has(key):
            s.inputs.put_bytes(key, b"jpeg-bytes")
    s.write_input_manifest(job_id, keys)
    return keys


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path)
    _stage_shared(s, "trip1")
    return s


def _curator(store, runner=fake_runner, clock=None):
    # minimal_pipeline: the plan is not what is under test here, but it must be a real one
    return JobCurator(store, spec_builder=SpecBuilder(), runner=runner,
                      clock=clock or (lambda: "2026-10-09T00:00:00+00:00"))


class ut_JobCurator:
    def test_album_json_is_written_with_picks_dropped_and_counts(self, store):
        res = _curator(store).calc(CurateJobInputs(job_id="trip1", pipeline="minimal"))
        assert res.ok and res.n_inputs == 4
        album = store.read_album("trip1")
        assert album == res.album
        assert [p["id"] for p in album["picks"]] == ["a", "c"]
        assert {d["id"]: d["reason"] for d in album["dropped"]} == {"b": "near_dup", "s": "set_aside"}
        assert album["counts"] == {"uploaded": 4, "kept": 2, "people": 1, "scenes": 2, "set_aside": 1}
        assert album["theme"] == {"label": "city trip"}

    def test_pick_carries_its_score_breakdown_and_caption(self, store):
        res = _curator(store).calc(CurateJobInputs(job_id="trip1", pipeline="minimal"))
        pick = res.album["picks"][0]
        assert pick["score"]["composite"] == 0.8 and pick["score"]["aesthetic"] == 5.1
        assert pick["caption"] == "a bridge at dusk"
        assert pick["beat"] == ["b"]

    def test_results_bundle_lists_every_staged_photo(self, store):
        """spec-112 TA1.2: the bundle covers all inputs, including ones absent from album.json."""
        res = _curator(store).calc(CurateJobInputs(job_id="trip1", pipeline="minimal"))
        bundle = store.read_results("trip1")
        assert res.bundle_path == str(store.paths("trip1").bundle_path)
        assert [p["photo_id"] for p in bundle["photos"]] == ["a", "b", "c", "s"]
        assert all(p["in_album"] for p in bundle["photos"])
        assert bundle["faces"] == [] and store.paths("trip1").faces_dir.is_dir()

    def test_a_failed_run_writes_no_bundle(self, store):
        _curator(store, runner=failing_runner).calc(CurateJobInputs(job_id="trip1",
                                                                    pipeline="minimal"))
        assert store.read_results("trip1") is None

    def test_by_default_the_worker_returns_no_pictures(self, store):
        """spec-115 R8: displays and face crops are made by the server (LocalRenders), not the worker."""
        _curator(store).calc(CurateJobInputs(job_id="trip1", pipeline="minimal"))
        paths = store.paths("trip1")
        assert not paths.display_dir.exists()
        assert list(paths.faces_dir.iterdir()) == []
        assert store.results_files("trip1") == ["album.json", "run.json", "results/bundle.json"]

    def test_worker_renders_can_be_asked_for(self, store):
        _curator(store).calc(CurateJobInputs(job_id="trip1", pipeline="minimal", write_renders=True))
        display_dir = store.paths("trip1").display_dir
        assert sorted(p.name for p in display_dir.iterdir()) == ["a.jpg", "c.jpg"]

    def test_run_record_names_the_device_and_the_flipped_steps(self, store):
        res = _curator(store).calc(CurateJobInputs(job_id="trip1", pipeline="minimal", device="cuda"))
        record = json.loads(store.paths("trip1").run_path.read_text(encoding="utf-8"))
        assert record["ok"] is True and record["device"] == "cuda"
        assert record["device_flipped_steps"] == res.flipped
        assert "score_ava" in record["device_flipped_steps"]

    def test_rerunning_the_same_job_reproduces_the_album(self, store):
        clock = itertools.count()
        curator = _curator(store, clock=lambda: f"2026-10-09T00:00:0{next(clock)}+00:00")
        first = curator.calc(CurateJobInputs(job_id="trip1", pipeline="minimal"))
        second = curator.calc(CurateJobInputs(job_id="trip1", pipeline="minimal"))
        assert first.album["created_at"] != second.album["created_at"]   # wall clock, by design
        assert stable_album(first.album) == stable_album(second.album)   # AC4
        assert stable_album(store.read_album("trip1")) == stable_album(second.album)

    def test_the_runner_gets_the_shared_store_paths_and_nothing_else(self, store):
        seen = {}

        def spy(images, plan):
            seen["images"] = list(images)
            return fake_runner(images, plan)

        # a stray file in the store must not reach the pipeline: it gets exactly the job's inputs
        store.inputs.put_bytes("ffffffffffffffff/stray.jpg", b"x")
        _curator(store, runner=spy).calc(CurateJobInputs(job_id="trip1", pipeline="minimal"))
        assert [p.name for p in seen["images"]] == list(NAMES)
        assert all(store.inputs.root in p.parents for p in seen["images"])

    def test_two_jobs_of_the_same_photos_hand_the_pipeline_the_same_paths(self, store):
        """spec-112 TW1.3: the feature cache is keyed by path, so this is what makes job 2 a cache hit."""
        _stage_shared(store, "trip2")
        seen = []

        def spy(images, plan):
            seen.append([str(p) for p in images])
            return fake_runner(images, plan)

        for job in ("trip1", "trip2"):
            _curator(store, runner=spy).calc(CurateJobInputs(job_id=job, pipeline="minimal"))
        assert seen[0] == seen[1]

    def test_an_older_job_with_images_in_input_still_runs(self, tmp_path):
        store = JobStore(tmp_path)
        store.stage_inputs("old", [(n, b"jpeg-bytes") for n in NAMES])
        res = _curator(store).calc(CurateJobInputs(job_id="old", pipeline="minimal"))
        assert res.ok and res.n_inputs == 4
        assert [p["id"] for p in res.album["picks"]] == ["a", "c"]

    def test_a_job_with_no_staged_inputs_fails_with_the_path_it_looked_in(self, tmp_path):
        store = JobStore(tmp_path)
        res = _curator(store).calc(CurateJobInputs(job_id="empty", pipeline="minimal"))
        assert not res.ok
        assert "no staged inputs" in res.error and "input" in res.error
        assert store.read_album("empty") is None

    def test_a_failed_step_is_reported_and_no_album_is_written(self, store):
        res = _curator(store, runner=failing_runner).calc(
            CurateJobInputs(job_id="trip1", pipeline="minimal"))
        assert not res.ok and res.failed_step == "score_ava" and res.error == "boom"
        assert store.read_album("trip1") is None
        record = json.loads(store.paths("trip1").run_path.read_text(encoding="utf-8"))
        assert record["ok"] is False

    def test_an_invalid_job_id_is_rejected_before_anything_runs(self, store):
        with pytest.raises(ValueError, match="invalid job_id"):
            _curator(store).calc(CurateJobInputs(job_id="../evil", pipeline="minimal"))


class ut_StableAlbum:
    def test_drops_only_created_at(self):
        album = {"job_id": "j", "created_at": "t", "picks": []}
        assert stable_album(album) == {"job_id": "j", "picks": []}

    def test_none_passes_through(self):
        assert stable_album(None) is None
