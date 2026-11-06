"""spec-116 R2: one spec, whichever machine runs it."""
from __future__ import annotations

import json

import pytest

from sim_bench.pipeline.run_spec import RunSpec
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder, SpecInputs
from sim_bench.serving.run_output import RunOutput
from sim_bench.serving.runners import LocalRunner, ModalRunner

from tests.control.fakes import fake_pipeline, stage_fake_album

STEPS = ["discover_images", "quality_gate", "build_face_knn_graph", "select_best"]


def _spec(**clustering):
    return RunSpec.build(STEPS, {"people_clustering": {"K": 11, **clustering},
                                 "select_best": {"max_images_per_cluster": 3}}, album_name="Trip")


class ut_RunSpec:
    def test_people_clustering_reaches_the_steps(self):
        cfg = _spec().step_configs
        assert cfg["build_face_knn_graph"]["K"] == 11
        assert "people_clustering" not in cfg
        assert cfg["select_best"] == {"max_images_per_cluster": 3}

    def test_the_old_key_still_reads(self):
        spec = RunSpec.build(STEPS, {"cluster_people": {"K": 9}})
        assert spec.step_configs["build_face_knn_graph"]["K"] == 9 and "cluster_people" not in spec.step_configs

    def test_a_step_set_on_its_own_wins_over_the_spread(self):
        spec = RunSpec.build(STEPS, {"people_clustering": {"K": 11}, "build_face_knn_graph": {"K": 3}})
        assert spec.step_configs["build_face_knn_graph"]["K"] == 3

    def test_a_run_without_the_clustering_steps_is_left_alone(self):
        cfg = {"people_clustering": {"K": 11}}
        assert RunSpec.build(["discover_images", "select_best"], cfg).step_configs == cfg

    def test_the_callers_config_is_not_changed(self):
        cfg = {"people_clustering": {"K": 11}}
        RunSpec.build(STEPS, cfg)
        assert cfg == {"people_clustering": {"K": 11}}

    def test_json_round_trip(self):
        spec = _spec()
        assert RunSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec

    def test_the_yaml_pipeline_gets_people_clustering_too(self):
        """A job with no spec of its own stops ignoring the block."""
        plan = SpecBuilder().calc(SpecInputs(pipeline="default"))
        assert "people_clustering" not in plan.step_configs and plan.step_configs["build_face_knn_graph"]


class _Remote:
    def __init__(self):
        self.spec = None

    def missing(self, keys):
        return []

    def upload(self, keys, store):
        pass

    def run_job(self, job_id, keys, pipeline, spec=None):
        self.spec = spec
        return {"ok": False, "error": "stop here"}

    def download_results(self, job_id, store):
        return 0

    def cancel(self):
        pass


class ut_BothMachines:
    def test_same_spec_for_both_machines(self, tmp_path):
        """This PC's child reads run_spec.json; the L4 is handed the same dict; both plan the same steps."""
        store = JobStore(tmp_path)
        stage_fake_album(store, "j1")
        spec = _spec().to_dict()
        store.write_run_spec("j1", spec)

        assert "--job-id" in LocalRunner(store).command("j1") and store.read_run_spec("j1") == spec
        remote = _Remote()
        ModalRunner(store, remote=remote).run("j1")
        assert remote.spec == spec

        seen = []

        def spy(images, plan, **kwargs):
            seen.append((plan.steps, plan.step_configs, kwargs["album_name"]))
            return fake_pipeline(images, plan)

        for device in ("cpu", "cuda"):
            JobCurator(store, runner=spy, output_builder=lambda c, r: RunOutput()).calc(
                CurateJobInputs(job_id="j1", device=device, run_spec=remote.spec))
        assert seen[0][0] == seen[1][0] == STEPS and seen[0][2] == "Trip"
        assert seen[0][1]["build_face_knn_graph"] == seen[1][1]["build_face_knn_graph"]
        assert seen[0][1]["build_face_knn_graph"]["K"] == 11

    def test_only_the_device_differs(self):
        spec = RunSpec.build(["discover_images", "extract_face_embeddings"],
                             {"extract_face_embeddings": {"device": "cpu", "backend": "insightface"},
                              "default_pipeline": ["a"], "people_clustering": {"K": 1}})
        builder = SpecBuilder()
        cpu, gpu = builder.place(spec, "cpu"), builder.place(spec, "cuda")
        assert gpu.step_configs["extract_face_embeddings"]["device"] == "cuda"
        assert spec.step_configs["extract_face_embeddings"]["device"] == "cpu"     # not mutated
        assert {**gpu.step_configs["extract_face_embeddings"], "device": "cpu"} == \
            cpu.step_configs["extract_face_embeddings"]
