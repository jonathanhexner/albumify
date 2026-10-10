"""spec-110 T2.1 - SpecBuilder: config load + per-step device placement (no Modal, no GPU)."""
from __future__ import annotations

import textwrap

import pytest

from sim_bench.serving.pipeline_runner import SpecBuilder, SpecInputs, default_config_path


class ut_SpecBuilder:
    """Real configs/pipeline.yaml, so a yaml/registry drift shows up here and not on Modal."""

    def test_default_pipeline_loads_with_configs(self):
        res = SpecBuilder().calc(SpecInputs(pipeline="default", device="cpu"))
        assert res.steps[0] == "discover_images"
        assert "select_best" in res.steps
        assert set(res.step_configs) == set(res.steps)

    def test_minimal_pipeline_key_is_derived_from_the_name(self):
        res = SpecBuilder().calc(SpecInputs(pipeline="minimal"))
        assert res.steps == ["discover_images", "score_iqa", "score_ava", "filter_quality",
                             "extract_scene_embedding", "cluster_scenes", "select_best"]

    def test_unknown_pipeline_raises_with_the_key_it_looked_for(self):
        with pytest.raises(KeyError, match="nope_pipeline"):
            SpecBuilder().calc(SpecInputs(pipeline="nope"))

    def test_cuda_flips_every_step_that_accepts_a_device(self):
        res = SpecBuilder().calc(SpecInputs(pipeline="default", device="cuda"))
        # the model-heavy steps the GPU run depends on (SIGHTING-123: score_ava regressed here once)
        for step in ("triage_content", "score_ava", "insightface_detect_faces",
                     "extract_scene_embedding", "extract_face_embeddings"):
            assert step in res.flipped, f"{step} should accept device=cuda"
            assert res.step_configs[step]["device"] == "cuda"
        assert not res.refused, f"typed configs now refuse: {res.refused}"

    def test_cpu_is_written_through_so_no_step_silently_keeps_a_yaml_gpu_default(self):
        res = SpecBuilder().calc(SpecInputs(pipeline="default", device="cpu"))
        assert all(res.step_configs[s]["device"] == "cpu" for s in res.flipped)

    def test_steps_without_a_device_knob_are_left_alone(self):
        res = SpecBuilder().calc(SpecInputs(pipeline="default", device="cuda"))
        untouched = [s for s in res.steps if s not in res.flipped and s not in res.refused]
        assert "discover_images" in untouched
        assert all("device" not in res.step_configs[s] for s in untouched)

    def test_config_path_override_is_honored(self, tmp_path):
        cfg = tmp_path / "pipeline.yaml"
        cfg.write_text(textwrap.dedent("""
            tiny_pipeline:
              - discover_images
            discover_images:
              recursive: false
        """), encoding="utf-8")
        res = SpecBuilder(config_path=cfg).calc(SpecInputs(pipeline="tiny"))
        assert res.steps == ["discover_images"]
        assert res.step_configs["discover_images"] == {"recursive": False}

    def test_as_pipeline_spec_round_trips_the_plan(self):
        res = SpecBuilder().calc(SpecInputs(pipeline="minimal", device="cpu"))
        spec = res.as_pipeline_spec()
        assert list(spec.steps) == res.steps

    def test_default_config_path_points_at_the_repo_config(self):
        assert default_config_path().is_file()
