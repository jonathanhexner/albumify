"""spec-115 R12: a run records what it ran with; two runs' settings can be diffed."""
from types import SimpleNamespace

from sim_bench.serving.run_settings import COMMIT_ENV, RunSettings

PLAN = SimpleNamespace(steps=["discover_images", "score_iqa"],
                       step_configs={"score_iqa": {"model": "a"}})


class ut_RunSettings:

    def test_collect_records_steps_params_runner_and_versions(self, monkeypatch):
        monkeypatch.setenv(COMMIT_ENV, "abc1234")
        s = RunSettings.collect(PLAN, pipeline="minimal", runner="modal_gpu", device="cuda")
        assert s.commit == "abc1234" and s.runner == "modal_gpu" and s.device == "cuda"
        assert s.steps == [{"step": "discover_images", "params": {}}, {"step": "score_iqa", "params": {"model": "a"}}]
        assert "numpy" in s.packages
        assert RunSettings.from_dict(s.to_dict()) == s

    def test_diff_names_what_changed(self):
        a = RunSettings("default", [{"step": "x", "params": {"k": 1}}, {"step": "y", "params": {}}],
                        "c1", "local", "cpu", {"numpy": "1.26"})
        b = RunSettings("default", [{"step": "x", "params": {"k": 2}}], "c2", "modal_gpu", "cuda",
                        {"numpy": "1.26", "torch": "2.4"})
        d = a.diff(b)
        assert d["commit"] == ["c1", "c2"] and d["runner"] == ["local", "modal_gpu"]
        assert d["x.k"] == [1, 2] and d["steps"] == [["y"], []] and d["package torch"] == [None, "2.4"]
        assert "pipeline" not in d and a.diff(a) == {}
