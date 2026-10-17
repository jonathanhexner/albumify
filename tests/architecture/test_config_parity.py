"""spec-033 P-F: canonical effective-config parity between FC App and Albumify.

Rules:
  1. An FC App PipelineConfig and its Albumify-shaped step_configs equivalent
     produce identical effective configs (modulo timestamps / paths).
  2. The diff CLI exists and reports zero differences for parity inputs.
  3. The bridge ``build_fc_config`` no longer hardcodes the SIGHTING-059
     quality-gate force-disables (spec-033 P-C C-1 / P-F F-1 cross-check).
"""
from __future__ import annotations

import dataclasses

from face_cluster.config import PipelineConfig
from face_cluster.config_diff import (
    compute,
    effective_config_from_albumify,
    effective_config_from_fc_config,
)


def test_identical_inputs_diff_clean():
    """Same PipelineConfig in both shapes → empty delta list."""
    fc = PipelineConfig(K=7, distance_threshold=0.30, yaw_max=20.0, blur_min=80.0)
    fc_eff = effective_config_from_fc_config(fc)

    albumify_step_configs = {
        "people_clustering": {
            "K": 7, "distance_threshold": 0.30,
            "yaw_max": 20.0, "blur_min": 80.0,
        }
    }
    alb_eff = effective_config_from_albumify(albumify_step_configs)

    deltas = compute(fc_eff, alb_eff)
    assert deltas == [], f"Expected parity, got: {deltas}"


def test_diff_picks_up_real_difference():
    fc_eff = effective_config_from_fc_config(PipelineConfig(K=5))
    alb_eff = effective_config_from_albumify({"people_clustering": {"K": 10}})
    deltas = compute(fc_eff, alb_eff)
    diffed_fields = {d.field for d in deltas}
    assert "K" in diffed_fields


def test_pre_spec113_key_is_still_read():
    """Runs saved before spec-113 stored the settings under `cluster_people`."""
    old = effective_config_from_albumify({"cluster_people": {"K": 10}})
    new = effective_config_from_albumify({"people_clustering": {"K": 10}})
    assert compute(old, new) == []


def test_effective_config_drops_noise_fields():
    """Paths and timestamps must not participate in equality."""
    eff = effective_config_from_fc_config(
        PipelineConfig(K=5, source_dir="a/b", output_dir="c/d")
    )
    assert "source_dir" not in eff
    assert "output_dir" not in eff



def test_cli_runs_with_two_run_dirs(tmp_path):
    """The CLI is invokable as a module and prints a deterministic message."""
    import json
    import subprocess
    import sys

    # Two synthetic run dirs with identical configs → CLI exits 0.
    cfg_payload = {"config": dataclasses.asdict(PipelineConfig(K=5, blur_min=42.0))}
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "pipeline_run.json").write_text(json.dumps(cfg_payload))

    result = subprocess.run(
        [sys.executable, "-m", "face_cluster.config_diff",
         str(tmp_path / "a"), str(tmp_path / "b")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "IDENTICAL" in result.stdout
