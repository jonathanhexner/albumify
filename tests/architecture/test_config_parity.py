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
import inspect
import re

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
        "cluster_people": {
            "K": 7, "distance_threshold": 0.30,
            "yaw_max": 20.0, "blur_min": 80.0,
        }
    }
    alb_eff = effective_config_from_albumify(albumify_step_configs)

    deltas = compute(fc_eff, alb_eff)
    assert deltas == [], f"Expected parity, got: {deltas}"


def test_diff_picks_up_real_difference():
    fc_eff = effective_config_from_fc_config(PipelineConfig(K=5))
    alb_eff = effective_config_from_albumify({"cluster_people": {"K": 10}})
    deltas = compute(fc_eff, alb_eff)
    diffed_fields = {d.field for d in deltas}
    assert "K" in diffed_fields


def test_effective_config_drops_noise_fields():
    """Paths and timestamps must not participate in equality."""
    eff = effective_config_from_fc_config(
        PipelineConfig(K=5, source_dir="a/b", output_dir="c/d")
    )
    assert "source_dir" not in eff
    assert "output_dir" not in eff


def test_bridge_pose_and_det_gates_read_from_config():
    """spec-033 P-F F-1 / P-C C-1 cross-check.

    The pre-fix bridge force-disabled FIVE gates with hardcoded literals.
    Post-fix, the pose (yaw/pitch/roll) and det_score gates read from
    config — only blur_min stays pinned, and only because the active
    InsightFace pipeline has no blur-scoring step (the docstring on
    build_fc_config explains).
    """
    from sim_bench.pipeline.steps import face_cluster_bridge

    src = inspect.getsource(face_cluster_bridge.build_fc_config)
    # The literal "999.0" passed as yaw_max=999.0 / pitch_max=999.0 / roll_max=999.0
    # was the SIGHTING-059 force-disable. The post-P-C call routes through
    # config.get(...) so these literals don't appear unbound on a single line.
    assert "yaw_max=999.0" not in src, (
        "build_fc_config still hardcodes yaw_max=999.0 — that's the "
        "SIGHTING-059 force-disable spec-033 P-C C-1 removed."
    )
    assert re.search(r'config\.get\(\s*["\']yaw_max["\']', src), (
        "build_fc_config must read yaw_max from config (spec-033 P-F F-1)."
    )
    assert re.search(r'config\.get\(\s*["\']det_score_min["\']', src), (
        "build_fc_config must read det_score_min from config (det data is plumbed)."
    )
    # blur_min remains pinned — the docstring must explain why so the
    # next maintainer doesn't innocently "fix" it back to config-driven.
    assert "no blur" in src.lower() or "blur step" in src.lower() or "score_blur" in src.lower(), (
        "build_fc_config's blur_min pin must be explained in the docstring "
        "(InsightFace pipeline currently has no blur step — see "
        "logs/2026-05-15_11-18-44/api.log for the regression this prevents)."
    )


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
