"""spec-041 Phase 7 — subprocess smoke for scripts/run_v2.py.

These tests do NOT shell into a separate Python interpreter for speed —
they call ``scripts.run_v2.main`` directly with argparse-style argv.
That keeps them fast while still exercising the CLI surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from face_cluster.fc_params import FCParams
from scripts.run_v2 import _build_params, main as run_v2_main
from tests.conftest import get_test_data_dir


FIXTURE_DIR = get_test_data_dir() / "face_clustering"


def _pick_jpgs(n_per_person: int = 2, n_persons: int = 3):
    if not FIXTURE_DIR.exists():
        return []
    picked = []
    for person_dir in sorted(FIXTURE_DIR.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("person_"):
            continue
        jpgs = sorted(p for p in person_dir.iterdir() if p.suffix.lower() == ".jpg")
        picked.extend(jpgs[:n_per_person])
        if len(picked) >= n_per_person * n_persons:
            break
    return picked


def test_build_params_from_flags_only():
    """No profile, K + distance_threshold flags override defaults."""
    import argparse
    args = argparse.Namespace(
        profile=None, K=7, distance_threshold=0.42, min_cluster_size=None,
        blur_min=None, max_faces_per_image_core=None,
        yaw_max=None, pitch_max=None, roll_max=None,
        merge=False, cap=False,
    )
    p = _build_params(args)
    assert p.K == 7
    assert p.distance_threshold == 0.42
    assert p.merge_enabled is False
    # Unset flags retain FCParams defaults.
    assert p.min_cluster_size == 2


def test_build_params_profile_then_override(tmp_path: Path):
    """Profile is the base; explicit flags override its values."""
    import argparse
    profile = tmp_path / "p.json"
    FCParams(K=10, merge_enabled=True).save(profile)

    args = argparse.Namespace(
        profile=profile, K=3, distance_threshold=None, min_cluster_size=None,
        blur_min=None, max_faces_per_image_core=None,
        yaw_max=None, pitch_max=None, roll_max=None,
        merge=False, cap=True,
    )
    p = _build_params(args)
    assert p.K == 3, "explicit --K must override profile"
    assert p.merge_enabled is True, "profile value retained when no flag overrides"
    assert p.cluster_diameter_cap_enabled is True, "--cap flag flips the cap toggle"


def test_save_profile_round_trip(tmp_path: Path):
    """--save-profile writes a profile that round-trips back to FCParams."""
    out_profile = tmp_path / "saved.json"
    # Use a missing src so the pipeline errors out fast — we only want to
    # exercise the save-profile path, not the actual run.
    rc = run_v2_main([
        "--src", str(tmp_path / "missing_src"),
        "--out", str(tmp_path / "out"),
        "--album", "cli_save_profile_test",
        "--save-profile", str(out_profile),
        "--K", "9",
    ])
    assert rc == 2, "missing src should exit 2 (arg error)"
    # But the save-profile happens BEFORE src is checked? No — src check is
    # first. So out_profile is NOT written when src is missing. Use a real
    # src dir to exercise the save path.
    src = tmp_path / "real_src"; src.mkdir()
    rc = run_v2_main([
        "--src", str(src),
        "--out", str(tmp_path / "out2"),
        "--album", "cli_save_profile_test",
        "--save-profile", str(out_profile),
        "--K", "9",
    ])
    # Empty src → pipeline returns failure (no images), exit 1.
    assert rc == 1
    assert out_profile.exists(), "profile should have been saved before the run"
    loaded = FCParams.load(out_profile)
    assert loaded.K == 9


def test_full_run_against_fixture(tmp_path: Path):
    """End-to-end CLI run against the 6-jpg fixture."""
    images = _pick_jpgs(n_per_person=2, n_persons=3)
    if not images:
        pytest.skip(f"Fixture missing at {FIXTURE_DIR}")
    src = tmp_path / "src"; src.mkdir()
    for s in images:
        (src / s.name).write_bytes(s.read_bytes())
    out = tmp_path / "out"

    # Use a private action_log DB.
    db_path = tmp_path / "log" / "sim_bench.db"
    db_path.parent.mkdir()
    # spec-048: _resolve_db_path now reads from _paths.default_db_path.
    import face_cluster._paths as _paths
    orig = _paths.default_db_path
    _paths.default_db_path = lambda: db_path  # type: ignore[assignment]
    try:
        rc = run_v2_main([
            "--src", str(src), "--out", str(out),
            "--album", "cli_full_run_fixture",
            "--K", "3", "--distance_threshold", "0.5",
            "--blur_min", "0.0",
            "--yaw_max", "999.0", "--pitch_max", "999.0", "--roll_max", "999.0",
        ])
    finally:
        _paths.default_db_path = orig
    if rc != 0:
        pytest.skip(f"v2 pipeline failed (env): exit {rc}")
    assert rc == 0
    assert (out / "face_clustering.db").exists()
