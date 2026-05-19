"""spec-040 Phase 5b — test scripts/migrate_fc_profiles.py.

Round-trip a flat legacy profile and a v2-shaped profile through the
migrator and assert the right outcome:

* Flat legacy profile → reshaped into ``{step_configs: {step: flat}, ...}``,
  backup file written, `legacy_flat` preserved.
* v2-shaped profile → left alone (idempotent re-run).
* Migrated profile loads correctly through the new FC App's expected
  step_configs flow (every clustering step gets the same flat dict).
"""
from __future__ import annotations

import json
from pathlib import Path

from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
from scripts.migrate_fc_profiles import (
    is_v2_shape,
    migrate_one,
    reshape_v1_to_v2,
)


def test_reshape_v1_to_v2_wraps_every_unified_step():
    legacy = {"K": 5, "distance_threshold": 0.3, "merge_enabled": True}
    out = reshape_v1_to_v2(legacy)
    assert out["version"] == 2
    assert set(out["step_configs"]) == set(UNIFIED_CLUSTERING_STEPS)
    for name in UNIFIED_CLUSTERING_STEPS:
        assert out["step_configs"][name] == legacy, (
            f"step {name!r} did not receive the full legacy flat dict"
        )
    # legacy_flat preserved so a v1 export is reconstructible.
    assert out["legacy_flat"] == legacy


def test_is_v2_shape_recognizes_migrated_profiles():
    assert is_v2_shape(reshape_v1_to_v2({"K": 5}))
    assert not is_v2_shape({"K": 5, "distance_threshold": 0.3})
    assert not is_v2_shape({})  # empty profile is treated as not v2


def test_migrate_one_writes_backup_and_overwrites(tmp_path: Path):
    p = tmp_path / "myprofile.json"
    legacy = {"K": 7, "distance_threshold": 0.4, "merge_enabled": False}
    p.write_text(json.dumps(legacy), encoding="utf-8")

    status = migrate_one(p)
    assert status == "migrated"

    backup = p.with_suffix(".v1.json")
    assert backup.exists(), "v1 backup not written"
    assert json.loads(backup.read_text(encoding="utf-8")) == legacy

    new = json.loads(p.read_text(encoding="utf-8"))
    assert is_v2_shape(new)
    assert new["legacy_flat"] == legacy


def test_migrate_one_is_idempotent(tmp_path: Path):
    """Re-running the migrator on an already-v2 profile is a no-op."""
    p = tmp_path / "myprofile.json"
    legacy = {"K": 7, "distance_threshold": 0.4}
    p.write_text(json.dumps(legacy), encoding="utf-8")
    assert migrate_one(p) == "migrated"
    snapshot = p.read_text(encoding="utf-8")
    assert migrate_one(p) == "already_v2"
    # Content unchanged on second run.
    assert p.read_text(encoding="utf-8") == snapshot


def test_migrate_one_dry_run_does_not_write(tmp_path: Path):
    p = tmp_path / "myprofile.json"
    legacy = {"K": 7}
    p.write_text(json.dumps(legacy), encoding="utf-8")
    assert migrate_one(p, dry_run=True) == "migrated"
    # File unchanged; no backup created.
    assert json.loads(p.read_text(encoding="utf-8")) == legacy
    assert not p.with_suffix(".v1.json").exists()


def test_migrate_one_skips_invalid(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert migrate_one(p) == "invalid"
