"""spec-041 — tests for scripts/migrate_fc_profiles.py (FCParams form).

Round-trip a flat legacy profile through the migrator and assert:

* Flat legacy profile → reshaped into ``FCParams.model_dump_json()`` shape,
  backup file written, content validates back to FCParams.
* Re-running on an already-migrated profile is a no-op (backup file present).
* Dry-run does not touch the filesystem.
* Invalid JSON / non-dict payloads are skipped, not crashed.
* spec-040 v2 shape (``{step_configs: {...}}``) round-trips through the
  ``_extract_flat`` path.
"""
from __future__ import annotations

import json
from pathlib import Path

from face_cluster.fc_params import FCParams
from scripts.migrate_fc_profiles import migrate_one


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
    # The migrated profile is a flat FCParams dump — every known field present.
    assert new["K"] == 7
    assert new["distance_threshold"] == 0.4
    assert new["merge_enabled"] is False
    # Round-trips through FCParams.
    FCParams.model_validate(new)


def test_migrate_one_is_idempotent(tmp_path: Path):
    """Backup file presence is the idempotence signal."""
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


def test_migrate_one_skips_non_dict_payload(tmp_path: Path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert migrate_one(p) == "invalid"


def test_migrate_one_skips_unrecognized_fields(tmp_path: Path):
    """A profile containing keys foreign to FCParams must be flagged invalid,
    not silently dropped — extra='forbid' guarantees this."""
    p = tmp_path / "weird.json"
    p.write_text(json.dumps({"K": 5, "totally_unknown_knob": 42}), encoding="utf-8")
    assert migrate_one(p) == "invalid"


def test_migrate_one_handles_spec040_v2_shape(tmp_path: Path):
    """spec-040 v2 shape (step_configs broadcast) → flat FCParams."""
    p = tmp_path / "spec040.json"
    flat = {"K": 9, "distance_threshold": 0.3}
    legacy_v2 = {
        "version": 2,
        "step_configs": {"quality_gate": flat, "build_face_knn_graph": flat},
        "legacy_flat": flat,
    }
    p.write_text(json.dumps(legacy_v2), encoding="utf-8")
    assert migrate_one(p) == "migrated"
    new = json.loads(p.read_text(encoding="utf-8"))
    assert new["K"] == 9
    assert new["distance_threshold"] == 0.3
    # Backup preserves the original spec-040 shape.
    assert json.loads(p.with_suffix(".v1.json").read_text(encoding="utf-8")) == legacy_v2
