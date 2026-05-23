"""spec-041 Phase 1 — unit tests for FCParams.

Locks in the contract: defaults are stable, unknown keys are rejected,
out-of-range values raise, JSON round-trip preserves equality, and the
two boundary helpers (``to_fc_config`` / ``to_step_configs``) produce the
shapes downstream code expects.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from face_cluster.config import PipelineConfig as FCConfig
from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
from face_cluster.fc_params import FCParams


def test_defaults_construct_without_error():
    p = FCParams()
    assert p.K == 5
    assert p.distance_threshold == 0.35
    assert p.min_cluster_size == 2
    assert p.merge_enabled is False
    assert p.cluster_diameter_cap_enabled is False


def test_defaults_round_trip_to_fc_config():
    """FCParams() should produce an FCConfig identical to FCConfig() on every shared field."""
    p = FCParams()
    cfg = p.to_fc_config()
    for name in FCParams.model_fields:
        assert getattr(cfg, name) == getattr(p, name), (
            f"field {name!r} drifted: FCParams={getattr(p, name)!r} "
            f"FCConfig={getattr(cfg, name)!r}"
        )


def test_extra_forbid_rejects_unknown_key():
    with pytest.raises(ValidationError):
        FCParams(K=5, bogus_knob=True)


@pytest.mark.parametrize("bad", [
    {"K": 0},                       # below ge=1
    {"K": 101},                     # above le=100
    {"distance_threshold": 2.5},    # above le=1.0
    {"distance_threshold": -0.1},   # below ge=0.01
    {"blur_min": -1.0},             # below ge=0.0
    {"merge_exemplar_percentile": 150},  # above le=100
    {"merge_threshold_alpha": -0.5},     # below ge=0
    {"max_full_diameter": 5.0},     # above le=2.0
])
def test_out_of_range_raises(bad):
    with pytest.raises(ValidationError):
        FCParams(**bad)


def test_optional_fields_accept_none():
    p = FCParams(min_face_area=None, det_score_min=None)
    assert p.min_face_area is None
    assert p.det_score_min is None


def test_model_dump_json_round_trip_preserves_equality():
    p1 = FCParams(K=7, distance_threshold=0.42, merge_enabled=True,
                  merge_candidate_threshold=0.5)
    p2 = FCParams.model_validate_json(p1.model_dump_json())
    assert p1 == p2


def test_to_step_configs_keys_match_unified_chain():
    p = FCParams()
    sc = p.to_step_configs()
    assert set(sc.keys()) == set(UNIFIED_CLUSTERING_STEPS)
    # Each step gets the same full dict — and they're independent copies
    # so a step mutating its config dict doesn't leak across steps.
    first = sc[UNIFIED_CLUSTERING_STEPS[0]]
    second = sc[UNIFIED_CLUSTERING_STEPS[1]]
    assert first == second
    first["K"] = 999
    assert second["K"] != 999, "step_configs entries must be independent copies"


def test_to_step_configs_dict_is_fcconfig_buildable():
    """Every step's config dict must produce a valid FCConfig when splatted in."""
    p = FCParams(K=7, merge_enabled=True)
    sc = p.to_step_configs()
    for name, cfg in sc.items():
        FCConfig(**cfg)  # raises if any key is wrong


def test_save_load_round_trip(tmp_path: Path):
    p1 = FCParams(K=8, distance_threshold=0.42, blur_min=25.0, merge_enabled=True)
    profile = tmp_path / "test_profile.json"
    p1.save(profile)
    assert profile.exists()
    raw = json.loads(profile.read_text(encoding="utf-8"))
    assert raw["K"] == 8
    assert raw["distance_threshold"] == 0.42
    p2 = FCParams.load(profile)
    assert p1 == p2


def test_non_default_values_propagate_through_to_fc_config():
    p = FCParams(
        K=8, distance_threshold=0.42,
        merge_enabled=True, merge_candidate_threshold=0.5,
        cluster_diameter_cap_enabled=True, max_full_diameter=1.0,
    )
    cfg = p.to_fc_config()
    assert cfg.K == 8
    assert cfg.distance_threshold == 0.42
    assert cfg.merge_enabled is True
    assert cfg.merge_candidate_threshold == 0.5
    assert cfg.cluster_diameter_cap_enabled is True
    assert cfg.max_full_diameter == 1.0
