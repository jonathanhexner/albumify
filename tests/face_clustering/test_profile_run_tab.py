"""Profile save/load round-trip for the face-clustering Run tab.

Covers SIGHTING-059 item 4: Run tab now exposes profile load/save in addition
to Recluster.  These are pure-Python tests against ProfileStore + the param
key constants — they intentionally do not boot Streamlit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from face_cluster.profile_store import ProfileStore

# state.py lives at app/face_clustering/state.py and is not a package, so add
# it to sys.path the same way main.py does.
_FC_APP_DIR = Path(__file__).resolve().parents[2] / "app" / "face_clustering"
if str(_FC_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_FC_APP_DIR))

from state import _RC_PARAM_KEYS, _RUN_PARAM_KEYS  # noqa: E402


class ut_RunProfileKeys:
    def test_run_keys_disjoint_from_rc(self):
        # Every Run-tab key must use the run_ prefix and never overlap rc_ keys
        # so a profile saved from one tab loads cleanly on the other.
        assert _RC_PARAM_KEYS.isdisjoint(_RUN_PARAM_KEYS)

    def test_every_run_key_has_run_prefix(self):
        bad = [k for k in _RUN_PARAM_KEYS if not k.startswith("run_")]
        assert not bad, f"Run keys missing run_ prefix: {bad}"

    def test_run_param_keys_cover_pipeline_stages(self):
        # Sanity: profile must carry each stage knob
        expected_subset = {
            "run_blur_min", "run_max_faces", "run_min_face_area", "run_det_score_min",
            "run_yaw_max", "run_pitch_max", "run_roll_max", "run_require_pose",
            "run_K", "run_dist", "run_min_cluster",
            "run_N_exemplars", "run_d10_thresh", "run_suppression",
            "run_split", "run_merge", "run_attach",
        }
        missing = expected_subset - _RUN_PARAM_KEYS
        assert not missing, f"_RUN_PARAM_KEYS is missing {missing}"

    def test_run_param_keys_cover_merge_knobs(self):
        # The merge knobs use the same physical widget keys (with run_ prefix)
        # as the recluster ones — confirm the parallel set is present.
        expected_merge_subset = {
            "run_merge_use_adaptive", "run_merge_exemplar_pct", "run_merge_global_pct",
            "run_merge_alpha", "run_merge_beta", "run_merge_candidate", "run_merge_exemplar_thresh",
            "run_merge_support_frac", "run_merge_support_min", "run_merge_margin", "run_merge_diameter",
            "run_merge_use_cross_gate", "run_merge_cross_thresh", "run_merge_cross_max_size",
            "run_merge_support_unique",
        }
        missing = expected_merge_subset - _RUN_PARAM_KEYS
        assert not missing, f"_RUN_PARAM_KEYS missing merge knobs: {missing}"


class ut_ProfileRoundTrip:
    @pytest.fixture
    def store(self, tmp_path):
        return ProfileStore(profiles_dir=tmp_path / "profiles")

    def test_save_then_load_preserves_run_params(self, store):
        sample = {
            "run_K": 7,
            "run_dist": 0.42,
            "run_blur_min": 80.0,
            "run_require_pose": True,
            "run_merge": True,
            "run_merge_margin": 0.05,
        }
        store.save("budapest_v1", sample)
        loaded = store.load("budapest_v1")
        assert loaded == sample

    def test_load_filters_to_allowed_keys(self, store):
        # A profile written with stray keys (perhaps from an older app version)
        # must not pollute session_state when reloaded.  The Run tab loader
        # filters by _RUN_PARAM_KEYS — emulate that here.
        store.save("mixed", {
            "run_K": 5,
            "rc_K": 999,                   # belongs to recluster, not run
            "totally_unknown_key": "xyz",  # garbage
        })
        loaded = store.load("mixed")
        kept = {k: v for k, v in loaded.items() if k in _RUN_PARAM_KEYS}
        assert kept == {"run_K": 5}

    def test_list_names_includes_saved_profile(self, store):
        store.save("alpha", {"run_K": 4})
        store.save("beta",  {"run_K": 6})
        names = store.list_names()
        assert "alpha" in names and "beta" in names

    def test_load_unknown_profile_returns_empty_dict(self, store):
        assert store.load("does_not_exist") == {}
