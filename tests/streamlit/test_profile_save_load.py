"""spec-087 / SIGHTING-106: profile Save/Load must round-trip the FULL config
(quality/detection/selection), not just the rc_* clustering subset.

The 3 helpers are pure (operate on a dict standing in for st.session_state), so the
save->load round-trip is testable without a running Streamlit app.
"""

from app.streamlit.components.pipeline_runner import (
    _build_profile_payload,
    _apply_profile_to_session,
    _resolve_saved_config,
)


class ut_ProfileSaveLoad:

    def test_payload_has_nested_config_and_flat_rc(self):
        """AC1/AC2: a saved profile carries the nested config AND the flat rc_* keys."""
        ss = {
            "rc_K": 5, "rc_dist": 0.4,
            "_last_built_config": {"filter_quality": {"min_sharpness": 0.1}},
        }
        payload = _build_profile_payload(ss)
        assert payload["config"]["filter_quality"]["min_sharpness"] == 0.1  # AC1
        assert payload["rc_K"] == 5 and payload["rc_dist"] == 0.4           # AC2 (cross-app)

    def test_load_clears_config_keys_and_stages_pending(self):
        """AC3: loading clears stale config_* widget keys and stages the profile config."""
        ss = {"config_min_sharpness": 0.9, "config_min_iqa": 0.5, "rc_K": 1}
        profile = {"rc_K": 7, "config": {"filter_quality": {"min_sharpness": 0.1}}}
        _apply_profile_to_session(profile, ss)
        assert ss["rc_K"] == 7
        assert "config_min_sharpness" not in ss  # cleared -> widget re-inits from profile
        assert "config_min_iqa" not in ss
        assert ss["_pending_profile_config"] == {"filter_quality": {"min_sharpness": 0.1}}

    def test_old_profile_without_config_loads_clean(self):
        """AC4: legacy profiles (flat rc_* only) load with no error, no pending stage."""
        ss = {"config_min_sharpness": 0.9}
        _apply_profile_to_session({"rc_K": 3}, ss)
        assert ss["rc_K"] == 3
        assert "_pending_profile_config" not in ss
        assert ss["config_min_sharpness"] == 0.9  # untouched (back-compat behaviour)

    def test_roundtrip_preserves_sharpness(self):
        """AC5: save sharpness=0.1, load into a stale session, the widget will show 0.1."""
        src = {"rc_K": 5, "_last_built_config": {"filter_quality": {"min_sharpness": 0.1}}}
        payload = _build_profile_payload(src)

        dst = {"config_min_sharpness": 0.9, "rc_K": 1}  # stale session
        _apply_profile_to_session(payload, dst)

        # The render-time resolver hands the profile config to the widget-init path.
        saved_config = _resolve_saved_config(dst, api_config={"filter_quality": {"min_sharpness": 0.9}})
        assert saved_config["filter_quality"]["min_sharpness"] == 0.1
        assert dst["rc_K"] == 5

    def test_resolve_prefers_pending_then_api_and_is_one_shot(self):
        ss = {"_pending_profile_config": {"x": 1}}
        assert _resolve_saved_config(ss, {"x": 2}) == {"x": 1}   # pending wins
        assert _resolve_saved_config(ss, {"x": 2}) == {"x": 2}   # one-shot: now API
