"""spec-087 / SIGHTING-106 / SIGHTING-109: profile Save/Load round-trips the FULL
config (quality/detection/selection), not just the rc_* clustering subset.

The helpers are pure (operate on a dict standing in for st.session_state), so the
save->load round-trip is testable without a running Streamlit app.
"""

from app.streamlit.components.pipeline_runner import (
    _build_profile_payload,
    _apply_profile_to_session,
)


class ut_ProfileSaveLoad:

    def test_payload_has_nested_config_and_flat_rc(self):
        """Save: a profile carries the nested config AND the flat rc_* keys."""
        ss = {
            "rc_K": 5, "rc_dist": 0.4,
            "_last_built_config": {"filter_quality": {"min_sharpness": 0.05}},
        }
        payload = _build_profile_payload(ss)
        assert payload["config"]["filter_quality"]["min_sharpness"] == 0.05
        assert payload["rc_K"] == 5 and payload["rc_dist"] == 0.4  # cross-app interop

    def test_load_sets_widget_keys_directly(self):
        """SIGHTING-109: load writes config values straight into the widget keys."""
        ss = {"config_min_sharpness": 0.2, "config_min_iqa": 0.5, "rc_K": 1}
        profile = {"rc_K": 7, "config": {
            "filter_quality": {"min_sharpness": 0.05, "min_iqa_score": 0.3},
            "select_best": {"max_images_per_cluster": 4, "siamese": {"enabled": True}},
        }}
        _apply_profile_to_session(profile, ss)
        assert ss["rc_K"] == 7
        assert ss["config_min_sharpness"] == 0.05   # the bug: now restored
        assert ss["config_min_iqa"] == 0.3
        assert ss["config_max_per_cluster"] == 4
        assert ss["config_siamese"] is True         # nested path works

    def test_old_profile_without_config_loads_clean(self):
        """Back-compat: legacy profiles (flat rc_* only) load, config_* untouched."""
        ss = {"config_min_sharpness": 0.9}
        _apply_profile_to_session({"rc_K": 3}, ss)
        assert ss["rc_K"] == 3
        assert ss["config_min_sharpness"] == 0.9  # nothing to restore -> left as-is

    def test_roundtrip_preserves_sharpness(self):
        """Save 0.05, load into a session showing 0.2 -> widget key becomes 0.05."""
        src = {"rc_K": 5, "_last_built_config": {"filter_quality": {"min_sharpness": 0.05}}}
        payload = _build_profile_payload(src)

        dst = {"config_min_sharpness": 0.2, "rc_K": 1}  # stale session
        _apply_profile_to_session(payload, dst)

        assert dst["config_min_sharpness"] == 0.05  # the slider will now show 0.05
        assert dst["rc_K"] == 5

    def test_missing_field_left_untouched(self):
        """A config that omits a field doesn't clobber that widget's current value."""
        ss = {"config_min_sharpness": 0.2}
        _apply_profile_to_session({"config": {"select_best": {"min_score_threshold": 0.5}}}, ss)
        assert ss["config_min_score"] == 0.5
        assert ss["config_min_sharpness"] == 0.2  # untouched (not in this profile)
