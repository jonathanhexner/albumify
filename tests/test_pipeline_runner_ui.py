"""Pipeline runner UI tests using streamlit.testing.v1.AppTest.

Verifies:
- Component renders without exceptions
- Widget values persist across reruns (no jump/reset)
- Config dict correctly reflects widget values (fc_K overwrite bug)
- Slider interactions don't crash
"""
import pytest

try:
    from streamlit.testing.v1 import AppTest
    HAS_APPTEST = True
except ImportError:
    HAS_APPTEST = False


# Minimal Streamlit page that exercises the pipeline runner with mocked API.
# Uses inline mocking so AppTest can run it standalone.
_TEST_PAGE = r'''
import sys
import streamlit as st
from unittest.mock import MagicMock, patch

# --- Mock the API layer before importing pipeline_runner ---
_mock_client = MagicMock()
_mock_client.get_available_pipelines.return_value = {
    "default_pipeline": ["discover_images", "detect_persons", "select_best"],
}
_mock_client.get_user_config.return_value = {
    "selected_pipeline": "default_pipeline",
    "config": {},
}
_mock_client.start_pipeline.return_value = "test-job-id"

# Patch get_client at module level
patch("app.streamlit.api_client.get_client", return_value=_mock_client).start()

# Clear any cached _load_user_settings from previous test runs
import app.streamlit.components.pipeline_runner as _pr_mod_early
if hasattr(_pr_mod_early._load_user_settings, 'clear'):
    _pr_mod_early._load_user_settings.clear()

# Initialize session state with a mock SessionState
from app.streamlit.session import SessionState
if "app_state" not in st.session_state:
    st.session_state.app_state = SessionState(api_connected=True)

# Intercept _start_pipeline to capture the config dict it builds
import app.streamlit.components.pipeline_runner as _pr_mod

_original_start = _pr_mod._start_pipeline
def _capturing_start(album_id, pipeline_name, steps, config):
    st.session_state["_captured_config"] = config
    return None  # Don't actually start pipeline
_pr_mod._start_pipeline = _capturing_start

from app.streamlit.models import Album

album = Album(album_id="test-album", name="Test Album", source_directory="/test/path", total_images=50)

st.header("Pipeline Runner Test Page")
job_id = _pr_mod.render_pipeline_runner(album)
'''


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
class ut_PipelineRunnerUI:
    """Pipeline runner UI tests."""

    def test_renders_without_exception(self):
        """Pipeline runner must render without any exceptions."""
        at = AppTest.from_string(_TEST_PAGE, default_timeout=15)
        at.run()
        assert not at.exception, f"App raised exception: {at.exception}"

    def test_has_run_pipeline_button(self):
        """Must have a 'Run Pipeline' button."""
        at = AppTest.from_string(_TEST_PAGE, default_timeout=15)
        at.run()
        assert not at.exception
        button_labels = [b.label for b in at.button]
        assert any("Run Pipeline" in label for label in button_labels), (
            f"No 'Run Pipeline' button. Found: {button_labels}"
        )

    def test_has_save_settings_button(self):
        """Must have a 'Save Settings' button."""
        at = AppTest.from_string(_TEST_PAGE, default_timeout=15)
        at.run()
        assert not at.exception
        button_labels = [b.label for b in at.button]
        assert any("Save" in label for label in button_labels), (
            f"No 'Save Settings' button. Found: {button_labels}"
        )

    def test_slider_value_persists_across_rerun(self):
        """Moving a slider must not reset other slider values on rerun."""
        at = AppTest.from_string(_TEST_PAGE, default_timeout=15)
        at.run()
        assert not at.exception

        # Find the Min IQA slider and set it
        iqa_slider = at.slider(key="config_min_iqa")
        iqa_slider.set_value(0.5)
        at.run()
        assert not at.exception

        # Value must persist
        assert at.slider(key="config_min_iqa").value == 0.5

    def test_multiple_slider_changes_persist(self):
        """Changing multiple sliders in sequence must preserve all values."""
        at = AppTest.from_string(_TEST_PAGE, default_timeout=15)
        at.run()
        assert not at.exception

        # Change IQA slider
        at.slider(key="config_min_iqa").set_value(0.6)
        at.run()
        assert not at.exception

        # Change sharpness slider
        at.slider(key="config_min_sharpness").set_value(0.3)
        at.run()
        assert not at.exception

        # Both values must persist
        assert at.slider(key="config_min_iqa").value == 0.6, "IQA slider reset!"
        assert at.slider(key="config_min_sharpness").value == 0.3, "Sharpness slider reset!"

    def test_fc_K_slider_value_reaches_config_dict(self):
        """The K slider value must reach the config dict, not be overwritten by defaults.

        Regression test: fc_K=5 was unconditionally set AFTER the slider,
        so the config dict always had K=5 regardless of slider position.
        This test captures the config dict by intercepting _start_pipeline.
        """
        at = AppTest.from_string(_TEST_PAGE, default_timeout=15)
        at.run()
        assert not at.exception

        # face_cluster_knn should be default method — change K slider to 15
        try:
            k_slider = at.slider(key="rc_K")
        except KeyError:
            pytest.skip("K slider not rendered (face_cluster_knn not selected)")

        k_slider.set_value(15)
        at.run()
        assert not at.exception

        # Click Run Pipeline to trigger config dict capture
        run_btn = None
        for b in at.button:
            if "Run Pipeline" in b.label:
                run_btn = b
                break
        assert run_btn is not None, "Run Pipeline button not found"
        run_btn.click()
        at.run()

        # Verify the captured config has K=15, not K=5
        assert "_captured_config" in at.session_state, (
            "Config was not captured — _start_pipeline was not called"
        )
        captured = at.session_state["_captured_config"]
        cluster_config = captured.get("cluster_people", {})
        actual_K = cluster_config.get("K")
        assert actual_K == 15, (
            f"fc_K overwrite bug! Config dict has K={actual_K}, expected 15. "
            f"The slider value was overwritten by the defaults block."
        )

    def test_no_exception_after_five_reruns(self):
        """Multiple consecutive reruns must not cause exceptions (stability test)."""
        at = AppTest.from_string(_TEST_PAGE, default_timeout=15)
        at.run()
        assert not at.exception

        for i in range(5):
            at.run()
            assert not at.exception, f"Exception on rerun {i+1}: {at.exception}"
