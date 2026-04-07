"""Streamlit app smoke tests using streamlit.testing.v1."""
import pytest

try:
    from streamlit.testing.v1 import AppTest
    HAS_APPTEST = True
except ImportError:
    HAS_APPTEST = False

APP_PATH = "app/face_clustering.py"


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception, f"App raised exception on load: {at.exception}"


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_run_tab_has_run_button():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    button_labels = [b.label for b in at.button]
    assert any("Run" in label for label in button_labels), (
        f"No Run button found. Buttons: {button_labels}"
    )


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_browse_tab_empty_state_no_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    # Browse tab loads without error when no data loaded
    assert not at.exception
