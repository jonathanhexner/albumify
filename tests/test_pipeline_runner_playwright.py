"""Playwright E2E tests for the pipeline runner UI.

Launches a real Streamlit app and tests actual browser interactions.
Catches issues that AppTest misses: session state conflicts, widget rendering,
expander collapse, page jumps.

Usage:
    .venv/Scripts/python -m pytest tests/test_pipeline_runner_playwright.py -v --headed
    (--headed to watch the browser; remove for headless CI)
"""
import subprocess
import sys
import time
import socket
import pytest

# Check if playwright is available
try:
    from playwright.sync_api import sync_playwright, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Minimal Streamlit test page — exercises the pipeline runner with mocked API
_TEST_APP = r'''
import streamlit as st
from unittest.mock import MagicMock, patch

# Mock the API layer
_mock_client = MagicMock()
_mock_client.get_available_pipelines.return_value = {
    "default_pipeline": ["discover_images", "detect_persons", "select_best"],
}
_mock_client.get_user_config.return_value = {
    "selected_pipeline": "default_pipeline",
    "config": {},
}
_mock_client.start_pipeline.return_value = "test-job-id"
patch("app.streamlit.api_client.get_client", return_value=_mock_client).start()

from app.streamlit.session import SessionState
if "app_state" not in st.session_state:
    st.session_state.app_state = SessionState(api_connected=True)

import app.streamlit.components.pipeline_runner as _pr_mod
_original_start = _pr_mod._start_pipeline
def _noop_start(album_id, pipeline_name, steps, config):
    st.session_state["_captured_config"] = config
    return None
_pr_mod._start_pipeline = _noop_start

from app.streamlit.models import Album
album = Album(album_id="test", name="Test Album", source_directory="/test", total_images=50)

st.header("Pipeline Runner Test")
_pr_mod.render_pipeline_runner(album)
st.success("PAGE_RENDERED_OK")
'''


@pytest.fixture(scope="module")
def app_server(tmp_path_factory):
    """Start a Streamlit server for testing."""
    if not HAS_PLAYWRIGHT:
        pytest.skip("playwright not available")

    tmp = tmp_path_factory.mktemp("streamlit_test")
    app_file = tmp / "test_app.py"
    app_file.write_text(_TEST_APP)

    port = _find_free_port()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = f"http://localhost:{port}"

    # Wait for server to start (max 15s)
    for _ in range(30):
        try:
            with socket.create_connection(("localhost", port), timeout=0.5):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    else:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"Streamlit server failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}")

    yield url

    proc.kill()
    proc.wait()


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not available")
class ut_PipelineRunnerE2E:
    """End-to-end browser tests for pipeline runner."""

    def test_page_renders_without_error(self, app_server):
        """Page must render completely without Streamlit exceptions."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Check for Streamlit error elements
            errors = page.locator(".stException, .element-container .stAlert [data-testid='stExceptionMessage']")
            error_count = errors.count()
            if error_count > 0:
                error_texts = [errors.nth(i).text_content() for i in range(error_count)]
                pytest.fail(f"Streamlit errors on page: {error_texts}")

            # Check success marker
            assert page.locator("text=PAGE_RENDERED_OK").count() > 0, "Page did not render completely"

            browser.close()

    def test_advanced_config_expander_works(self, app_server):
        """Opening Advanced Configuration must not crash."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Find and click Advanced Configuration expander
            expander = page.locator("text=Advanced Configuration")
            if expander.count() > 0:
                expander.first.click()
                page.wait_for_timeout(1000)

            # No errors after expanding
            errors = page.locator(".stException")
            assert errors.count() == 0, f"Error after expanding: {errors.first.text_content() if errors.count() > 0 else ''}"

            browser.close()

    def test_profile_load_no_session_state_error(self, app_server):
        """Clicking Load profile must not cause session state error."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Open Advanced Configuration
            expander = page.locator("text=Advanced Configuration")
            if expander.count() > 0:
                expander.first.click()
                page.wait_for_timeout(1000)

            # Look for the Load button
            load_btn = page.locator("button:has-text('Load')")
            if load_btn.count() > 0:
                load_btn.first.click()
                page.wait_for_timeout(2000)

            # Must NOT have session_state error
            error_text = page.locator("text=cannot be modified after the widget").count()
            assert error_text == 0, "Session state modification error after profile load!"

            browser.close()

    def test_merge_checkbox_shows_merge_params(self, app_server):
        """Checking merge_enabled must show merge parameters without error."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Open Advanced Configuration
            expander = page.locator("text=Advanced Configuration")
            if expander.count() > 0:
                expander.first.click()
                page.wait_for_timeout(1000)

            # Find and check merge_enabled checkbox
            merge_checkbox = page.locator("text=merge_enabled").first
            if merge_checkbox.count() > 0:
                merge_checkbox.click()
                page.wait_for_timeout(2000)

                # Merge Parameters expander should now be visible
                merge_expander = page.locator("text=Merge Parameters")
                assert merge_expander.count() > 0, "Merge Parameters expander not shown after checking merge_enabled"

            # No errors
            errors = page.locator(".stException")
            assert errors.count() == 0, f"Error after merge checkbox: {errors.first.text_content() if errors.count() > 0 else ''}"

            browser.close()

    def test_no_errors_after_slider_interaction(self, app_server):
        """Moving a slider must not cause any errors."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Open Advanced Configuration
            expander = page.locator("text=Advanced Configuration")
            if expander.count() > 0:
                expander.first.click()
                page.wait_for_timeout(1000)

            # Find any slider thumb and drag it
            slider = page.locator('[data-testid="stSlider"]').first
            if slider.count() > 0:
                thumb = slider.locator('[role="slider"]')
                if thumb.count() > 0:
                    box = thumb.bounding_box()
                    if box:
                        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                        page.mouse.down()
                        page.mouse.move(box["x"] + 30, box["y"] + box["height"] / 2)
                        page.mouse.up()
                        page.wait_for_timeout(2000)

            # No errors after slider interaction
            errors = page.locator(".stException")
            assert errors.count() == 0, f"Error after slider: {errors.first.text_content() if errors.count() > 0 else ''}"

            browser.close()
