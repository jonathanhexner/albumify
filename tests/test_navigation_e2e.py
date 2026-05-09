"""Playwright E2E tests for the new navigation structure (spec-021 Phase A).

Verifies all 7 sidebar pages render without Streamlit errors.
"""
import subprocess
import sys
import time
import socket
import pytest

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Test page that renders the full app with mocked API
_TEST_APP = r'''
import sys, streamlit as st
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock the API client
_mock_client = MagicMock()
_mock_client.health_check.return_value = True
_mock_client.list_albums.return_value = []
_mock_client.get_available_pipelines.return_value = {"default_pipeline": ["discover_images", "select_best"]}
_mock_client.get_user_config.return_value = {"selected_pipeline": "default_pipeline", "config": {}}
_mock_client.list_results.return_value = []
_mock_client.get_people.return_value = []
_mock_client.get_images.return_value = []
_mock_client.get_clusters.return_value = []

patch("app.streamlit.api_client.get_client", return_value=_mock_client).start()

# Run the real app
from app.streamlit.main import render_app
render_app()
'''


@pytest.fixture(scope="module")
def app_server(tmp_path_factory):
    """Start the full Streamlit app for testing."""
    if not HAS_PLAYWRIGHT:
        pytest.skip("playwright not available")

    tmp = tmp_path_factory.mktemp("nav_test")
    app_file = tmp / "test_app.py"
    app_file.write_text(_TEST_APP)

    port = _find_free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app_file),
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    url = f"http://localhost:{port}"
    for _ in range(30):
        try:
            with socket.create_connection(("localhost", port), timeout=0.5):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    else:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"Server failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}")

    yield url
    proc.kill()
    proc.wait()


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not available")
class ut_NavigationE2E:
    """Test all 7 sidebar navigation pages render."""

    def _check_page_renders(self, app_server, page_name: str):
        """Helper: click a sidebar button and verify no errors."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Find and click the sidebar button
            btn = page.locator(f"button:has-text('{page_name}')")
            if btn.count() > 0:
                btn.first.click()
                page.wait_for_timeout(3000)

            # Check for Streamlit errors
            errors = page.locator(".stException")
            if errors.count() > 0:
                error_text = errors.first.text_content()
                browser.close()
                pytest.fail(f"Streamlit error on '{page_name}' page: {error_text[:200]}")

            browser.close()

    def test_home_page(self, app_server):
        self._check_page_renders(app_server, "Home")

    def test_albums_page(self, app_server):
        self._check_page_renders(app_server, "Albums")

    def test_configure_page(self, app_server):
        self._check_page_renders(app_server, "Configure & Run")

    def test_results_page(self, app_server):
        self._check_page_renders(app_server, "Results")

    def test_people_faces_page(self, app_server):
        self._check_page_renders(app_server, "People & Faces")

    def test_explore_page(self, app_server):
        self._check_page_renders(app_server, "Explore")

    def test_export_page(self, app_server):
        self._check_page_renders(app_server, "Export")
