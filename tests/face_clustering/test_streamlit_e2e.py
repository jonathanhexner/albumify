"""Browser-based E2E tests using Playwright against D:\\Google_Germany.

Requirements:
    pip install playwright pytest-playwright
    playwright install chromium

Run with Streamlit already running:
    streamlit run app/face_clustering.py &
    pytest tests/face_clustering/test_streamlit_e2e.py -m e2e

These tests are excluded from default pytest run (marked @pytest.mark.e2e).
"""
import pytest
from pathlib import Path

GOOGLE_GERMANY = r"D:\Google_Germany"
APP_PORT = 8502  # override with --app-port pytest option if needed
APP_URL = f"http://localhost:{APP_PORT}"
PIPELINE_TIMEOUT_MS = 600_000  # 10 minutes for large album


@pytest.mark.e2e
def test_full_pipeline_no_errors(page):
    """Run pipeline on D:\\Google_Germany, verify completion without error banner."""
    page.goto(APP_URL)
    page.wait_for_load_state("networkidle")

    # Fill image directory
    page.get_by_label("Image directory").fill(GOOGLE_GERMANY)

    # Set output dir
    output_dir = str(Path(GOOGLE_GERMANY) / "clustering_output_test")
    page.get_by_label("Output directory").fill(output_dir)

    # Click Run
    page.get_by_role("button", name="Run Pipeline").click()

    # Wait for completion status
    page.wait_for_selector("text=Pipeline complete", timeout=PIPELINE_TIMEOUT_MS)

    # No error alert
    error_alerts = page.query_selector_all("[data-testid='stAlert'][kind='error']")
    assert len(error_alerts) == 0, "Error alert appeared during pipeline run"

    # Clusters metric > 0
    metrics = page.query_selector_all("[data-testid='metric-container']")
    assert len(metrics) >= 3, "Expected at least 3 metrics (faces, clusters, noise)"


@pytest.mark.e2e
def test_browse_tab_renders_faces(page):
    """Browse tab shows face images when results dir is loaded."""
    output_dir = str(Path(GOOGLE_GERMANY) / "clustering_output_test")
    page.goto(APP_URL)
    page.wait_for_load_state("networkidle")

    # Go to Browse tab
    page.get_by_role("tab", name="Browse Clusters").click()
    page.wait_for_load_state("networkidle")

    # Load results
    page.get_by_label("Results directory").fill(output_dir)
    page.get_by_role("button", name="Load").click()
    page.wait_for_load_state("networkidle")

    # Should show at least one cluster expander
    expanders = page.query_selector_all("[data-testid='stExpander']")
    assert len(expanders) > 0, "No cluster expanders found in Browse tab"


@pytest.mark.e2e
def test_debug_tab_loads(page):
    """Debug tab renders without exceptions after results are loaded."""
    page.goto(APP_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("tab", name="Debug").click()
    page.wait_for_load_state("networkidle")

    # No exception markers
    exceptions = page.query_selector_all("[data-testid='stException']")
    assert len(exceptions) == 0, "Exception rendered in Debug tab"
