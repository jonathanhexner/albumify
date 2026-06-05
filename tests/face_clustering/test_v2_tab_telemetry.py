"""spec-068 — v2 tab render telemetry.

Proves the ``fc_app_v2.tabs`` logger emits a driver-agnostic ``tab.start`` /
``tab.done`` / ``tab.skipped`` line per tab, capturable without a browser.
We attach our own handler to the ``fc_app_v2.tabs`` logger (rather than rely
on ``caplog`` propagation, which the app's logging setup may alter) so the
assertions are deterministic.

AC3: seeded run -> ``tab.done`` for the analysis tabs (+ History).
AC4: no run    -> ``tab.skipped reason=no_run_loaded`` instead of done.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

# Self-contained synthetic run dir (same builder the smoke + repo tests use).
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
    _add_no_op_merge_round,
)

MAIN_PY = str(
    Path(__file__).resolve().parents[2] / "app" / "face_clustering_v2" / "main.py"
)


class _Capture(logging.Handler):
    """Collect ``fc_app_v2.tabs`` log records as plain message strings."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _run_main_capturing(query_params: dict[str, str]) -> list[str]:
    from streamlit.testing.v1 import AppTest

    cap = _Capture()
    logger = logging.getLogger("fc_app_v2.tabs")
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(cap)
    try:
        at = AppTest.from_file(MAIN_PY)
        for k, v in query_params.items():
            at.query_params[k] = v
        at.run(timeout=60)
        assert len(at.exception) == 0, (
            f"main.py raised: {[str(e.value) for e in at.exception]}"
        )
    finally:
        logger.removeHandler(cap)
        logger.setLevel(prev_level)
    return cap.lines


@pytest.fixture
def synthetic_run_dir(tmp_path: Path) -> Path:
    run_dir = _build_synthetic_run_dir(tmp_path)
    _add_no_op_merge_round(run_dir)
    return run_dir


@pytest.mark.slow
def test_seeded_run_emits_tab_done(synthetic_run_dir: Path):
    """AC3: with a run seeded, the analysis tabs + History log ``tab.done``."""
    lines = _run_main_capturing({"current_run_dir": str(synthetic_run_dir)})
    joined = "\n".join(lines)

    # Every analysis tab should have started against the seeded run dir.
    for tab in ("cluster_analysis", "merged_clusters", "quality", "history",
                "face_metrics"):
        assert any(f"name={tab}" in ln for ln in lines), (
            f"No telemetry line for tab '{tab}'. Captured:\n{joined}"
        )
    # Quality + History + Face Metrics reach tab.done (no selection gate).
    assert any("tab.done name=quality" in ln for ln in lines), joined
    assert any("tab.done name=history" in ln for ln in lines), joined
    assert any("tab.done name=face_metrics" in ln for ln in lines), joined
    # ASCII-only, single line per event.
    for ln in lines:
        assert ln.isascii(), f"Non-ASCII telemetry line: {ln!r}"
        assert "\n" not in ln, f"Multi-line telemetry: {ln!r}"


@pytest.mark.slow
def test_no_run_emits_tab_skipped():
    """AC4: with no run seeded, analysis tabs log ``tab.skipped`` with
    reason=no_run_loaded rather than tab.done."""
    lines = _run_main_capturing({})
    joined = "\n".join(lines)

    for tab in ("cluster_analysis", "face_analysis", "merged_clusters", "quality"):
        assert any(
            f"tab.skipped name={tab} reason=no_run_loaded" in ln for ln in lines
        ), f"Expected tab.skipped no_run_loaded for '{tab}'. Captured:\n{joined}"
    # No analysis tab should claim tab.done when nothing is loaded.
    assert not any("tab.done name=quality" in ln for ln in lines), joined
