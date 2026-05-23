"""spec-041 follow-up — every entry point calls ``setup_logging`` with the right surface name.

Adding a new app or script that bypasses ``sim_bench.logging_setup`` is
a regression. This test reads each entry point's source and asserts:

1. It imports ``setup_logging`` (from anywhere — either directly from
   ``sim_bench.logging_setup`` or via the back-compat shim).
2. It calls ``setup_logging("<expected_surface>")`` with the matching
   surface literal.

The test is intentionally lo-fi (string match) — it doesn't import the
modules, since v2/legacy mains import Streamlit at the top.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical mapping from entry-point file → surface name.
# Adding a new top-level app/script means adding a row here.
ENTRY_POINTS: dict[str, str] = {
    "sim_bench/api/main.py":           "api",
    "app/face_clustering_v2/main.py":  "fc_app_v2",
    "app/face_clustering/main.py":     "fc_app_legacy",
    "scripts/run_v2.py":               "cli_run_v2",
}


@pytest.mark.parametrize("rel_path,surface", sorted(ENTRY_POINTS.items()))
def test_entry_point_calls_setup_logging_with_correct_surface(rel_path: str, surface: str):
    path = REPO_ROOT / rel_path
    assert path.exists(), f"entry point file missing: {rel_path}"
    src = path.read_text(encoding="utf-8")
    assert "setup_logging" in src, (
        f"{rel_path} does not call setup_logging — every surface must "
        "configure logging via sim_bench.logging_setup. Add:\n"
        "    from sim_bench.logging_setup import setup_logging\n"
        f"    setup_logging({surface!r})"
    )
    # Match either single or double quoted surface literal.
    needle_dq = f'setup_logging("{surface}")'
    needle_sq = f"setup_logging('{surface}')"
    assert (needle_dq in src) or (needle_sq in src), (
        f"{rel_path} calls setup_logging but not with surface={surface!r}. "
        f"Expected one of: {needle_dq} | {needle_sq}"
    )


def test_no_entry_point_uses_logging_basicConfig_directly():
    """Local ``logging.basicConfig`` bypasses the standard. Catch it
    everywhere except the shared module itself and Python tests."""
    for rel_path in ENTRY_POINTS:
        path = REPO_ROOT / rel_path
        src = path.read_text(encoding="utf-8")
        assert "logging.basicConfig" not in src, (
            f"{rel_path} calls logging.basicConfig directly — use "
            "sim_bench.logging_setup.setup_logging instead so the log "
            "format / dir layout stay aligned across surfaces."
        )


def test_logging_setup_module_exists_and_exports_expected_api():
    from sim_bench import logging_setup
    for name in ("setup_logging", "get_log_dir", "get_logger"):
        assert hasattr(logging_setup, name), (
            f"sim_bench.logging_setup missing {name!r}"
        )
