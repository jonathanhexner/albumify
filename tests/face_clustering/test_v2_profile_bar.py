"""SIGHTING-136: the v2 profile bar loads into the tab's own knobs (Run: no prefix, Recluster: recluster_).

Headless Streamlit (AppTest) against a temporary profiles directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from face_cluster.fc_params import FCParams


def _app(profile_dir: str, key_prefix: str) -> None:
    import app.face_clustering_v2._profile_bar as bar
    from pathlib import Path
    bar.PROFILE_DIR = Path(profile_dir)
    bar.render_profile_bar(key_prefix=key_prefix)


def _load(tmp_path: Path, key_prefix: str) -> AppTest:
    FCParams(K=51, blur_min=73.3, merge_enabled=True).save(tmp_path / "p.json")
    at = AppTest.from_function(_app, args=(str(tmp_path), key_prefix), default_timeout=30).run()
    at.selectbox(key=f"{key_prefix}v2_profile_select").select("p.json").run()
    at.button(key=f"{key_prefix}v2_profile_load_btn").click().run()
    assert not at.exception, at.exception
    return at


@pytest.fixture(autouse=True)
def _keep_main_module():
    """AppTest swaps ``sys.modules['__main__']`` for its temporary script and leaves it there; later tests that
    start child processes (spawn re-imports ``__main__``) then crash in the child. Put the real one back, and the
    profile directory the app function points at the temporary one."""
    import app.face_clustering_v2._profile_bar as bar
    main, profile_dir = sys.modules["__main__"], bar.PROFILE_DIR
    yield
    sys.modules["__main__"], bar.PROFILE_DIR = main, profile_dir


class ut_V2ProfileBar:
    def test_recluster_load_fills_only_the_recluster_knobs(self, tmp_path):
        state = _load(tmp_path, "recluster_").session_state
        assert state["v2_recluster_K"] == 51
        assert state["v2_recluster_blur_min"] == 73.3
        assert state["v2_recluster_merge_enabled"] is True
        assert "v2_K" not in state                     # the Run tab's knobs are untouched
        assert state["recluster_v2_profile_last_loaded"] == "p.json"

    def test_run_tab_keys_are_unchanged(self, tmp_path):
        state = _load(tmp_path, "").session_state
        assert state["v2_K"] == 51 and state["v2_profile_last_loaded"] == "p.json"
