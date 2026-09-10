"""Tests for face_popup comment persistence helpers (spec-015)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# face_popup lives inside app/face_clustering/ which uses local-module imports
# (not a proper package).  Import helpers directly without going through the
# Streamlit app entry point.
_APP_DIR = Path(__file__).parents[2] / "app" / "face_clustering"
sys.path.insert(0, str(_APP_DIR))

# Stub out streamlit (and transitive app-only imports) so the module can be
# imported without a live session.
#
# SIGHTING-119: these stubs go into the GLOBAL sys.modules, so if we don't undo
# them they poison every test collected after this file — e.g. tests/streamlit/
# imports real ``@st.cache_data`` and hits "module 'streamlit' has no attribute
# 'cache_data'". We snapshot the originals here and restore them in
# ``teardown_module`` so the stubs live only for this file's tests.
import types, unittest.mock as mock

_STUBBED = ["streamlit", "cache_helpers", "quality_panels",
            "face_cluster", "face_cluster.analysis_views", "face_cluster.pipeline"]
_saved_modules = {name: sys.modules.get(name) for name in _STUBBED}
_saved_syspath = list(sys.path)

_st_stub = types.ModuleType("streamlit")
_st_stub.session_state = {}
_st_stub.error   = mock.MagicMock()
_st_stub.warning = mock.MagicMock()
_st_stub.success = mock.MagicMock()
_st_stub.button  = mock.MagicMock(return_value=False)
_st_stub.dialog  = lambda *a, **kw: (lambda fn: fn)   # no-op decorator
sys.modules["streamlit"] = _st_stub

# Also stub transitive imports that require the full app environment
for _mod in [
    "face_cluster", "face_cluster.analysis_views", "face_cluster.pipeline",
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

_cache_stub = types.ModuleType("cache_helpers")
_cache_stub._crop_for_face  = mock.MagicMock(return_value=None)
_cache_stub._load_faces_df  = mock.MagicMock(return_value=None)
_cache_stub._load_manifest  = mock.MagicMock(return_value={})
sys.modules["cache_helpers"] = _cache_stub

_qp_stub = types.ModuleType("quality_panels")
_qp_stub._render_quality_report = mock.MagicMock()
sys.modules["quality_panels"] = _qp_stub

from face_popup import _load_comments, _save_comment, _COMMENT_MAX  # noqa: E402

# The stubs were only needed to IMPORT face_popup (it binds its own reference to
# them and keeps using it). Restore the real modules NOW — at import time, before
# pytest collects any other file — so the stubs don't leak into global state.
# (A teardown fixture would be too late: collection imports every test module up
# front, so the leak must be undone here, not after this file's tests run.)
for _name, _original in _saved_modules.items():
    if _original is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original
sys.path[:] = _saved_syspath


class ut_FacePopupComments:
    """Unit tests for comment persistence helpers."""

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert _load_comments(tmp_path) == {}

    def test_save_and_reload_single_comment(self, tmp_path):
        ok = _save_comment(tmp_path, face_id=42, text="falsely filtered")
        assert ok is True
        comments = _load_comments(tmp_path)
        assert comments["42"] == "falsely filtered"

    def test_multiple_face_ids_all_persisted(self, tmp_path):
        _save_comment(tmp_path, face_id=1, text="note one")
        _save_comment(tmp_path, face_id=2, text="note two")
        comments = _load_comments(tmp_path)
        assert comments["1"] == "note one"
        assert comments["2"] == "note two"

    def test_overwrite_existing_comment(self, tmp_path):
        _save_comment(tmp_path, face_id=7, text="first")
        _save_comment(tmp_path, face_id=7, text="updated")
        assert _load_comments(tmp_path)["7"] == "updated"

    def test_comment_too_long_returns_false(self, tmp_path):
        long_text = "x" * (_COMMENT_MAX + 1)
        ok = _save_comment(tmp_path, face_id=99, text=long_text)
        assert ok is False
        # File must NOT have been written with the oversized comment
        assert "99" not in _load_comments(tmp_path)

    def test_comment_at_max_length_is_accepted(self, tmp_path):
        text = "y" * _COMMENT_MAX
        ok = _save_comment(tmp_path, face_id=5, text=text)
        assert ok is True
        assert _load_comments(tmp_path)["5"] == text

    def test_load_corrupted_file_returns_empty(self, tmp_path):
        (tmp_path / "face_comments.json").write_text("not json", encoding="utf-8")
        assert _load_comments(tmp_path) == {}

    def test_key_uniqueness_scheme(self):
        """Button key scheme must not collide for same face_id in different contexts."""
        seen = set()
        for ctx in ["gp_cv_100_0_0", "gp_ex_100_0_0", "ca_af_3_0_0", "fa_same_7_0"]:
            key = f"popup_btn_{ctx}"
            assert key not in seen, f"Duplicate key: {key}"
            seen.add(key)
