"""Shared fixtures for the Albumify UI tests."""
import sys

import pytest


@pytest.fixture(autouse=True)
def keep_main_module():
    """Streamlit's ``AppTest`` replaces ``sys.modules['__main__']`` with its generated script and leaves it there.
    A 'spawn' child process started later re-runs that script and crashes (``__args`` undefined), which failed the
    spec-104 step-isolation tests whenever they ran after ``test_gallery_keys.py`` (2026-11-14)."""
    main = sys.modules.get("__main__")
    yield
    if main is not None:
        sys.modules["__main__"] = main
