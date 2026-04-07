"""Shared test utilities and fixtures."""
from pathlib import Path


def get_test_data_dir() -> Path:
    """Return the absolute path to the test_data/ directory at the project root."""
    return Path(__file__).parent.parent / "test_data"
