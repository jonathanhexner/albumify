"""
General utility functions for analysis notebooks and scripts.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    This function finds the project root by looking for the directory 
    containing the 'sim_bench' package and common project markers.
    
    Returns:
        Path to the project root directory
    """
    # Start from this file's location
    current = Path(__file__).resolve()
    
    # Go up until we find the project root (directory containing sim_bench package)
    # This file is at: sim_bench/analysis/utils.py
    # So we need to go up 2 levels to get to project root
    project_root = current.parent.parent.parent
    
    # Verify we found the right directory by checking for key markers
    if (project_root / "sim_bench").exists() and (project_root / "artifacts").exists():
        return project_root
    
    # Fallback: try to find by looking for pyproject.toml or setup.py
    for parent in [current.parent.parent.parent, current.parent.parent]:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    
    # If all else fails, return the calculated root
    return project_root

