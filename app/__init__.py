"""
Photo Organization Application

A production-grade Streamlit application for organizing photos using AI agents.

Architecture:
    - core/: Business logic (framework-agnostic)
    - ui/: Presentation layer (Streamlit-specific)
    - state/: State management
    - config/: Configuration and constants

Usage:
    streamlit run app/main.py
"""

__version__ = "2.0.0"
__author__ = "sim-bench"

from .ui import render_app

__all__ = ["render_app"]
