"""
Main entry point for the Photo Organization Streamlit app.

This file should be minimal - just imports and runs the app.
All business logic is in core/, UI is in ui/, state in state/.

Run with: streamlit run app/main.py
"""

import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging before anything else
from sim_bench.config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

# Import and run app
from app.ui import render_app


def main() -> None:
    """Application entry point."""
    try:
        render_app()
    except Exception as e:
        logger.exception("Fatal error in application")
        import streamlit as st
        st.error(f"❌ Fatal Error: {str(e)}")
        st.error("Please check the logs for details.")


if __name__ == "__main__":
    main()
