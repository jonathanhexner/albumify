"""
Main entry point for the Album Organizer Streamlit app.

This is a pure frontend that communicates with the FastAPI backend via HTTP.
Run with: streamlit run app/streamlit/main.py
"""

import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from app.streamlit.config import get_config
from app.streamlit.session import get_session, pop_notifications, set_api_connected, set_current_album
from app.streamlit.api_client import get_client
from app.streamlit.components.sidebar import render_sidebar
from app.streamlit.pages.home import render_home_page
from app.streamlit.pages.albums import render_albums_page
from app.streamlit.pages.results import render_results_page
from app.streamlit.pages.people import render_people_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CUSTOM_CSS = """
<style>
    .main { padding: 1rem; }
    .status-ready { color: #4caf50; font-weight: 600; }
    .status-error { color: #f44336; font-weight: 600; }
    .status-warning { color: #ff9800; font-weight: 600; }
    .workflow-step { padding: 0.75rem; margin: 0.5rem 0; border-radius: 0.25rem; font-family: monospace; font-size: 0.9rem; }
    .step-completed { background-color: #c8e6c9; color: #2e7d32; }
    .step-running { background-color: #fff9c4; color: #f57f17; }
    .step-pending { background-color: #eeeeee; color: #757575; }
    .step-failed { background-color: #ffcdd2; color: #c62828; }
    .sidebar-section { margin-bottom: 2rem; }
    .image-card { padding: 0.5rem; margin: 0.25rem; background-color: #fafafa; border-radius: 0.25rem; border: 1px solid #e0e0e0; transition: all 0.2s ease; }
    .image-card:hover { border-color: #2196f3; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .image-card.selected { border-color: #4caf50; border-width: 2px; }
    .info-box { padding: 1rem; border-radius: 0.5rem; background-color: #e3f2fd; border-left: 4px solid #2196f3; margin: 1rem 0; }
    .person-card { text-align: center; padding: 0.5rem; cursor: pointer; transition: transform 0.2s ease; }
    .person-card:hover { transform: scale(1.05); }
    .person-card img { border-radius: 50%; width: 80px; height: 80px; object-fit: cover; border: 2px solid #e0e0e0; }
    .metric-card { background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #1976d2; }
    .metric-label { color: #757575; font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
    .stProgress > div > div { background-color: #4caf50; }
</style>
"""


def check_api_connection() -> bool:
    """Check if API is available."""
    client = get_client()
    connected = client.health_check()
    set_api_connected(connected)
    return connected


def load_persisted_state() -> None:
    """Load persisted state from API on startup."""
    state = get_session()

    if not state.api_connected:
        return

    # Load albums and auto-select the first one if none selected
    if not state.current_album:
        client = get_client()
        albums = client.list_albums()
        if albums:
            set_current_album(albums[0])
            logger.info(f"Auto-selected album: {albums[0].name}")


def show_notifications() -> None:
    """Display any pending notifications."""
    for notif in pop_notifications():
        msg, type_ = notif["message"], notif["type"]
        {"success": st.success, "error": st.error, "warning": st.warning}.get(type_, st.info)(msg)


def render_app() -> None:
    """Main app rendering orchestration."""
    config = get_config()

    st.set_page_config(page_title=config.page_title, page_icon=config.page_icon, layout=config.layout)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    if "initialized" not in st.session_state:
        check_api_connection()
        load_persisted_state()
        st.session_state.initialized = True

    show_notifications()
    render_sidebar()

    page = st.session_state.current_page
    pages = {"home": render_home_page, "albums": render_albums_page, "results": render_results_page, "people": render_people_page}
    pages.get(page, render_home_page)()


def main() -> None:
    """Application entry point."""
    render_app()


if __name__ == "__main__":
    main()
