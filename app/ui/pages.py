"""Page orchestration - coordinates UI components."""

import streamlit as st

from ..config import Settings
from ..state import StateManager
from .components import (
    render_sidebar,
    render_chat_interface,
    render_welcome_screen
)
from .styles import get_custom_css


def render_app() -> None:
    """
    Main app rendering orchestration.

    This is the single entry point for the entire UI.
    It coordinates all components and handles the page flow.
    """
    # Configure page
    st.set_page_config(
        page_title=Settings.APP.title,
        page_icon=Settings.APP.page_icon,
        layout=Settings.APP.layout
    )

    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Initialize state
    StateManager.initialize()

    # Render header
    _render_header()

    # Render sidebar
    render_sidebar()

    # Render main content based on state
    if StateManager.is_ready():
        # App is ready - show chat interface
        render_chat_interface()
    else:
        # App not ready - show welcome screen
        render_welcome_screen()


def _render_header() -> None:
    """Render app header."""
    st.title(Settings.APP.title)
    st.caption(Settings.APP.subtitle)
