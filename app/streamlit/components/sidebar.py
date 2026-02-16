"""Sidebar component with navigation and configuration."""

import streamlit as st
from typing import Callable, Optional

from app.streamlit.config import get_config
from app.streamlit.session import get_session
from app.streamlit.api_client import get_client


def render_sidebar(on_page_change: Optional[Callable[[str], None]] = None) -> str:
    """Render the sidebar with navigation and status."""
    config = get_config()
    state = get_session()

    with st.sidebar:
        st.title("Album Organizer")

        # API Status
        _render_api_status(state.api_connected)
        st.divider()

        # Navigation
        selected_page = _render_navigation(on_page_change)
        st.divider()

        # Current album info
        if state.current_album:
            _render_current_album_info(state.current_album)
            st.divider()

        # Settings
        _render_settings(config)

    return selected_page


def _render_api_status(connected: bool) -> None:
    """Render API connection status."""
    st.subheader("Status")

    col1, col2 = st.columns([3, 1])
    with col1:
        if connected:
            st.success("API Connected")
        else:
            st.error("API Disconnected")

    with col2:
        if st.button("↻", key="refresh_connection", help="Refresh connection"):
            client = get_client()
            if client.health_check():
                st.session_state.api_connected = True
                st.rerun()


def _render_navigation(on_page_change: Optional[Callable[[str], None]] = None) -> str:
    """Render navigation buttons."""
    st.subheader("Navigation")

    pages = [
        ("Home", "home", "🏠"),
        ("Albums", "albums", "📁"),
        ("Results", "results", "📊"),
        ("People", "people", "👥"),
        ("Faces", "faces", "🎭"),
        ("Debug", "debug", "🔍"),
    ]

    current_page = st.session_state.get("current_page", "home")

    for label, page_id, icon in pages:
        is_current = current_page == page_id
        button_type = "primary" if is_current else "secondary"

        if st.button(f"{icon} {label}", key=f"nav_{page_id}", use_container_width=True, type=button_type):
            st.session_state.current_page = page_id
            if on_page_change:
                on_page_change(page_id)
            st.rerun()

    return current_page


def _render_current_album_info(album) -> None:
    """Render current album information."""
    st.subheader("Current Album")
    st.write(f"**{album.name}**")
    st.caption(album.source_directory)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Photos", album.total_images)
    with col2:
        st.metric("Selected", album.selected_images)

    status = album.status.value if hasattr(album.status, 'value') else str(album.status)
    status_colors = {"idle": "gray", "running": "orange", "completed": "green", "failed": "red"}
    color = status_colors.get(status, "gray")
    st.markdown(f"Status: :{color}[{status.upper()}]")


def _render_settings(config) -> None:
    """Render settings section."""
    with st.expander("Settings", expanded=False):
        st.caption(f"**API URL:** {config.api_base_url}")
        st.caption(f"**Poll Interval:** {config.poll_interval_sec}s")
        st.caption(f"**Images/Row:** {config.images_per_row}")

        new_per_row = st.slider(
            "Images per row",
            min_value=2,
            max_value=8,
            value=config.images_per_row,
            key="settings_images_per_row",
        )
        if new_per_row != config.images_per_row:
            config.images_per_row = new_per_row
