"""Home page - Welcome screen and quick start."""

import streamlit as st

from app.streamlit.config import get_config
from app.streamlit.session import get_session, set_api_connected
from app.streamlit.api_client import get_client
from app.streamlit.components.album_selector import render_album_selector


def check_api_connection() -> bool:
    """Check if API is available."""
    client = get_client()
    connected = client.health_check()
    set_api_connected(connected)
    return connected


def render_home_page() -> None:
    """Render the home page."""
    state = get_session()

    st.header("Welcome to Album Organizer")

    if not state.api_connected:
        _render_disconnected_state()
        return

    _render_welcome_content()
    st.divider()

    st.subheader("Select Album")
    album = render_album_selector()

    if album:
        _render_album_actions(album)


def _render_disconnected_state() -> None:
    """Render UI when API is not connected."""
    st.warning("Please connect to the API server to get started.")
    st.info(f"Make sure the FastAPI backend is running at {get_config().api_base_url}")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Retry Connection", type="primary"):
            if check_api_connection():
                st.rerun()

    with st.expander("Troubleshooting", expanded=False):
        st.markdown("""
        **Common issues:**

        1. **Backend not running** - Start the FastAPI server:
           ```bash
           python -m uvicorn sim_bench.api.main:app --reload
           ```

        2. **Wrong port** - Check that the API is running on port 8000

        3. **Firewall blocking** - Ensure localhost connections are allowed
        """)


def _render_welcome_content() -> None:
    """Render the welcome content and instructions."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Album Organizer** helps you curate your photo albums by:
        - Detecting and grouping similar photos
        - Identifying people in your photos
        - Selecting the best photos from each group
        - Filtering near-duplicates using AI

        ### Quick Start

        1. Go to **Albums** to create or select an album
        2. Run the **Pipeline** to process your photos
        3. View **Results** to see selected photos
        4. Browse by **People** to see photos of specific people
        5. **Export** your curated selection
        """)

    with col2:
        st.markdown("""
        ### Features

        - **Face Detection** - Find faces in photos
        - **Scene Clustering** - Group similar scenes
        - **Quality Scoring** - Rate image quality
        - **Smart Selection** - Pick the best shots
        - **People Browser** - Filter by person
        """)


def _render_album_actions(album) -> None:
    """Render action buttons for selected album."""
    st.success(f"Selected: **{album.name}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Photos", album.total_images)
    with col2:
        st.metric("Selected", album.selected_images)
    with col3:
        status = album.status.value if hasattr(album.status, 'value') else str(album.status)
        st.metric("Status", status.capitalize())

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Pipeline", type="primary", use_container_width=True):
            st.session_state.current_page = "results"
            st.rerun()

    with col2:
        if st.button("View Results", use_container_width=True):
            st.session_state.current_page = "results"
            st.rerun()

    with col3:
        if st.button("Browse People", use_container_width=True):
            st.session_state.current_page = "people"
            st.rerun()
