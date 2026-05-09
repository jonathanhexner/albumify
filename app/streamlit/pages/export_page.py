"""Export page - Export selected images."""

import streamlit as st

from app.streamlit.session import get_session, add_notification
from app.streamlit.components.album_selector import render_album_selector
from app.streamlit.api_client import get_client
from app.streamlit.components.export_panel import render_export_panel


def render_export_page() -> None:
    """Render the Export page."""
    st.header("Export")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to export results.")
        return

    album = render_album_selector()

    if not album:
        st.info("Select an album to export results.")
        return

    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No results to export. Go to Configure & Run to run the pipeline first.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))
    num_selected = latest.get("num_selected", 0)
    total_filtered = latest.get("filtered_images", latest.get("total_images", 0))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected Images", num_selected)
    with col2:
        st.metric("All Processed", total_filtered)

    st.divider()

    render_export_panel(
        job_id, num_selected,
        on_export_complete=lambda p: add_notification(f"Exported to {p}", "success")
    )
