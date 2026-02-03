"""Results page - Pipeline execution and results viewing."""

import time
import streamlit as st

from app.streamlit.config import get_config
from app.streamlit.session import get_session, add_notification, clear_pipeline_state
from app.streamlit.api_client import get_client
from app.streamlit.models import PipelineStatus, Album
from app.streamlit.components.album_selector import render_album_selector
from app.streamlit.components.pipeline_runner import render_pipeline_runner, render_pipeline_progress, poll_pipeline_status
from app.streamlit.components.gallery import render_image_gallery, render_cluster_gallery
from app.streamlit.components.metrics import render_pipeline_metrics, render_step_timings, render_image_metrics_table
from app.streamlit.components.people_browser import render_people_summary_row
from app.streamlit.components.export_panel import render_export_panel


def render_results_page() -> None:
    """Render the results page with pipeline and gallery."""
    st.header("Results")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to view results.")
        return

    album = render_album_selector()

    if not album:
        st.info("Select an album to view results or run the pipeline.")
        return

    st.divider()

    if state.pipeline_status == PipelineStatus.RUNNING:
        _render_running_pipeline()
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Run Pipeline", "View Results", "Metrics Table", "Export"])

    with tab1:
        job_id = render_pipeline_runner(album)
        if job_id:
            st.rerun()
        st.divider()
        _render_run_history(album)

    with tab2:
        _render_results_tab(album)

    with tab3:
        _render_metrics_table_tab(album)

    with tab4:
        _render_export_tab(album)


def _render_running_pipeline() -> None:
    """Render UI while pipeline is running."""
    render_pipeline_progress()

    job_id = st.session_state.get("current_job_id")
    if not job_id:
        st.warning("No active pipeline job found")
        return

    progress = poll_pipeline_status(job_id)

    if progress.status == PipelineStatus.COMPLETED:
        add_notification("Pipeline completed!", "success")
        st.session_state.pipeline_completed = True
        st.rerun()

    if progress.status == PipelineStatus.FAILED:
        add_notification("Pipeline failed", "error")
        st.rerun()

    if st.button("Cancel Pipeline"):
        clear_pipeline_state()
        add_notification("Pipeline cancelled", "warning")
        st.rerun()

    time.sleep(get_config().poll_interval_sec)
    st.rerun()


def _render_run_history(album: Album) -> None:
    """Render history of pipeline runs."""
    with st.expander("Previous Runs", expanded=False):
        client = get_client()
        results = client.list_results(album.album_id)

        if not results:
            st.info("No previous runs")
            return

        for result in results[:5]:
            job_id = result.get("job_id", result.get("id", ""))[:8]
            status = result.get("status", "unknown")
            total = result.get("total_images", 0)
            selected = result.get("num_selected", 0)
            duration = result.get("total_duration_ms", 0) / 1000

            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"Run `{job_id}...`")
            with col2:
                st.write(f"{selected}/{total} selected")
            with col3:
                st.write(f"{duration:.1f}s")
            with col4:
                st.write(status)


def _render_results_tab(album: Album) -> None:
    """Render the results viewing tab."""
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No pipeline results yet. Run the pipeline first.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))

    render_pipeline_metrics(latest)

    step_timings = latest.get("step_timings", {})
    if step_timings:
        render_step_timings(step_timings)

    st.divider()

    # People summary
    people = client.get_people(album.album_id)
    if people:
        render_people_summary_row(people)
        st.divider()

    _render_image_views(job_id)


def _render_image_views(job_id: str) -> None:
    """Render image viewing options and gallery."""
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        view_mode = st.radio("View Mode", ["Selected Only", "All Filtered", "By Cluster"], horizontal=True, key="results_view_mode")

    with col2:
        columns = st.slider("Columns", 2, 8, 4, key="gallery_cols")

    with col3:
        show_scores = st.checkbox("Scores", value=True, key="show_scores")

    st.divider()

    client = get_client()

    if view_mode == "Selected Only":
        images = client.get_selected_images(job_id)
        st.write(f"**{len(images)}** selected images")
        render_image_gallery(images, show_scores=show_scores, columns=columns)

    elif view_mode == "All Filtered":
        images = client.get_images(job_id)
        st.write(f"**{len(images)}** images passed quality filter")
        render_image_gallery(images, show_scores=show_scores, columns=columns)

    else:
        clusters = client.get_clusters(job_id)
        st.write(f"**{len(clusters)}** clusters")
        render_cluster_gallery(clusters, show_all_images=False)


def _render_metrics_table_tab(album: Album) -> None:
    """Render the per-image metrics table tab."""
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No pipeline results yet. Run the pipeline first.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))

    all_images = client.get_images(job_id)
    selected = client.get_selected_images(job_id)
    selected_paths = {img.path for img in selected}

    render_image_metrics_table(all_images, selected_paths)


def _render_export_tab(album: Album) -> None:
    """Render the export tab."""
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No results to export. Run the pipeline first.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))
    num_selected = latest.get("num_selected", 0)
    total_filtered = latest.get("filtered_images", latest.get("total_images", 0))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected Images", num_selected)
    with col2:
        st.metric("All Filtered", total_filtered)

    st.divider()

    render_export_panel(job_id, num_selected, on_export_complete=lambda p: add_notification(f"Exported to {p}", "success"))
