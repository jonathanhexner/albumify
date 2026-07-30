"""Configure & Run page - Pipeline configuration and execution."""

import streamlit as st

from app.streamlit.session import get_session
from app.streamlit.components.album_selector import render_album_selector
from app.streamlit.components.pipeline_runner import render_pipeline_runner, render_pipeline_progress, poll_pipeline_status, STEP_DISPLAY_NAMES
from app.streamlit.models import PipelineStatus
from app.streamlit.session import add_notification, clear_pipeline_state
from app.streamlit.config import get_config


def render_configure_page() -> None:
    """Render the Configure & Run page."""
    st.header("Configure & Run")

    state = get_session()

    if not state.api_connected:
        st.warning("Connect to API to configure and run pipelines.")
        return

    album = render_album_selector()

    if not album:
        st.info("Select an album to configure and run the pipeline.")
        return

    st.divider()

    if state.pipeline_status == PipelineStatus.RUNNING:
        _render_running_pipeline()
        return

    # Pipeline configuration and run button
    job_id = render_pipeline_runner(album)
    if job_id:
        st.rerun()

    st.divider()

    # Run history
    _render_run_history(album)


def _render_running_pipeline() -> None:
    """Render UI while pipeline is running.

    Uses @st.fragment(run_every=2) to poll without crashing the frontend.
    Only the progress fragment re-renders every 2 seconds — the rest of the page stays stable.
    """
    job_id = st.session_state.get("current_job_id")
    if not job_id:
        st.warning("No active pipeline job found")
        return

    st.subheader("Pipeline Running...")

    # Cancel button (outside fragment so it persists)
    if st.button("Cancel Pipeline", type="secondary"):
        clear_pipeline_state()
        add_notification("Pipeline cancelled", "warning")
        st.rerun()

    # Progress fragment — auto-refreshes every 2 seconds
    _pipeline_progress_fragment(job_id)


@st.fragment(run_every=2)
def _pipeline_progress_fragment(job_id: str) -> None:
    """Auto-refreshing pipeline progress display. Polls every 2 seconds."""
    from app.streamlit.api_client import ApiError
    try:
        progress = poll_pipeline_status(job_id)
    except ApiError:
        # SIGHTING-110: a heavy pipeline step can block the API longer than the
        # status read timeout. The run is still going — show a note and let the
        # fragment retry on its next 2s tick instead of crashing the page.
        st.info("Still working - a heavy step is busy; status check will retry shortly.")
        return

    # Check for completion/failure
    if progress.status == PipelineStatus.COMPLETED:
        add_notification("Pipeline completed!", "success")
        st.session_state.pipeline_completed = True
        st.rerun(scope="app")
        return

    if progress.status == PipelineStatus.FAILED:
        add_notification("Pipeline failed", "error")
        st.rerun(scope="app")
        return

    # Render step-by-step progress
    completed_steps = getattr(progress, 'completed_steps', None) or []
    total_steps = getattr(progress, 'total_steps', None) or 0
    current_step = progress.current_step

    if total_steps > 0:
        n_done = len(completed_steps)
        st.progress(n_done / total_steps, text=f"Step {n_done}/{total_steps}")

    # Step list
    for step_info in completed_steps:
        step_name = step_info.get("step", "?")
        duration = step_info.get("duration_ms", 0) / 1000
        status = step_info.get("status", "completed")
        display = STEP_DISPLAY_NAMES.get(step_name, step_name)
        error = step_info.get("error")

        if status == "completed":
            st.write(f":green[OK] {display} ({duration:.1f}s)")
        elif status == "failed":
            st.write(f":red[FAIL] {display} ({duration:.1f}s)")
            if error:
                st.error(error)

    if current_step:
        display = STEP_DISPLAY_NAMES.get(current_step, current_step)
        st.write(f":orange[>>] {display} ...")

    # Show elapsed time
    if progress.started_at:
        from datetime import datetime
        try:
            started = progress.started_at if isinstance(progress.started_at, datetime) else datetime.fromisoformat(str(progress.started_at))
            elapsed = (datetime.utcnow() - started).total_seconds()
            st.caption(f"Elapsed: {elapsed:.0f}s")
        except Exception:
            pass


def _render_run_history(album) -> None:
    """Render run history table."""
    from app.streamlit.api_client import get_client

    st.subheader("Run History")
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No pipeline runs yet. Configure parameters above and run the pipeline.")
        return

    for result in results[:10]:
        job_id = result.get("job_id", result.get("id", ""))[:8]
        status = result.get("status", "unknown")
        total = result.get("total_images", 0)
        selected = result.get("num_selected", 0)
        num_people = result.get("num_people", "-")
        duration = result.get("total_duration_ms", 0) / 1000

        status_color = {"completed": "green", "failed": "red", "running": "orange"}.get(status, "gray")

        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        with col1:
            st.markdown(f":{status_color}[{status.upper()}]")
        with col2:
            st.write(f"`{job_id}...`")
        with col3:
            st.write(f"{num_people} people")
        with col4:
            st.write(f"{selected}/{total} selected")
        with col5:
            st.write(f"{duration:.1f}s")
        with col6:
            if st.button("View", key=f"hist_view_{job_id}"):
                st.session_state.current_page = "results"
                st.rerun()
