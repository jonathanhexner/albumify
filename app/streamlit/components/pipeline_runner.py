"""Pipeline runner component with progress display."""

import streamlit as st
from typing import Optional, Dict, Any, List

from app.streamlit.api_client import get_client, ApiError
from app.streamlit.models import PipelineProgress, PipelineStatus, Album
from app.streamlit.session import get_session, update_pipeline_progress, set_pipeline_error, add_notification
from app.streamlit.config import get_config


DEFAULT_PIPELINE = [
    "discover_images",
    "detect_faces",
    "score_iqa",
    "score_ava",
    "score_face_pose",
    "score_face_eyes",
    "score_face_smile",
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "extract_face_embeddings",
    "cluster_by_identity",
    "select_best",
]

MINIMAL_PIPELINE = [
    "discover_images",
    "score_iqa",
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "select_best",
]

STEP_DISPLAY_NAMES = {
    "discover_images": "Discover Images",
    "detect_faces": "Detect Faces",
    "score_iqa": "Score Image Quality",
    "score_ava": "Score Aesthetics",
    "score_face_pose": "Score Face Pose",
    "score_face_eyes": "Score Eyes Open",
    "score_face_smile": "Score Smile",
    "filter_quality": "Filter Low Quality",
    "extract_scene_embedding": "Extract Scene Features",
    "cluster_scenes": "Cluster Similar Scenes",
    "extract_face_embeddings": "Extract Face Features",
    "cluster_by_identity": "Cluster by Person",
    "select_best": "Select Best Photos",
}


def render_pipeline_runner(album: Album) -> Optional[str]:
    """Render pipeline configuration and run button. Returns job ID if started."""
    state = get_session()

    st.subheader("Run Pipeline")

    pipeline_type = st.radio(
        "Pipeline Type",
        options=["Full (with faces)", "Minimal (no faces)"],
        horizontal=True,
        key="pipeline_type",
    )

    steps = DEFAULT_PIPELINE if pipeline_type == "Full (with faces)" else MINIMAL_PIPELINE

    with st.expander("Pipeline Steps", expanded=False):
        for i, step in enumerate(steps, 1):
            st.write(f"{i}. {STEP_DISPLAY_NAMES.get(step, step)}")

    with st.expander("Advanced Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            min_iqa = st.slider("Min IQA Score", 0.0, 1.0, 0.2, 0.05, key="config_min_iqa")
            max_per_cluster = st.number_input("Max Images per Cluster", 1, 10, 2, key="config_max_per_cluster")

        with col2:
            min_score_threshold = st.slider("Min Score Threshold", 0.0, 1.0, 0.3, 0.05, key="config_min_score")
            tiebreaker_threshold = st.slider("Tiebreaker Threshold", 0.01, 0.2, 0.05, 0.01, key="config_tiebreaker")

    config = {
        "filter_quality": {"min_iqa_score": min_iqa, "min_sharpness": 0.1},
        "select_best": {
            "max_images_per_cluster": max_per_cluster,
            "min_score_threshold": min_score_threshold,
            "tiebreaker_threshold": tiebreaker_threshold,
        },
    }

    is_running = state.pipeline_status == PipelineStatus.RUNNING

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(
            "Run Pipeline" if not is_running else "Running...",
            type="primary",
            disabled=is_running,
            use_container_width=True,
            key="run_pipeline_btn",
        ):
            return _start_pipeline(album.album_id, steps, config)

    with col2:
        if is_running and st.button("Cancel", use_container_width=True, key="cancel_pipeline_btn"):
            from app.streamlit.session import clear_pipeline_state
            clear_pipeline_state()
            add_notification("Pipeline cancelled", "warning")
            st.rerun()

    return None


def render_pipeline_progress() -> None:
    """Render pipeline progress display."""
    state = get_session()
    progress = state.pipeline_progress

    if not progress:
        return

    st.subheader("Pipeline Progress")

    if progress.total_steps > 0:
        completed = len(progress.completed_steps)
        st.progress(completed / progress.total_steps, text=f"Step {completed}/{progress.total_steps}")

    if progress.current_step:
        display_name = STEP_DISPLAY_NAMES.get(progress.current_step, progress.current_step)
        st.write(f"**Current:** {display_name}")

        if progress.current_step_progress > 0:
            st.progress(progress.current_step_progress, text=progress.current_step_message or "")

    if progress.status == PipelineStatus.COMPLETED:
        st.success("Pipeline completed successfully!")
    elif progress.status == PipelineStatus.FAILED:
        st.error("Pipeline failed")
        if state.pipeline_error:
            st.error(state.pipeline_error)

    if progress.completed_steps:
        with st.expander("Completed Steps", expanded=False):
            for step in progress.completed_steps:
                st.write(f"✓ {STEP_DISPLAY_NAMES.get(step, step)}")


def render_step_list(steps: List[str], completed: List[str], current: Optional[str] = None) -> None:
    """Render a visual list of pipeline steps with status."""
    for step in steps:
        display_name = STEP_DISPLAY_NAMES.get(step, step)

        if step in completed:
            st.markdown(f'<div class="workflow-step step-completed">✓ {display_name}</div>', unsafe_allow_html=True)
        elif step == current:
            st.markdown(f'<div class="workflow-step step-running">⟳ {display_name}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="workflow-step step-pending">○ {display_name}</div>', unsafe_allow_html=True)


def _start_pipeline(album_id: str, steps: List[str], config: Dict[str, Any]) -> Optional[str]:
    """Start the pipeline execution."""
    client = get_client()
    job_id = client.start_pipeline(album_id, steps, config)

    update_pipeline_progress(PipelineProgress(
        status=PipelineStatus.RUNNING,
        current_step=steps[0] if steps else None,
        total_steps=len(steps),
    ))

    add_notification(f"Pipeline started (job: {job_id[:8]}...)", "info")
    st.session_state.current_job_id = job_id

    return job_id


def poll_pipeline_status(job_id: str) -> PipelineProgress:
    """Poll the pipeline status and update session."""
    client = get_client()
    progress = client.get_pipeline_status(job_id)
    update_pipeline_progress(progress)
    return progress
