"""Pipeline runner component with progress display."""

import uuid
import streamlit as st
from typing import Optional, Dict, Any, List

from app.streamlit.api_client import get_client, ApiError
from app.streamlit.models import PipelineProgress, PipelineStatus, Album
from app.streamlit.session import get_session, update_pipeline_progress, set_pipeline_error, add_notification
from app.streamlit.config import get_config


# No more hardcoded pipelines - fetched from API
# See configs/pipeline.yaml for pipeline definitions

STEP_DISPLAY_NAMES = {
    "discover_images": "Discover Images",
    "detect_persons": "Detect People (YOLOv8)",
    "insightface_detect_faces": "Detect Faces (InsightFace)",
    "insightface_score_expression": "Score Expression",
    "insightface_score_eyes": "Score Eyes Open",
    "insightface_score_pose": "Score Face Pose",
    "score_iqa": "Score Image Quality",
    "score_ava": "Score Aesthetics",
    # Legacy MediaPipe steps (kept for compatibility)
    "detect_faces": "Detect Faces (MediaPipe)",
    "score_face_pose": "Score Face Pose (MediaPipe)",
    "score_face_eyes": "Score Eyes Open (MediaPipe)",
    "score_face_smile": "Score Smile (MediaPipe)",
    "filter_quality": "Filter Low Quality",
    "extract_scene_embedding": "Extract Scene Features",
    "cluster_scenes": "Cluster Similar Scenes",
    "extract_face_embeddings": "Extract Face Features",
    "cluster_people": "Identify People",
    "cluster_by_identity": "Cluster by Person",
    "select_best": "Select Best Photos",
}

PIPELINE_DISPLAY_NAMES = {
    "default_pipeline": "Full (InsightFace)",
    "minimal_pipeline": "Minimal (no faces)",
    "mediapipe_pipeline": "Legacy (MediaPipe)",
}


def _get_user_id() -> str:
    """Get or create a persistent user ID for this session."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id


@st.cache_data(ttl=60)
def _fetch_pipelines() -> Dict[str, List[str]]:
    """Fetch available pipelines from API (cached for 60 seconds)."""
    try:
        client = get_client()
        return client.get_available_pipelines()
    except Exception as e:
        st.warning(f"Could not fetch pipelines from API: {e}")
        # Return minimal fallback
        return {
            "default_pipeline": ["discover_images", "select_best"],
        }


def _load_user_settings() -> Dict[str, Any]:
    """Load user's saved settings from API."""
    try:
        client = get_client()
        user_id = _get_user_id()
        return client.get_user_config(user_id)
    except Exception:
        return {"selected_pipeline": "default_pipeline", "config": {}}


def _save_user_settings(selected_pipeline: str, config_overrides: Dict[str, Any]) -> None:
    """Save user's settings to API."""
    try:
        client = get_client()
        user_id = _get_user_id()
        client.save_user_config(user_id, selected_pipeline, config_overrides)
        add_notification("Settings saved!", "success")
    except Exception as e:
        add_notification(f"Failed to save settings: {e}", "error")


def render_pipeline_runner(album: Album) -> Optional[str]:
    """Render pipeline configuration and run button. Returns job ID if started."""
    state = get_session()

    st.subheader("Run Pipeline")

    # Fetch pipelines from API
    pipelines = _fetch_pipelines()
    pipeline_names = list(pipelines.keys())

    # Load user's saved settings
    user_settings = _load_user_settings()
    saved_pipeline = user_settings.get("selected_pipeline", "default_pipeline")
    saved_config = user_settings.get("config", {})

    # Pipeline selection
    default_index = pipeline_names.index(saved_pipeline) if saved_pipeline in pipeline_names else 0
    selected_pipeline = st.radio(
        "Pipeline Type",
        options=pipeline_names,
        format_func=lambda x: PIPELINE_DISPLAY_NAMES.get(x, x),
        index=default_index,
        horizontal=True,
        key="pipeline_type",
    )

    steps = pipelines.get(selected_pipeline, [])

    with st.expander("Pipeline Steps", expanded=False):
        for i, step in enumerate(steps, 1):
            st.write(f"{i}. {STEP_DISPLAY_NAMES.get(step, step)}")

    # Get saved config values (with defaults)
    saved_filter = saved_config.get("filter_quality", {})
    saved_select = saved_config.get("select_best", {})
    saved_detect = saved_config.get("detect_persons", {})
    saved_insightface = saved_config.get("insightface_detect_faces", {})
    saved_cluster_people = saved_config.get("cluster_people", {})
    saved_embedding = saved_config.get("extract_face_embeddings", {})

    with st.expander("Advanced Configuration", expanded=False):
        st.markdown("**Quality Filtering**")
        col1, col2 = st.columns(2)
        with col1:
            min_iqa = st.slider(
                "Min IQA Score", 0.0, 1.0,
                value=float(saved_filter.get("min_iqa_score", 0.2)),
                step=0.05, key="config_min_iqa",
                help="Minimum technical image quality (0-1)"
            )
        with col2:
            min_sharpness = st.slider(
                "Min Sharpness", 0.0, 1.0,
                value=float(saved_filter.get("min_sharpness", 0.1)),
                step=0.05, key="config_min_sharpness",
                help="Minimum image sharpness (0-1)"
            )

        st.markdown("**Person & Face Detection**")
        col1, col2 = st.columns(2)
        with col1:
            detection_confidence = st.slider(
                "Detection Confidence", 0.05, 0.5,
                value=float(saved_detect.get("confidence_threshold", 0.25)),
                step=0.05, key="config_det_conf",
                help="Confidence threshold for person and face detection"
            )
        with col2:
            min_face_size = st.slider(
                "Min Face Size (px)", 20, 100,
                value=int(saved_insightface.get("min_face_size", 50)),
                step=10, key="config_min_face_size",
                help="Minimum face size in pixels to be considered (smaller faces ignored)"
            )

        st.markdown("**Face Embedding**")
        col1, col2 = st.columns(2)
        with col1:
            embedding_backend = st.selectbox(
                "Embedding Model",
                options=["insightface", "custom"],
                index=0 if saved_embedding.get("backend", "insightface") == "insightface" else 1,
                key="config_embedding_backend",
                help="InsightFace: built-in model (rotation invariant). Custom: your trained ArcFace."
            )
        with col2:
            backend_info = "InsightFace w600k_r50" if embedding_backend == "insightface" else "arcface_resnet50.pt"
            st.info(f"Using: {backend_info}")

        st.markdown("**Selection**")
        col1, col2 = st.columns(2)
        with col1:
            max_per_cluster = st.number_input(
                "Max Images per Cluster", 1, 10,
                value=int(saved_select.get("max_images_per_cluster", 2)),
                key="config_max_per_cluster",
                help="Maximum photos to keep from each cluster"
            )
            min_score_threshold = st.slider(
                "Min Composite Score", 0.0, 1.0,
                value=float(saved_select.get("min_score_threshold", 0.4)),
                step=0.05, key="config_min_score",
                help="Minimum composite score (quality + penalties) to be selected"
            )
        with col2:
            st.info("Using new composite scoring: quality score + person penalties")

        st.markdown("**People Clustering**")
        col1, col2 = st.columns(2)
        with col1:
            people_method = st.selectbox(
                "Clustering Method",
                options=["hdbscan", "agglomerative"],
                index=0 if saved_cluster_people.get("method", "hdbscan") == "hdbscan" else 1,
                key="config_people_method",
                help="HDBSCAN: auto-determines clusters. Agglomerative: uses distance threshold."
            )
        with col2:
            if people_method == "hdbscan":
                people_min_cluster_size = st.slider(
                    "Min Faces per Person", 1, 5,
                    value=int(saved_cluster_people.get("min_cluster_size", 2)),
                    key="config_people_min_cluster",
                    help="Minimum face occurrences to form a person cluster"
                )
                people_distance_threshold = 0.5  # Not used for HDBSCAN
            else:
                people_distance_threshold = st.slider(
                    "Identity Distance Threshold", 0.3, 0.9,
                    value=float(saved_cluster_people.get("distance_threshold", 0.5)),
                    step=0.05, key="config_people_dist",
                    help="Lower = stricter (more clusters), Higher = lenient (fewer clusters)"
                )
                people_min_cluster_size = 2  # Not used for agglomerative

        # HDBSCAN merge epsilon - key for reducing over-segmentation
        if people_method == "hdbscan":
            cluster_merge_epsilon = st.slider(
                "Cluster Merge Distance", 0.0, 0.8,
                value=float(saved_cluster_people.get("cluster_selection_epsilon", 0.3)),
                step=0.05, key="config_cluster_epsilon",
                help="Higher = merge more clusters = fewer people (reduces over-segmentation)"
            )
        else:
            cluster_merge_epsilon = 0.3

        st.markdown("**Image Similarity & Quality**")
        col1, col2 = st.columns(2)
        with col1:
            duplicate_threshold = st.slider(
                "Dissimilarity Threshold", 0.80, 0.95,
                value=float(saved_select.get("dissimilarity_threshold", 0.85)),
                step=0.01, key="config_dup_thresh",
                help="Select images with similarity below this threshold"
            )
        with col2:
            siamese_config = saved_select.get("siamese", {})
            siamese_enabled = st.checkbox(
                "Enable Siamese Quality Refinement",
                value=bool(siamese_config.get("enabled", True)),
                key="config_siamese",
                help="Use Siamese CNN to refine quality scores for top candidates"
            )

    config = {
        "filter_quality": {"min_iqa_score": min_iqa, "min_sharpness": min_sharpness},
        # InsightFace detection config
        "insightface_detect_faces": {
            "detection_threshold": detection_confidence,
            "min_face_size": min_face_size,
        },
        # InsightFace scoring configs (use same min_face_size)
        "insightface_score_expression": {"min_face_size": min_face_size},
        "insightface_score_eyes": {"min_face_size": min_face_size},
        "insightface_score_pose": {"min_face_size": min_face_size},
        # Person detection config
        "detect_persons": {
            "confidence_threshold": detection_confidence,
        },
        # Face embedding extraction config
        "extract_face_embeddings": {
            "backend": embedding_backend,
            "checkpoint_path": "models/album_app/arcface_resnet50.pt",
            "device": "cpu",
            "model_name": "buffalo_l",
        },
        # People clustering config (global identity clustering for People tab)
        "cluster_people": {
            "method": people_method,
            "min_cluster_size": people_min_cluster_size,
            "min_samples": people_min_cluster_size,
            "distance_threshold": people_distance_threshold,
            "cluster_selection_epsilon": cluster_merge_epsilon,
        },
        # Identity sub-clustering config (within scene clusters)
        "cluster_by_identity": {
            "distance_threshold": people_distance_threshold,
        },
        # Select best config (new composite scoring)
        "select_best": {
            "max_images_per_cluster": max_per_cluster,
            "min_score_threshold": min_score_threshold,
            "dissimilarity_threshold": duplicate_threshold,
            "siamese": {"enabled": siamese_enabled},
        },
    }

    is_running = state.pipeline_status == PipelineStatus.RUNNING

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button(
            "Run Pipeline" if not is_running else "Running...",
            type="primary",
            disabled=is_running,
            use_container_width=True,
            key="run_pipeline_btn",
        ):
            return _start_pipeline(album.album_id, selected_pipeline, steps, config)

    with col2:
        if st.button(
            "Save Settings",
            disabled=is_running,
            use_container_width=True,
            key="save_settings_btn",
            help="Save your settings for next time",
        ):
            _save_user_settings(selected_pipeline, config)
            st.rerun()

    with col3:
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


def _start_pipeline(
    album_id: str,
    pipeline_name: str,
    steps: List[str],
    config: Dict[str, Any]
) -> Optional[str]:
    """Start the pipeline execution."""
    client = get_client()
    # Pass steps=None to let backend use the pipeline_name from config
    # But also pass the config overrides
    job_id = client.start_pipeline(album_id, steps=steps, config=config)

    update_pipeline_progress(PipelineProgress(
        status=PipelineStatus.RUNNING,
        current_step=steps[0] if steps else None,
        total_steps=len(steps),
    ))

    add_notification(f"Pipeline '{pipeline_name}' started (job: {job_id[:8]}...)", "info")
    st.session_state.current_job_id = job_id

    return job_id


def poll_pipeline_status(job_id: str) -> PipelineProgress:
    """Poll the pipeline status and update session."""
    client = get_client()
    progress = client.get_pipeline_status(job_id)
    update_pipeline_progress(progress)
    return progress
