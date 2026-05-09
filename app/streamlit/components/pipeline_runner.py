"""Pipeline runner component with progress display."""

import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st

from app.streamlit.api_client import get_client, ApiError
from app.streamlit.models import PipelineProgress, PipelineStatus, Album
from app.streamlit.session import get_session, update_pipeline_progress, set_pipeline_error, add_notification
from app.streamlit.config import get_config
from face_cluster.profile_store import ProfileStore

# Make app/shared/ importable for shared UI controls
_shared_dir = str(Path(__file__).resolve().parents[2] / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)
from merge_controls import render_merge_params as _render_merge_params

# Parameter keys shared with face_clustering app (same as _RC_PARAM_KEYS in state.py).
# Using rc_* prefix so profiles saved in either app work in both.
_RC_PARAM_KEYS: frozenset = frozenset({
    "rc_K", "rc_dist", "rc_min_cluster",
    "rc_split", "rc_merge", "rc_attach",
    "rc_merge_use_adaptive", "rc_merge_exemplar_pct", "rc_merge_global_pct",
    "rc_merge_alpha", "rc_merge_beta", "rc_merge_candidate", "rc_merge_exemplar_thresh",
    "rc_merge_support_frac", "rc_merge_support_min", "rc_merge_margin", "rc_merge_diameter",
    "rc_merge_use_cross_gate", "rc_merge_cross_thresh", "rc_merge_cross_max_size", "rc_merge_support_unique",
})


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


@st.cache_data(ttl=60)
def _load_user_settings(_user_id: str) -> Dict[str, Any]:
    """Load user's saved settings from API (cached 60s to avoid latency on fragment reruns)."""
    try:
        client = get_client()
        return client.get_user_config(_user_id)
    except Exception:
        return {"selected_pipeline": "default_pipeline", "config": {}}


def _save_user_settings(selected_pipeline: str, config_overrides: Dict[str, Any]) -> None:
    """Save user's settings to API."""
    try:
        client = get_client()
        user_id = _get_user_id()
        client.save_user_config(user_id, selected_pipeline, config_overrides)
        _load_user_settings.clear()  # Invalidate cached settings
        add_notification("Settings saved!", "success")
    except Exception as e:
        add_notification(f"Failed to save settings: {e}", "error")


def _render_profile_bar() -> None:
    """Profile load/save bar — shared with Face Clustering App (~/.sim_bench/profiles/)."""
    store = ProfileStore()
    names = store.list_names()

    col_sel, col_load, col_name, col_save, col_default = st.columns([2, 1, 2, 1, 1])
    with col_sel:
        options = ["(none)"] + names
        st.selectbox("Load profile", options, key="prf_select", label_visibility="collapsed")
    with col_load:
        if st.button("Load", key="prf_load"):
            selected = st.session_state.get("prf_select", "(none)")
            if selected != "(none)":
                params = store.load(selected)
                for k, v in params.items():
                    if k in _RC_PARAM_KEYS:
                        st.session_state[k] = v
                add_notification(f"Loaded profile '{selected}'", "info")
                st.rerun(scope="app")
    with col_name:
        st.text_input("Profile name", key="prf_name", label_visibility="collapsed", placeholder="profile name")
    with col_save:
        if st.button("Save", key="prf_save"):
            name = st.session_state.get("prf_name", "").strip()
            if name:
                params = {k: st.session_state[k] for k in _RC_PARAM_KEYS if k in st.session_state}
                store.save(name, params)
                add_notification(f"Saved profile '{name}'", "success")
    with col_default:
        if st.button("Default", key="prf_default", help="Save current params as default profile"):
            params = {k: st.session_state[k] for k in _RC_PARAM_KEYS if k in st.session_state}
            store.save("default", params)
            add_notification("Saved as default profile", "success")


def render_pipeline_runner(album: Album) -> Optional[str]:
    """Render pipeline configuration and run button. Returns job ID if started."""
    return _render_pipeline_config(album)


def _render_pipeline_config(album: Album) -> Optional[str]:
    """Flat pipeline config UI — NO expanders, NO fragments. All params visible."""
    state = get_session()

    # Fetch pipelines from API
    pipelines = _fetch_pipelines()
    pipeline_names = list(pipelines.keys())

    # Load user's saved settings (cached)
    user_settings = _load_user_settings(_get_user_id())
    saved_pipeline = user_settings.get("selected_pipeline", "default_pipeline")
    saved_config = user_settings.get("config", {})

    # Pipeline selection
    default_index = pipeline_names.index(saved_pipeline) if saved_pipeline in pipeline_names else 0
    selected_pipeline = st.radio(
        "Pipeline",
        options=pipeline_names,
        format_func=lambda x: PIPELINE_DISPLAY_NAMES.get(x, x),
        index=default_index,
        horizontal=True,
        key="pipeline_type",
    )

    steps = pipelines.get(selected_pipeline, [])

    # Get saved config values
    saved_filter = saved_config.get("filter_quality", {})
    saved_select = saved_config.get("select_best", {})
    saved_detect = saved_config.get("detect_persons", {})
    saved_insightface = saved_config.get("insightface_detect_faces", {})
    saved_cluster_people = saved_config.get("cluster_people", {})
    saved_embedding = saved_config.get("extract_face_embeddings", {})

    # --- Profile bar (full-width, above config grid) ---
    # Show profile bar when face_cluster_knn is selected (check session state for live value)
    current_method = st.session_state.get("config_people_method", saved_cluster_people.get("method", "face_cluster_knn"))
    if current_method == "face_cluster_knn":
        _render_profile_bar()

    # --- FLAT CONFIG GRID (3 columns, always visible) ---

    col_detect, col_cluster, col_select = st.columns(3)

    # Column 1: Detection & Quality
    with col_detect:
        st.markdown("**Detection & Quality**")
        detection_confidence = st.slider(
            "Detection Confidence", 0.05, 0.5,
            value=float(saved_detect.get("confidence_threshold", 0.25)),
            step=0.05, key="config_det_conf",
        )
        min_face_size = st.slider(
            "Min Face Size (px)", 20, 100,
            value=int(saved_insightface.get("min_face_size", 50)),
            step=10, key="config_min_face_size",
        )
        min_iqa = st.slider(
            "Min IQA Score", 0.0, 1.0,
            value=float(saved_filter.get("min_iqa_score", 0.2)),
            step=0.05, key="config_min_iqa",
        )
        min_sharpness = st.slider(
            "Min Sharpness", 0.0, 1.0,
            value=float(saved_filter.get("min_sharpness", 0.1)),
            step=0.05, key="config_min_sharpness",
        )
        embedding_backend = st.selectbox(
            "Embedding Model",
            options=["insightface", "custom"],
            index=0 if saved_embedding.get("backend", "insightface") == "insightface" else 1,
            key="config_embedding_backend",
        )

    # Column 2: Face Clustering
    with col_cluster:
        st.markdown("**Face Clustering**")
        clustering_methods = ["face_cluster_knn", "hdbscan", "hdbscan_pca", "mutual_knn", "agglomerative"]
        saved_method = saved_cluster_people.get("method", "face_cluster_knn")
        method_index = clustering_methods.index(saved_method) if saved_method in clustering_methods else 0

        people_method = st.selectbox(
            "Method", options=clustering_methods, index=method_index,
            key="config_people_method",
        )

        # Defaults (overridden by method-specific widgets)
        people_min_cluster_size = 2
        people_distance_threshold = 0.5
        cluster_merge_epsilon = 0.3
        pca_components = 128
        knn_k = 10
        knn_similarity_threshold = 0.70
        fc_K = 5
        fc_dist_threshold = 0.35
        fc_min_cluster = 2
        fc_merge_enabled = False
        fc_attach_enabled = False
        fc_export = True
        fc_merge_params = {}

        if people_method == "face_cluster_knn":
            fc_K = st.slider("K (neighbors)", 1, 100, value=int(saved_cluster_people.get("K", 5)), key="rc_K")
            fc_dist_threshold = st.slider("Distance Threshold", 0.01, 1.0, value=float(saved_cluster_people.get("distance_threshold", 0.35)), step=0.01, key="rc_dist")
            fc_min_cluster = st.slider("Min Cluster Size", 1, 20, value=int(saved_cluster_people.get("min_cluster_size", 2)), key="rc_min_cluster")
            fc_merge_enabled = st.checkbox("Merge", value=bool(saved_cluster_people.get("merge_enabled", False)), key="rc_merge")
            fc_attach_enabled = st.checkbox("Attach holdouts", value=bool(saved_cluster_people.get("attach_enabled", False)), key="rc_attach")
            fc_export = st.checkbox("Export for analysis", value=bool(saved_cluster_people.get("export_for_analysis", True)), key="config_fc_export")
        elif people_method == "hdbscan":
            people_min_cluster_size = st.slider("Min Faces/Person", 1, 5, value=int(saved_cluster_people.get("min_cluster_size", 2)), key="config_people_min_cluster")
            cluster_merge_epsilon = st.slider("Merge Distance", 0.0, 0.8, value=float(saved_cluster_people.get("cluster_selection_epsilon", 0.3)), step=0.05, key="config_cluster_epsilon")
        elif people_method == "hdbscan_pca":
            people_min_cluster_size = st.slider("Min Faces/Person", 1, 5, value=int(saved_cluster_people.get("min_cluster_size", 2)), key="config_people_min_cluster_pca")
            pca_components = st.selectbox("PCA Dims", options=[64, 128, 256], index=[64, 128, 256].index(saved_cluster_people.get("pca_components", 128)) if saved_cluster_people.get("pca_components", 128) in [64, 128, 256] else 1, key="config_pca_components")
            cluster_merge_epsilon = st.slider("Merge Distance", 0.0, 0.8, value=float(saved_cluster_people.get("cluster_selection_epsilon", 0.3)), step=0.05, key="config_cluster_epsilon_pca")
        elif people_method == "mutual_knn":
            knn_k = st.slider("KNN Neighbors", 3, 20, value=int(saved_cluster_people.get("k", 10)), key="config_knn_k")
            knn_similarity_threshold = st.slider("Similarity Thresh", 0.50, 0.90, value=float(saved_cluster_people.get("similarity_threshold", 0.70)), step=0.05, key="config_knn_sim_threshold")
        elif people_method == "agglomerative":
            people_distance_threshold = st.slider("Distance Threshold", 0.3, 0.9, value=float(saved_cluster_people.get("distance_threshold", 0.5)), step=0.05, key="config_people_dist")

    # Column 3: Selection
    with col_select:
        st.markdown("**Selection**")
        max_per_cluster = st.number_input(
            "Max per Cluster", 1, 10,
            value=int(saved_select.get("max_images_per_cluster", 2)),
            key="config_max_per_cluster",
        )
        min_score_threshold = st.slider(
            "Min Score", 0.0, 1.0,
            value=float(saved_select.get("min_score_threshold", 0.4)),
            step=0.05, key="config_min_score",
        )
        duplicate_threshold = st.slider(
            "Dissimilarity Thresh", 0.80, 0.95,
            value=float(saved_select.get("dissimilarity_threshold", 0.85)),
            step=0.01, key="config_dup_thresh",
        )
        siamese_config = saved_select.get("siamese", {})
        siamese_enabled = st.checkbox(
            "Siamese Refinement",
            value=bool(siamese_config.get("enabled", True)),
            key="config_siamese",
        )

    # --- Merge Parameters (full-width, visible when merge enabled, NOT an expander) ---
    if people_method == "face_cluster_knn" and fc_merge_enabled:
        st.divider()
        st.markdown("**Merge Parameters**")
        fc_merge_params = _render_merge_params(key_prefix="rc_")

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
        # People clustering config — only include params relevant to selected method
        "cluster_people": {
            "method": people_method,
            **(
                # face_cluster_knn params
                {
                    "K": fc_K,
                    "distance_threshold": fc_dist_threshold,
                    "min_cluster_size": fc_min_cluster,
                    "merge_enabled": fc_merge_enabled,
                    "attach_enabled": fc_attach_enabled,
                    "export_for_analysis": fc_export,
                    **fc_merge_params,
                } if people_method == "face_cluster_knn"
                else {
                    # Legacy method params
                    "min_cluster_size": people_min_cluster_size,
                    "min_samples": people_min_cluster_size,
                    "distance_threshold": people_distance_threshold,
                    **({"cluster_selection_epsilon": cluster_merge_epsilon} if people_method in ("hdbscan", "hdbscan_pca") else {}),
                    **({"pca_components": pca_components} if people_method == "hdbscan_pca" else {}),
                    **({"k": knn_k, "similarity_threshold": knn_similarity_threshold} if people_method == "mutual_knn" else {}),
                }
            ),
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

    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button(
            "Run Pipeline" if not is_running else "Running...",
            type="primary",
            disabled=is_running,
            use_container_width=True,
            key="run_pipeline_btn",
        ):
            job_id = _start_pipeline(album.album_id, selected_pipeline, steps, config)
            if job_id:
                return job_id

    with col2:
        if st.button(
            "Save Settings",
            disabled=is_running,
            use_container_width=True,
            key="save_settings_btn",
        ):
            _save_user_settings(selected_pipeline, config)

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
