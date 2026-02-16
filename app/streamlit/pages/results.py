"""Results page - Pipeline execution and results viewing."""

import time
import io
import base64
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps

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


def _load_thumbnail(image_path: str, size: int = 100) -> str:
    """Load image and return base64 thumbnail (square, fixed size)."""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            # Crop to center square
            w, h = img.size
            sq = min(w, h)
            left, top = (w - sq) // 2, (h - sq) // 2
            img = img.crop((left, top, left + sq, top + sq))
            # Force exact size
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Run Pipeline", "View Results", "Metrics Table", "Comparisons", "Sub-Clusters", "Export"])

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
        _render_comparisons_tab(album)

    with tab5:
        _render_subclusters_tab(album)

    with tab6:
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
        view_mode = st.radio("View Mode", ["Selected Only", "All Processed", "By Cluster"], horizontal=True, key="results_view_mode")

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

    elif view_mode == "All Processed":
        images = client.get_images(job_id)
        st.write(f"**{len(images)}** images processed by pipeline")
        render_image_gallery(images, show_scores=show_scores, columns=columns)

    else:
        clusters = client.get_clusters(job_id)
        st.write(f"**{len(clusters)}** clusters")
        render_cluster_gallery(clusters, show_all_images=True)


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


def _render_comparisons_tab(album: Album) -> None:
    """Render Siamese/duplicate comparison log for debugging."""
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No pipeline results yet. Run the pipeline first.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))

    comparisons = client.get_comparisons(job_id)

    if not comparisons:
        st.info("No comparison data available. Run pipeline with Siamese enabled.")
        return

    st.write(f"**{len(comparisons)}** comparisons performed")

    # Separate by type
    tiebreakers = [c for c in comparisons if c.get('type') == 'tiebreaker']
    duplicates = [c for c in comparisons if c.get('type') == 'duplicate_check']

    if tiebreakers:
        st.subheader(f"Tiebreaker Comparisons ({len(tiebreakers)})")
        st.caption("When top images have similar scores, Siamese decides the winner")
        for c in tiebreakers:
            col1, col2, col3 = st.columns([1, 1, 1])
            winner = c.get('winner', '?')
            with col1:
                thumb1 = _load_thumbnail(c.get('img1_path', ''))
                if thumb1:
                    st.image(thumb1, width=100)
                score1 = c.get('score1', '?')
                is_winner1 = winner == c['img1']
                label1 = f"**{c['img1']}**" + (" ✓ WINNER" if is_winner1 else "")
                st.markdown(label1)
                st.caption(f"Score: {score1}")
            with col2:
                conf = c.get('confidence', 0)
                st.markdown("**VS**")
                st.metric("Confidence", f"{conf:.2f}" if isinstance(conf, float) else conf)
            with col3:
                thumb2 = _load_thumbnail(c.get('img2_path', ''))
                if thumb2:
                    st.image(thumb2, width=100)
                score2 = c.get('score2', '?')
                is_winner2 = winner == c['img2']
                label2 = f"**{c['img2']}**" + (" ✓ WINNER" if is_winner2 else "")
                st.markdown(label2)
                st.caption(f"Score: {score2}")
            st.divider()

    if duplicates:
        st.subheader(f"Duplicate Checks ({len(duplicates)})")
        st.caption("Checking if #2 image is too similar to #1 (near-duplicate)")
        for c in duplicates:
            is_dup = c.get('is_duplicate', False)
            method = c.get('method', 'unknown')
            conf = c.get('confidence', c.get('threshold', '?'))

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                thumb1 = _load_thumbnail(c.get('img1_path', ''))
                if thumb1:
                    st.image(thumb1, width=100)
                st.markdown(f"**{c['img1']}**")
            with col2:
                if is_dup:
                    st.error(f"DUPLICATE")
                    st.caption(f"{method}, conf: {conf}")
                else:
                    st.success(f"Different")
                    st.caption(f"{method}, conf: {conf}")
            with col3:
                thumb2 = _load_thumbnail(c.get('img2_path', ''))
                if thumb2:
                    st.image(thumb2, width=100)
                st.markdown(f"**{c['img2']}**")
            st.divider()


def _render_subclusters_tab(album: Album) -> None:
    """Render face sub-clusters within scene clusters."""
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No pipeline results yet. Run the pipeline first.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))

    subclusters = client.get_subclusters(job_id)

    if not subclusters:
        st.info("No sub-cluster data available. Ensure `cluster_by_identity` step ran.")
        return

    st.write(f"**{len(subclusters)}** scene clusters with face-based sub-clusters")
    st.caption("Each scene cluster is split by unique face combinations (e.g., A+B, A-only, B-only, no faces)")

    # Sort scene clusters by ID
    sorted_scenes = sorted(subclusters.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)

    for scene_id, sub_dict in sorted_scenes:
        if not sub_dict:
            continue

        with st.expander(f"Scene Cluster {scene_id} ({len(sub_dict)} sub-clusters)", expanded=False):
            # Sort sub-clusters by face count descending
            def parse_face_count(val):
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    # Handle strings like "3+" by stripping non-numeric chars
                    cleaned = ''.join(c for c in val if c.isdigit())
                    return int(cleaned) if cleaned else 0
                return 0

            sorted_subs = sorted(
                sub_dict.items(),
                key=lambda x: parse_face_count(x[1].get("face_count", 0)),
                reverse=True
            )

            for sub_id, sub_info in sorted_subs:
                face_count = sub_info.get("face_count", 0)
                identity = sub_info.get("identity", "")
                images = sub_info.get("images", [])
                has_faces = sub_info.get("has_faces", False)

                # Build sub-cluster header
                if has_faces:
                    header = f"👥 Sub-cluster {sub_id}: {face_count} face(s)"
                    if identity:
                        header += f" - Identity: {identity}"
                else:
                    header = f"📷 Sub-cluster {sub_id}: No faces"

                st.markdown(f"**{header}** ({len(images)} images)")

                if images:
                    # Display thumbnails in columns
                    cols = st.columns(min(len(images), 6))
                    for i, img_path in enumerate(images[:6]):
                        with cols[i % 6]:
                            thumb = _load_thumbnail(img_path, size=120)
                            if thumb:
                                st.image(thumb, caption=Path(img_path).name, use_container_width=True)
                            else:
                                st.caption(Path(img_path).name)

                    if len(images) > 6:
                        st.caption(f"... and {len(images) - 6} more images")

                st.divider()


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
        st.metric("All Processed", total_filtered)

    st.divider()

    render_export_panel(job_id, num_selected, on_export_complete=lambda p: add_notification(f"Exported to {p}", "success"))
