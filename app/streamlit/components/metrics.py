"""Metrics display components."""

import io
import base64
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import statistics
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from app.streamlit.models import ImageInfo


def _image_to_base64_thumbnail(image_path: Path, size: int = 60) -> Optional[str]:
    """Load an image, resize to thumbnail, and return a base64 data URI."""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((size, size))
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def render_pipeline_metrics(result: Dict[str, Any], title: str = "Pipeline Results") -> None:
    """Render pipeline result metrics."""
    st.subheader(title)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", result.get("total_images", 0))
    with col2:
        st.metric("After Filtering", result.get("filtered_images", 0))
    with col3:
        st.metric("Clusters", result.get("num_clusters", 0))
    with col4:
        st.metric("Selected", result.get("num_selected", 0))

    total = result.get("total_images", 0)
    selected = result.get("num_selected", 0)
    if total > 0:
        reduction = (total - selected) / total * 100
        st.progress(1 - (selected / total), text=f"Reduced by {reduction:.1f}%")

    duration_ms = result.get("total_duration_ms", 0)
    if duration_ms > 0:
        st.caption(f"Total duration: {duration_ms / 1000:.1f}s")


def render_step_timings(step_timings: Dict[str, float], title: str = "Step Timings") -> None:
    """Render step timing breakdown."""
    if not step_timings:
        return

    with st.expander(title, expanded=False):
        sorted_steps = sorted(step_timings.items(), key=lambda x: x[1], reverse=True)
        total_ms = sum(step_timings.values())

        for step, duration_ms in sorted_steps:
            pct = (duration_ms / total_ms * 100) if total_ms > 0 else 0
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(step)
            with col2:
                st.write(f"{duration_ms / 1000:.2f}s")
            with col3:
                st.write(f"{pct:.1f}%")


def render_quality_distribution(scores: List[float], title: str = "Quality Distribution", score_type: str = "IQA") -> None:
    """Render a histogram of quality scores."""
    if not scores:
        return

    st.subheader(title)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"Min {score_type}", f"{min(scores):.2f}")
    with col2:
        st.metric(f"Max {score_type}", f"{max(scores):.2f}")
    with col3:
        st.metric(f"Mean {score_type}", f"{statistics.mean(scores):.2f}")
    with col4:
        std = statistics.stdev(scores) if len(scores) > 1 else 0
        st.metric("Std Dev", f"{std:.2f}")

    buckets = [0] * 10
    for score in scores:
        bucket_idx = min(int(score * 10), 9)
        buckets[bucket_idx] += 1

    chart_data = {
        "Range": [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)],
        "Count": buckets,
    }
    st.bar_chart(chart_data, x="Range", y="Count")


def render_cluster_summary(clusters: List[Dict[str, Any]], title: str = "Cluster Summary") -> None:
    """Render cluster statistics."""
    if not clusters:
        return

    st.subheader(title)

    total_clusters = len(clusters)
    face_clusters = sum(1 for c in clusters if c.get("has_faces", False))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clusters", total_clusters)
    with col2:
        st.metric("With Faces", face_clusters)
    with col3:
        st.metric("Without Faces", total_clusters - face_clusters)

    sizes = [c.get("image_count", 0) for c in clusters]
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        st.caption(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={avg_size:.1f}")


def render_people_summary(people: List[Dict[str, Any]], title: str = "People Summary") -> None:
    """Render people detection statistics."""
    if not people:
        st.info("No people detected")
        return

    st.subheader(title)

    total_people = len(people)
    total_faces = sum(p.get("face_count", 0) for p in people)
    named_people = sum(1 for p in people if p.get("name"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("People", total_people)
    with col2:
        st.metric("Total Faces", total_faces)
    with col3:
        st.metric("Named", named_people)


def render_metric_card(label: str, value: Any, delta: Optional[float] = None, delta_color: str = "normal") -> None:
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_image_metrics_table(images: List["ImageInfo"], selected_paths: set = None) -> None:
    """Render a detailed per-image metrics table with thumbnails and CSV download."""
    import pandas as pd
    from app.streamlit.models import ImageInfo

    if not images:
        st.info("No image metrics available")
        return

    st.subheader("Per-Image Metrics")

    if selected_paths is None:
        selected_paths = set()

    rows = []
    for img in images:
        is_sel = img.is_selected or img.path in selected_paths
        status = "Selected" if is_sel else "Filtered"

        best_pose = img.face_pose_scores[0] if img.face_pose_scores else None
        best_eyes = img.face_eyes_scores[0] if img.face_eyes_scores else None
        best_smile = img.face_smile_scores[0] if img.face_smile_scores else None

        # Generate thumbnail
        thumb = _image_to_base64_thumbnail(Path(img.path))

        rows.append({
            "Thumbnail": thumb,
            "Image": Path(img.path).name,
            "Status": status,
            "Final": f"{img.composite_score:.2f}" if img.composite_score is not None else "N/A",
            "AVA": f"{img.ava_score:.1f}" if img.ava_score is not None else "N/A",
            "IQA": f"{img.iqa_score:.2f}" if img.iqa_score is not None else "N/A",
            "Sharpness": f"{img.sharpness:.2f}" if img.sharpness is not None else "N/A",
            "Faces": img.face_count or 0,
            "Pose": f"{best_pose:.2f}" if best_pose is not None else "",
            "Eyes": f"{best_eyes:.2f}" if best_eyes is not None else "",
            "Smile": f"{best_smile:.2f}" if best_smile is not None else "",
            "Cluster": str(img.cluster_id) if img.cluster_id is not None else "-",
        })

    df = pd.DataFrame(rows)

    # Configure column with image display
    st.dataframe(
        df,
        column_config={
            "Thumbnail": st.column_config.ImageColumn("Image", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    # CSV export without thumbnails
    csv_df = df.drop(columns=["Thumbnail"])
    csv = csv_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "image_metrics.csv", "text/csv")


def render_results_table(results: List[Dict[str, Any]], title: str = "Pipeline Runs") -> None:
    """Render a table of pipeline run results."""
    if not results:
        st.info("No pipeline results yet")
        return

    st.subheader(title)

    table_data = []
    for r in results:
        table_data.append({
            "Album": r.get("album_name", r.get("album_id", "")[:8]),
            "Status": r.get("status", "unknown"),
            "Images": r.get("total_images", 0),
            "Selected": r.get("num_selected", 0),
            "Duration": f"{r.get('total_duration_ms', 0) / 1000:.1f}s",
            "Date": r.get("created_at", "")[:10] if r.get("created_at") else "",
        })

    st.dataframe(table_data, use_container_width=True)
