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


@st.cache_data(show_spinner=False)
def _image_to_base64_thumbnail(image_path: Path, size: int = 60) -> Optional[str]:
    """Load an image, resize to square thumbnail, and return a base64 data URI.

    Cached: Streamlit reruns the whole script on every interaction, so without
    this the table re-encodes every thumbnail from disk on each rerun (SIGHTING-103).
    """
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


# spec-084: single source of truth for column tooltips (meaning; range). Every
# column rendered in the metrics table MUST have an entry here -- the column
# config is generated from this dict so no column ships without a tooltip.
METRIC_HELP = {
    "Thumbnail": "Center-cropped preview of the image.",
    "Image": "Source image filename.",
    "Status": "Selected = kept for the album; Filtered = not selected.",
    "Reason": "Why the image was selected or filtered (from the select_best step).",
    "Final": "Composite score = Quality + Penalty; higher is better (typically 0-1).",
    "Quality": "Quality half of the composite (IQA/AVA blend); 0-1, higher is better.",
    "Penalty": "Person penalty subtracted from Quality (occlusion, eyes closed, ...); <= 0.",
    "AVA": "Aesthetic score (AVA model); 0-1, higher = more aesthetically pleasing.",
    "IQA": "Technical image-quality score; 0-1, higher = sharper/cleaner.",
    "Sharp": "Sharpness; 0-1, higher = less blur.",
    "Body": "Y if a person body was detected (YOLO pose).",
    "Faces": "Faces passed / total after quality filtering.",
    "Frontal": "Best face frontal score; 0-1, 1 = facing camera, low = profile.",
    "Central": "Best face centrality; 0-1, 1 = centered in frame.",
    "Roll": "Head roll (tilt) angle of the best face, in degrees; 0 = level.",
    "Cluster": "Number of faces eligible for identity clustering.",
    "BodyPose": "Body-facing-camera score; 0-1, higher = facing camera.",
    "FacePose": "Face pose frontal score; 0-1, higher = more frontal.",
    "Eyes": "Eyes-open score; 0-1, 1 = wide open, low = closed/blinking.",
    "Smile": "Smile score; 0-1, higher = bigger smile.",
    "SceneCluster": "Scene cluster ID this image was grouped into.",
}


def _build_metric_row(img: "ImageInfo", is_sel: bool) -> dict:
    """Build one metrics-table row dict for an image. spec-084.

    Every key here MUST have a matching entry in ``METRIC_HELP`` (enforced by
    ``tests/api/test_results_metrics.py``) so no column ships without a tooltip.
    """
    status = "Selected" if is_sel else "Filtered"

    best_pose = img.face_pose_scores[0] if img.face_pose_scores else None
    best_eyes = img.face_eyes_scores[0] if img.face_eyes_scores else None
    best_smile = img.face_smile_scores[0] if img.face_smile_scores else None

    thumb = _image_to_base64_thumbnail(Path(img.path))

    has_body = img.person_detected if img.person_detected is not None else False
    has_face = (img.face_count or 0) > 0

    filter_stats = getattr(img, 'filter_stats', None) or {}
    faces_passed = filter_stats.get('passed', img.face_count or 0)
    faces_filtered = filter_stats.get('filtered', 0)

    best_frontal = getattr(img, 'best_frontal_score', None)
    best_centrality = getattr(img, 'best_centrality', None)
    roll_angles = getattr(img, 'roll_angles', None) or []
    best_roll = roll_angles[0] if roll_angles else None

    frontal_stats = getattr(img, 'frontal_stats', None) or {}
    clusterable_count = frontal_stats.get('clusterable', faces_passed)

    # spec-084: composite breakdown + decision reason.
    quality = getattr(img, "quality_score", None)
    penalty = getattr(img, "person_penalty", None)
    reason = getattr(img, "filter_reason", None) or ""

    return {
        "Thumbnail": thumb,
        "Image": Path(img.path).name,
        "Status": status,
        "Reason": reason,
        "Final": f"{img.composite_score:.2f}" if img.composite_score is not None else "N/A",
        "Quality": f"{quality:.2f}" if quality is not None else "N/A",
        "Penalty": f"{penalty:+.2f}" if penalty is not None else "N/A",
        "AVA": f"{img.ava_score:.2f}" if img.ava_score is not None else "N/A",
        "IQA": f"{img.iqa_score:.2f}" if img.iqa_score is not None else "N/A",
        "Sharp": f"{img.sharpness:.2f}" if img.sharpness is not None else "N/A",
        # Body/Face detection columns
        "Body": "Y" if has_body else "",
        "Faces": f"{faces_passed}/{img.face_count}" if faces_filtered > 0 else (f"{img.face_count}" if has_face else ""),
        "Frontal": f"{best_frontal:.2f}" if best_frontal is not None else "",
        "Central": f"{best_centrality:.2f}" if best_centrality is not None else "",
        "Roll": f"{best_roll:.1f}" if best_roll is not None else "",
        "Cluster": f"{clusterable_count}" if clusterable_count else "",
        "BodyPose": f"{img.body_facing_score:.2f}" if img.body_facing_score is not None else "",
        "FacePose": f"{best_pose:.2f}" if best_pose is not None else "",
        "Eyes": f"{best_eyes:.2f}" if best_eyes is not None else "",
        "Smile": f"{best_smile:.2f}" if best_smile is not None else "",
        "SceneCluster": str(img.cluster_id) if img.cluster_id is not None else "-",
    }


def render_image_metrics_table(images: List["ImageInfo"], selected_paths: set = None) -> None:
    """Render a detailed per-image metrics table with thumbnails and CSV download."""
    import pandas as pd

    if not images:
        st.info("No image metrics available")
        return

    st.subheader("Per-Image Metrics")

    if selected_paths is None:
        selected_paths = set()

    rows = [
        _build_metric_row(img, img.is_selected or img.path in selected_paths)
        for img in images
    ]

    df = pd.DataFrame(rows)

    # spec-084: generate column config from METRIC_HELP so EVERY column carries a
    # tooltip (meaning; range). Wide columns for free text, small for scores.
    _wide = {"Image", "Reason"}
    column_config = {}
    for col in df.columns:
        help_text = METRIC_HELP.get(col)
        if col == "Thumbnail":
            column_config[col] = st.column_config.ImageColumn("Thumb", width="small", help=help_text)
        else:
            width = "large" if col == "Reason" else ("medium" if col in _wide else "small")
            label = "Scene" if col == "SceneCluster" else col
            column_config[col] = st.column_config.TextColumn(label, width=width, help=help_text)

    st.dataframe(
        df,
        column_config=column_config,
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    # CSV export without thumbnails
    csv_df = df.drop(columns=["Thumbnail"])
    csv = csv_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "image_metrics.csv", "text/csv")

    # spec-084: the dataframe thumbnail isn't openable, so offer a full-image
    # viewer. st.image renders a built-in fullscreen-expand button on hover.
    # Index-prefixed labels keep duplicate filenames distinct.
    options = {f"{i+1}. {Path(img.path).name}": img.path for i, img in enumerate(images)}
    choice = st.selectbox("View full image", ["(none)"] + list(options.keys()),
                          key="metrics_full_image")
    if choice and choice != "(none)":
        full_path = options[choice]
        if Path(full_path).exists():
            st.image(full_path, caption=Path(full_path).name, use_container_width=True)
        else:
            st.warning(f"Image not found on disk: {full_path}")


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
