"""Image gallery component for displaying photos."""

import base64
import io
from collections import defaultdict

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Callable
from PIL import Image, ImageOps

from app.streamlit.models import ImageInfo, ClusterInfo
from app.streamlit.config import get_config


THUMBNAIL_WIDTH = 200  # Fixed width for gallery thumbnails (fits 4 per row)


@st.cache_data(show_spinner=False)
def _load_thumbnail_cached(image_path: str) -> Optional[bytes]:
    """Load and cache thumbnail as JPEG bytes (preserves aspect ratio)."""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            # Resize to fixed width, maintain aspect ratio
            w, h = img.size
            new_width = THUMBNAIL_WIDTH
            new_height = int(h * (new_width / w))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception:
        return None


def _load_thumbnail(image_path: Path) -> Optional[Image.Image]:
    """Load image thumbnail (fixed width, preserves aspect ratio)."""
    try:
        if not image_path or not image_path.exists():
            return None
        img_bytes = _load_thumbnail_cached(str(image_path))
        if img_bytes is None:
            return None
        return Image.open(io.BytesIO(img_bytes))
    except Exception:
        return None


def render_image_gallery(
    images: List[ImageInfo],
    title: Optional[str] = None,
    show_scores: bool = True,
    show_selection: bool = True,
    columns: Optional[int] = None,
    on_image_click: Optional[Callable[[ImageInfo], None]] = None,
) -> None:
    """Render a grid gallery of images."""
    if not images:
        st.info("No images to display")
        return

    config = get_config()
    num_cols = columns or config.images_per_row

    if title:
        st.subheader(title)

    cols = st.columns(num_cols)

    for i, image in enumerate(images):
        with cols[i % num_cols]:
            render_image_card(image, show_scores, show_selection, on_image_click)


def render_image_card(
    image: ImageInfo,
    show_scores: bool = True,
    show_selection: bool = True,
    on_click: Optional[Callable[[ImageInfo], None]] = None,
) -> None:
    """Render a single image card with fixed-size thumbnail."""
    if not image.path:
        st.warning(f"No image path (filename: {image.filename or 'unknown'})")
        return

    image_path = Path(image.path)

    # Display thumbnail at fixed width
    img = _load_thumbnail(image_path)
    if img:
        st.image(img, width=THUMBNAIL_WIDTH)
    else:
        st.warning(f"Image not found: {image_path.name}")

    st.caption(image.filename[:20] + "..." if len(image.filename) > 20 else image.filename)

    if show_scores:
        _render_scores(image)

    if image.face_count and image.face_count > 0:
        _render_face_info(image)

    if show_selection and image.is_selected:
        st.success("Selected")

    if on_click:
        if st.button("View", key=f"view_{image.path}"):
            on_click(image)


def _render_scores(image: ImageInfo) -> None:
    """Render score information."""
    parts = []
    if image.iqa_score is not None:
        parts.append(f"IQA: {image.iqa_score:.2f}")
    if image.ava_score is not None:
        parts.append(f"AVA: {image.ava_score:.1f}")
    if image.composite_score is not None:
        parts.append(f"Score: {image.composite_score:.2f}")
    if parts:
        st.caption(" | ".join(parts))


def _render_face_info(image: ImageInfo) -> None:
    """Render face and person information."""
    parts = []

    # Person detection (InsightFace)
    if image.person_detected is not None:
        if image.person_detected:
            body_score = image.body_facing_score or 0
            parts.append(f"body: {body_score:.0%}")
        else:
            parts.append("no person")

    # Face info
    if image.face_count:
        faces_label = "face" if image.face_count == 1 else "faces"
        parts.append(f"{image.face_count} {faces_label}")

    if image.face_eyes_scores and image.face_eyes_scores[0] is not None:
        parts.append("eyes open" if image.face_eyes_scores[0] > 0.5 else "eyes closed")
    if image.face_smile_scores and image.face_smile_scores[0] is not None:
        parts.append("smiling" if image.face_smile_scores[0] > 0.5 else "neutral")

    if parts:
        st.caption(" | ".join(parts))


def render_cluster_gallery(
    clusters: List[ClusterInfo],
    show_all_images: bool = False,
    max_preview: int = 4,
    on_cluster_click: Optional[Callable[[ClusterInfo], None]] = None,
) -> None:
    """Render clusters as expandable galleries."""
    if not clusters:
        st.info("No clusters to display")
        return

    for cluster in clusters:
        _render_cluster_section(cluster, show_all_images, max_preview, on_cluster_click)


def _group_images_by_people(
    images: List[ImageInfo],
    person_labels: Dict[str, List[str]],
) -> Dict[str, List[ImageInfo]]:
    """Group images by their person combination."""
    groups: Dict[str, List[ImageInfo]] = defaultdict(list)
    for img in images:
        people = person_labels.get(img.path, [])
        key = ", ".join(sorted(people)) if people else "No identified people"
        groups[key].append(img)
    return dict(groups)


def _image_to_base64_thumbnail(image_path: Path, size: int = 80) -> Optional[str]:
    """Load an image, resize to square thumbnail, and return a base64 data URI."""
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


def _render_cluster_score_table(cluster: ClusterInfo) -> None:
    """Render a per-image debug spreadsheet with thumbnails for a cluster."""
    rows = []
    for img in cluster.images:
        thumb = _image_to_base64_thumbnail(Path(img.path))
        people = cluster.person_labels.get(img.path, [])

        pose = img.face_pose_scores[0] if img.face_pose_scores else None
        eyes = img.face_eyes_scores[0] if img.face_eyes_scores else None
        smile = img.face_smile_scores[0] if img.face_smile_scores else None

        row = {
            "Thumbnail": thumb,
            "Image": img.filename,
            "Selected": img.is_selected,
            "Final Score": round(img.composite_score, 3) if img.composite_score else None,
            "IQA": round(img.iqa_score, 3) if img.iqa_score else None,
            "AVA": round(img.ava_score, 2) if img.ava_score else None,
            "Faces": img.face_count or 0,
            "Pose": round(pose, 2) if pose else None,
            "Eyes": round(eyes, 2) if eyes else None,
            "Smile": round(smile, 2) if smile else None,
        }

        # Add InsightFace metrics if available
        if img.person_detected is not None:
            row["Person"] = "Yes" if img.person_detected else "No"
        if img.body_facing_score is not None:
            row["Body"] = round(img.body_facing_score, 2)

        row["People"] = ", ".join(people) if people else ""
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        column_config={
            "Thumbnail": st.column_config.ImageColumn("Image", width="small"),
            "Selected": st.column_config.CheckboxColumn("Selected", disabled=True),
        },
        hide_index=True,
    )


def _render_cluster_section(
    cluster: ClusterInfo,
    show_all: bool = False,
    max_preview: int = 4,
    on_click: Optional[Callable[[ClusterInfo], None]] = None,
) -> None:
    """Render a single cluster section."""
    face_info = ""
    if cluster.has_faces and cluster.face_count:
        faces_label = "face" if cluster.face_count == 1 else "faces"
        face_info = f" | {cluster.face_count} {faces_label}"

    header = f"Cluster {cluster.cluster_id} ({cluster.image_count} images, {cluster.selected_count} selected{face_info})"

    with st.expander(header, expanded=False):
        if on_click:
            if st.button("View Full Cluster", key=f"cluster_{cluster.cluster_id}"):
                on_click(cluster)

        groups = _group_images_by_people(cluster.images, cluster.person_labels)
        multiple_groups = len(groups) > 1

        for group_label, group_images in groups.items():
            if multiple_groups:
                st.markdown(f"**{group_label}** ({len(group_images)} images)")

            images_to_show = group_images if show_all else group_images[:max_preview]

            if images_to_show:
                num_cols = min(6 if show_all else 4, len(images_to_show))
                render_image_gallery(images_to_show, show_scores=True, show_selection=True, columns=num_cols)
                remaining = len(group_images) - len(images_to_show)
                if remaining > 0:
                    st.caption(f"... and {remaining} more images")

        st.markdown("---")
        st.caption("Score Details")
        _render_cluster_score_table(cluster)


def render_image_comparison(images: List[ImageInfo], title: str = "Compare Images") -> None:
    """Render side-by-side image comparison."""
    if not images:
        return

    st.subheader(title)
    num_images = min(len(images), 4)
    cols = st.columns(num_images)

    for i, image in enumerate(images[:num_images]):
        with cols[i]:
            image_path = Path(image.path)
            if image_path.exists():
                img = _load_thumbnail(image_path)
                st.image(img, width=THUMBNAIL_WIDTH)
            st.caption(image.filename[:20] + "..." if len(image.filename) > 20 else image.filename)
            if image.iqa_score is not None:
                st.metric("IQA", f"{image.iqa_score:.2f}")
            if image.ava_score is not None:
                st.metric("AVA", f"{image.ava_score:.1f}")


def render_selected_summary(images: List[ImageInfo], total_images: int) -> None:
    """Render a summary of selected images."""
    selected_count = len(images)
    reduction = ((total_images - selected_count) / total_images * 100) if total_images > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected", selected_count)
    with col2:
        st.metric("Total", total_images)
    with col3:
        st.metric("Reduction", f"{reduction:.1f}%")
