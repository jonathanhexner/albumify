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


def _load_image_for_display(image_path: Path) -> Image.Image:
    """Load image with EXIF orientation correction."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return img.copy()


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
            render_image_card(
                image,
                show_scores=show_scores,
                show_selection=show_selection,
                on_click=on_image_click,
            )


def render_image_card(
    image: ImageInfo,
    show_scores: bool = True,
    show_selection: bool = True,
    on_click: Optional[Callable[[ImageInfo], None]] = None,
) -> None:
    """Render a single image card."""
    with st.container():
        image_path = Path(image.path)

        if image_path.exists():
            img = _load_image_for_display(image_path)
            st.image(img, use_container_width=True)
        else:
            st.warning("Image not found")

        st.caption(image.filename)

        # Scores
        if show_scores:
            score_parts = []
            if image.iqa_score is not None:
                score_parts.append(f"IQA: {image.iqa_score:.2f}")
            if image.ava_score is not None:
                score_parts.append(f"AVA: {image.ava_score:.1f}")
            if image.sharpness is not None:
                score_parts.append(f"Sharp: {image.sharpness:.2f}")
            if image.composite_score is not None:
                score_parts.append(f"Score: {image.composite_score:.2f}")
            if score_parts:
                st.caption(" | ".join(score_parts))

        if image.face_count and image.face_count > 0:
            faces_label = "face" if image.face_count == 1 else "faces"
            parts = [f"{image.face_count} {faces_label}"]
            if image.face_eyes_scores:
                eyes_open = image.face_eyes_scores[0] > 0.5
                parts.append("eyes open" if eyes_open else "eyes closed")
            if image.face_smile_scores:
                smiling = image.face_smile_scores[0] > 0.5
                parts.append("smiling" if smiling else "neutral")
            st.caption(" | ".join(parts))

        if show_selection and image.is_selected:
            st.success("Selected")

        if on_click and st.button("View", key=f"view_{image.path}", use_container_width=True):
            on_click(image)


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
    """Group images by their person combination.

    Returns a dict mapping a display label (sorted person names joined)
    to the list of images sharing that combination.  Images with no
    person labels are grouped under "No identified people".
    """
    groups: Dict[str, List[ImageInfo]] = defaultdict(list)
    for img in images:
        people = person_labels.get(img.path, [])
        if people:
            key = ", ".join(sorted(people))
        else:
            key = "No identified people"
        groups[key].append(img)
    return dict(groups)


def _image_to_base64_thumbnail(image_path: Path, size: int = 80) -> Optional[str]:
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


def _render_cluster_score_table(cluster: ClusterInfo) -> None:
    """Render a per-image debug spreadsheet with thumbnails for a cluster."""
    rows = []
    for img in cluster.images:
        thumb = _image_to_base64_thumbnail(Path(img.path))
        people = cluster.person_labels.get(img.path, [])

        # Summarise face sub-scores as single representative values
        pose = None
        if img.face_pose_scores:
            pose = round(img.face_pose_scores[0], 2)
        eyes = None
        if img.face_eyes_scores:
            eyes = round(img.face_eyes_scores[0], 2)
        smile = None
        if img.face_smile_scores:
            smile = round(img.face_smile_scores[0], 2)

        rows.append({
            "Thumbnail": thumb,
            "Image": img.filename,
            "Selected": img.is_selected,
            "Final Score": round(img.composite_score, 3) if img.composite_score is not None else None,
            "IQA": round(img.iqa_score, 3) if img.iqa_score is not None else None,
            "AVA": round(img.ava_score, 2) if img.ava_score is not None else None,
            "Sharpness": round(img.sharpness, 2) if img.sharpness is not None else None,
            "Faces": img.face_count or 0,
            "Pose": pose,
            "Eyes": eyes,
            "Smile": smile,
            "People": ", ".join(people) if people else "",
        })

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
        use_container_width=True,
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
        if on_click and st.button("View Full Cluster", key=f"cluster_{cluster.cluster_id}"):
            on_click(cluster)

        # Group images by person combination when person_labels exist
        groups = _group_images_by_people(cluster.images, cluster.person_labels)
        multiple_groups = len(groups) > 1

        for group_label, group_images in groups.items():
            if multiple_groups:
                st.markdown(f"**{group_label}** ({len(group_images)} images)")

            images_to_show = group_images[:max_preview] if not show_all else group_images

            if images_to_show:
                render_image_gallery(
                    images_to_show,
                    show_scores=True,
                    show_selection=True,
                    columns=min(4, len(images_to_show)),
                )
                remaining = len(group_images) - len(images_to_show)
                if remaining > 0:
                    st.caption(f"... and {remaining} more images")
            else:
                st.info("No images loaded for this cluster")

        # Debug score table with thumbnails
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
                img = _load_image_for_display(image_path)
                st.image(img, use_container_width=True)

            st.caption(image.filename)

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
