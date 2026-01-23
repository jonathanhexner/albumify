"""Gallery component for displaying images."""

from pathlib import Path
from typing import Optional
import streamlit as st
from PIL import Image, ImageOps

from sim_bench.album import WorkflowResult


def load_image_for_display(image_path):
    """Load image with EXIF orientation correction."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return img.copy()


def render_gallery(result: WorkflowResult):
    """Render cluster gallery with images."""
    st.subheader("Image Gallery")

    filter_options = ['All Images', 'Clustered Only', 'Filtered Out']
    filter_options += [f"Cluster {cid}" for cid in sorted(result.clusters.keys())]

    view_filter = st.selectbox("View", options=filter_options, index=0)

    if view_filter == 'All Images':
        _render_all_images(result)
    elif view_filter == 'Filtered Out':
        _render_filtered_out(result)
    elif view_filter == 'Clustered Only':
        _render_clusters(result, result.clusters)
    else:
        cluster_id = int(view_filter.split()[1])
        _render_clusters(result, {cluster_id: result.clusters[cluster_id]})


def render_image_card(img_path: str, result: WorkflowResult):
    """Render a single image card with metrics."""
    try:
        img = load_image_for_display(img_path)
        st.image(img, use_container_width=True)

        is_selected = img_path in result.selected_images
        is_filtered = hasattr(result, 'filtered_out') and img_path in result.filtered_out

        if is_filtered:
            st.error("🚫 Filtered")
        elif is_selected:
            st.success("⭐ Selected")

        metric = result.metrics.get(img_path)
        if metric:
            _render_metric_caption(metric)
        else:
            st.caption("No metrics")

    except Exception as e:
        st.error(f"Error: {e}")


def _render_metric_caption(metric):
    """Render metric caption for an image."""
    lines = []

    total = metric.get_composite_score() if hasattr(metric, 'get_composite_score') else None
    if total is not None:
        lines.append(f"**Score: {total:.2f}**")

    parts = []
    if metric.ava_score is not None:
        parts.append(f"AVA:{metric.ava_score:.1f}")
    if metric.iqa_score is not None:
        parts.append(f"IQA:{metric.iqa_score:.2f}")
    if metric.sharpness is not None:
        parts.append(f"Sharp:{metric.sharpness:.2f}")
    if parts:
        lines.append(" | ".join(parts))

    if metric.is_portrait:
        eyes = "👁️" if metric.eyes_open else "😑"
        smile = "😊" if metric.is_smiling else "😐"
        lines.append(f"Portrait: {eyes} {smile}")

    st.caption("  \n".join(lines))


def _render_all_images(result: WorkflowResult):
    """Render all images grouped by status."""
    filtered_out = getattr(result, 'filtered_out', [])
    if filtered_out:
        with st.expander(f"🚫 Filtered Out ({len(filtered_out)} images)", expanded=False):
            cols = st.columns(4)
            for idx, img_path in enumerate(filtered_out):
                with cols[idx % 4]:
                    render_image_card(img_path, result)

    _render_clusters(result, result.clusters)


def _render_filtered_out(result: WorkflowResult):
    """Render only filtered out images."""
    filtered_out = getattr(result, 'filtered_out', [])
    if not filtered_out:
        st.info("No images were filtered out")
        return

    st.markdown(f"### 🚫 Filtered Out ({len(filtered_out)} images)")
    cols = st.columns(4)
    for idx, img_path in enumerate(filtered_out):
        with cols[idx % 4]:
            render_image_card(img_path, result)


def _render_clusters(result: WorkflowResult, clusters_to_show):
    """Render cluster gallery."""
    for cluster_id, images in sorted(clusters_to_show.items()):
        selected_count = sum(1 for img in images if img in result.selected_images)
        st.markdown(f"### 📁 Cluster {cluster_id}")
        st.caption(f"{len(images)} images, {selected_count} selected")

        cols = st.columns(4)
        for idx, img_path in enumerate(images):
            with cols[idx % 4]:
                render_image_card(img_path, result)

        st.divider()
