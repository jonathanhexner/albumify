"""Results component - main results view orchestrator."""

from pathlib import Path
from typing import Optional
import streamlit as st

from sim_bench.album import WorkflowResult
from app.album.components.gallery import render_gallery
from app.album.components.metrics import render_metrics, render_performance


def render_results(result: Optional[WorkflowResult]):
    """Display workflow results with tabs."""
    if result is None:
        st.info("👆 Run workflow to see results")
        return

    st.header("📸 Results")

    filtered_count = len(getattr(result, 'filtered_out', []))
    st.success(
        f"✅ Selected {len(result.selected_images)} best images from {len(result.clusters)} clusters  \n"
        f"📊 {result.total_images} total → {result.filtered_images} passed → {len(result.selected_images)} selected"
    )

    if filtered_count > 0:
        st.warning(f"🚫 {filtered_count} images filtered out")

    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Gallery", "📊 Metrics", "⚡ Performance", "📁 Export"])

    with tab1:
        render_gallery(result)

    with tab2:
        render_metrics(result)

    with tab3:
        render_performance(result)

    with tab4:
        render_export_info(result)


def render_export_info(result: WorkflowResult):
    """Render export information and download options."""
    st.subheader("Export Information")

    if result.export_path:
        st.success(f"✅ Images exported to: `{result.export_path}`")
        st.info(f"📊 **{len(result.selected_images)}** images exported")

        if result.export_path.is_dir():
            st.markdown("**Directory structure:**")
            st.code(f"{result.export_path}/\n├── cluster_0/\n├── cluster_1/\n├── thumbnails/\n└── metadata.json")
        elif result.export_path.suffix == '.zip':
            st.markdown(f"**ZIP archive:** `{result.export_path.name}`")
            st.caption(f"Size: {result.export_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        st.info("ℹ️ No output directory specified")

    st.subheader("Selected Images")

    selected_list = "\n".join([f"- {Path(p).name}" for p in result.selected_images])
    st.text_area("Selected filenames:", value=selected_list, height=200)

    st.download_button(
        "📥 Download Selected List",
        selected_list,
        "selected_images.txt",
        "text/plain"
    )
