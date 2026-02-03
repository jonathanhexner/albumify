"""Export panel component for exporting results."""

import streamlit as st
from pathlib import Path
from typing import Optional, Callable

from app.streamlit.api_client import get_client
from app.streamlit.session import add_notification


def render_export_panel(
    job_id: str,
    num_selected: int,
    on_export_complete: Optional[Callable[[str], None]] = None,
) -> None:
    """Render export options panel."""
    st.subheader("Export Results")

    if num_selected == 0:
        st.warning("No images selected to export")
        return

    st.write(f"**{num_selected}** images ready to export")

    with st.form("export_form"):
        output_path = st.text_input("Output Directory", placeholder="C:/Photos/Exported", help="Directory where exported images will be saved")

        st.write("**Options:**")

        col1, col2 = st.columns(2)

        with col1:
            include_selected = st.checkbox("Include selected images", value=True, help="Export images marked as selected")
            include_all = st.checkbox("Include all filtered images", value=False, help="Also export images that passed quality filter but weren't selected")

        with col2:
            organize_by_cluster = st.checkbox("Organize by cluster", value=False, help="Create subfolders for each cluster")
            organize_by_person = st.checkbox("Organize by person", value=False, help="Create subfolders for each detected person")

        copy_mode = st.radio("Copy Mode", options=["copy", "symlink", "move"], horizontal=True, help="How to transfer files to output directory")

        if st.form_submit_button("Export", type="primary"):
            if not output_path:
                st.error("Please enter an output directory")
            else:
                _do_export(job_id, output_path, include_selected, include_all, organize_by_cluster, organize_by_person, copy_mode, on_export_complete)


def _do_export(
    job_id: str,
    output_path: str,
    include_selected: bool,
    include_all: bool,
    organize_by_cluster: bool,
    organize_by_person: bool,
    copy_mode: str,
    on_complete: Optional[Callable[[str], None]] = None,
) -> None:
    """Execute the export operation."""
    output_dir = Path(output_path)

    if not output_dir.parent.exists():
        st.error(f"Parent directory does not exist: {output_dir.parent}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with st.spinner("Exporting images..."):
        client = get_client()
        result = client.export_result(
            job_id=job_id,
            output_path=str(output_dir),
            include_selected=include_selected,
            include_all_filtered=include_all,
            organize_by_cluster=organize_by_cluster,
            organize_by_person=organize_by_person,
            copy_mode=copy_mode,
        )

    exported_count = result.get("exported_count", 0)
    st.success(f"Exported {exported_count} images to {output_path}")
    add_notification(f"Exported {exported_count} images", "success")

    if on_complete:
        on_complete(output_path)


def render_quick_export(job_id: str, num_selected: int, default_path: Optional[str] = None) -> None:
    """Render a compact quick export button."""
    if num_selected == 0:
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        output = st.text_input("Export to", value=default_path or "", placeholder="Output directory", key="quick_export_path", label_visibility="collapsed")

    with col2:
        if st.button("Export", key="quick_export_btn") and output:
            _do_export(job_id, output, True, False, False, False, "copy")
