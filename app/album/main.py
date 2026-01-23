"""
Album Organization - Streamlit App

Main entry point for the album organization UI.
Run with: streamlit run app/album/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.album.session import AlbumSession
from app.album.components import (
    render_config_panel,
    render_workflow_form,
    render_workflow_runner,
    render_results
)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Album Organization",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📸 Photo Album Organization")
    st.markdown("Automatically organize and select best photos from your albums")

    # Initialize session
    AlbumSession.initialize()

    # Sidebar: Navigation + Config
    with st.sidebar:
        st.header("Navigation")
        st.markdown("""
        ### Workflow Steps
        1. ⚙️ Configure settings
        2. 📂 Select album directory
        3. 🚀 Run workflow
        4. 📸 View results
        """)
        st.divider()

        config_overrides = render_config_panel()
        AlbumSession.set_config_overrides(config_overrides)

        st.divider()
        st.markdown("""
        ### About
        This app uses:
        - **IQA** for technical quality
        - **AVA** for aesthetic scoring
        - **MediaPipe** for portrait analysis
        - **DINOv2** for feature extraction
        - **HDBSCAN** for clustering
        """)

    # Main content
    st.divider()

    form_data = render_workflow_form()

    if form_data:
        source_dir, output_dir, album_name = form_data
        st.divider()

        result = render_workflow_runner(
            source_directory=source_dir,
            output_directory=output_dir,
            album_name=album_name,
            config_overrides=config_overrides
        )

        if result:
            AlbumSession.set_result(result)

    st.divider()

    render_results(AlbumSession.get_result())


if __name__ == "__main__":
    main()
