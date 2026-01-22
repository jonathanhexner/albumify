"""
Album Organization - Streamlit App

Main entry point for the album organization UI.
Run with: streamlit run app/album/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.album.config_panel import render_album_config
from app.album.workflow_runner import render_workflow_form, render_workflow_runner
from app.album.results_viewer import render_results


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
    
    st.sidebar.header("Navigation")
    
    st.sidebar.markdown("""
    ### Workflow Steps
    1. ⚙️ Configure settings
    2. 📂 Select album directory
    3. 🚀 Run workflow
    4. 📸 View results
    """)
    
    st.sidebar.divider()
    
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    config_overrides = render_album_config()
    
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
            st.session_state.workflow_result = result
    
    st.divider()
    
    render_results(st.session_state.workflow_result)
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### About
    
    This app uses:
    - **IQA** for technical quality
    - **AVA** for aesthetic scoring
    - **MediaPipe** for portrait analysis
    - **DINOv2** for feature extraction
    - **HDBSCAN** for clustering
    
    All settings can be customized above.
    """)


if __name__ == "__main__":
    main()
