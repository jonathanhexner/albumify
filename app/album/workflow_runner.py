"""
Workflow runner UI with progress tracking.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional


def render_workflow_runner(
    source_directory: Path,
    output_directory: Path,
    album_name: str,
    config_overrides: Dict[str, Any]
):
    """
    Run album workflow with progress display.
    
    Args:
        source_directory: Directory containing images
        output_directory: Where to export results
        album_name: Name of the album
        config_overrides: Configuration overrides from UI
    
    Returns:
        WorkflowResult or None if not run yet
    """
    import streamlit as st
    from sim_bench.album import create_album_workflow
    
    st.header("🚀 Workflow Execution")
    
    st.info(f"**Album:** {album_name}")
    st.info(f"**Source:** {source_directory}")
    st.info(f"**Output:** {output_directory}")
    
    if not source_directory.exists():
        st.error(f"❌ Source directory not found: {source_directory}")
        return None
    
    run_button = st.button("▶️ Start Workflow", type="primary", use_container_width=True)
    
    if not run_button:
        return None
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    detail_text = st.empty()
    
    # Stats display
    stats_cols = st.columns(4)
    with stats_cols[0]:
        processed_metric = st.empty()
    with stats_cols[1]:
        rate_metric = st.empty()
    with stats_cols[2]:
        elapsed_metric = st.empty()
    with stats_cols[3]:
        eta_metric = st.empty()
    
    stage_descriptions = {
        'discover_images': '🔍 Discovering images',
        'preprocess': '⚡ Generating thumbnails',
        'analyze_quality': '📊 Analyzing quality and portraits',
        'filter_quality': '✂️ Filtering by quality',
        'filter_portrait': '👤 Filtering portraits',
        'extract_features': '🧬 Extracting features',
        'cluster_images': '🔗 Clustering images',
        'select_best': '⭐ Selecting best images',
        'export_results': '📤 Exporting results',
        'complete': '✅ Complete!',
    }
    
    # State tracking (mutable for closure)
    state = {
        'start_time': time.time(),
        'last_update': time.time(),
        'processed': 0,
        'last_stage': None
    }
    
    def progress_callback(stage: str, pct: float, operation: str = None, image_name: str = None):
        """Enhanced progress callback with details."""
        progress_bar.progress(pct)
        
        # Main status
        stage_desc = stage_descriptions.get(stage, stage)
        status_text.markdown(f"**{stage_desc}**")
        
        # Detailed info
        if operation and image_name:
            elapsed = time.time() - state['last_update']
            detail_text.text(f"🔧 {operation} | 📄 {image_name} | ⏱️ {elapsed:.1f}s")
            state['processed'] += 1
            state['last_update'] = time.time()
        elif stage != state['last_stage']:
            detail_text.text(f"Stage: {stage_desc}")
            state['last_stage'] = stage
        
        # Update stats
        elapsed_total = time.time() - state['start_time']
        rate = state['processed'] / elapsed_total if elapsed_total > 0 else 0
        
        processed_metric.metric("Processed", state['processed'])
        rate_metric.metric("Rate", f"{rate:.1f} img/s")
        elapsed_metric.metric("Elapsed", f"{elapsed_total:.0f}s")
        
        # Estimate remaining (rough)
        if rate > 0 and state['processed'] > 0:
            # Assume similar processing for remaining items
            eta = (100 - state['processed']) / rate if rate > 0 else 0
            eta_metric.metric("ETA", f"{eta:.0f}s")
        else:
            eta_metric.metric("ETA", "--")
    
    workflow = create_album_workflow(
        source_directory=source_directory,
        album_name=album_name,
        overrides=config_overrides
    )
    
    result = workflow.run(
        source_directory=source_directory,
        output_directory=output_directory,
        progress_callback=progress_callback
    )
    
    progress_bar.progress(1.0)
    status_text.markdown("**✅ Workflow completed successfully!**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", result.total_images)
    
    with col2:
        st.metric("After Filtering", result.filtered_images)
    
    with col3:
        st.metric("Clusters", len(result.clusters))
    
    with col4:
        st.metric("Selected", len(result.selected_images))
    
    with st.expander("📈 Cluster Statistics", expanded=True):
        if result.cluster_stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Min Cluster Size", result.cluster_stats['min_size'])
            
            with col2:
                st.metric("Max Cluster Size", result.cluster_stats['max_size'])
            
            with col3:
                st.metric("Avg Cluster Size", f"{result.cluster_stats['avg_size']:.1f}")
    
    return result


def render_workflow_form():
    """
    Render form for workflow parameters.
    
    Returns:
        Tuple of (source_dir, output_dir, album_name) or None if incomplete
    """
    import streamlit as st
    
    st.header("📂 Album Selection")
    
    source_dir = st.text_input(
        "Source Directory",
        placeholder="e.g., C:/Users/Me/Photos/Vacation2024",
        help="Directory containing images to organize"
    )
    
    album_name = st.text_input(
        "Album Name",
        placeholder="e.g., Summer Vacation 2024",
        help="Name for this album"
    )
    
    output_dir = st.text_input(
        "Output Directory",
        placeholder="e.g., C:/Users/Me/Photos/Organized/Vacation2024",
        help="Where to export selected images"
    )
    
    if not source_dir or not album_name or not output_dir:
        st.warning("⚠️ Please fill in all fields")
        return None
    
    return Path(source_dir), Path(output_dir), album_name
