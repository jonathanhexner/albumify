"""
Results viewer UI for album organization.
"""

from pathlib import Path
from typing import Optional
from PIL import Image, ImageOps


def load_image_for_display(image_path):
    """Load image with EXIF orientation correction for proper display."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return img.copy()  # Return copy since context manager closes file


def render_results(result):
    """
    Display workflow results with cluster gallery.
    
    Args:
        result: WorkflowResult from workflow execution
    """
    import streamlit as st
    
    if result is None:
        st.info("👆 Run workflow to see results")
        return
    
    st.header("📸 Results")
    
    st.success(f"✅ Selected {len(result.selected_images)} best images from {len(result.clusters)} clusters")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Gallery", "📊 Metrics", "⚡ Performance", "📁 Export"])
    
    with tab1:
        render_gallery(result)
    
    with tab2:
        render_metrics(result)
    
    with tab3:
        render_performance(result)
    
    with tab4:
        render_export_info(result)


def render_gallery(result):
    """Render cluster gallery with images."""
    import streamlit as st
    
    st.subheader("Cluster Gallery")
    
    cluster_filter = st.selectbox(
        "Filter by Cluster",
        options=['All'] + [f"Cluster {cid}" for cid in sorted(result.clusters.keys())],
        index=0
    )
    
    if cluster_filter == 'All':
        clusters_to_show = result.clusters
    else:
        cluster_id = int(cluster_filter.split()[1])
        clusters_to_show = {cluster_id: result.clusters[cluster_id]}
    
    for cluster_id, images in sorted(clusters_to_show.items()):
        st.markdown(f"### 📁 Cluster {cluster_id}")
        st.caption(f"{len(images)} images, {sum(1 for img in images if img in result.selected_images)} selected")
        
        cols = st.columns(4)
        
        for idx, img_path in enumerate(images):
            with cols[idx % 4]:
                try:
                    img = load_image_for_display(img_path)
                    st.image(img, use_column_width=True)
                    
                    if img_path in result.selected_images:
                        st.success("⭐ Selected")
                    
                    metric = result.metrics.get(img_path)
                    if metric:
                        caption_parts = []
                        if metric.iqa_score:
                            caption_parts.append(f"IQA: {metric.iqa_score:.2f}")
                        if metric.ava_score:
                            caption_parts.append(f"AVA: {metric.ava_score:.1f}")
                        
                        st.caption(" | ".join(caption_parts))
                
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        
        st.divider()


def render_metrics(result):
    """Render detailed metrics and statistics."""
    import streamlit as st
    import pandas as pd
    
    st.subheader("Image Metrics")
    
    if not result.metrics:
        st.info("No metrics available")
        return
    
    rows = []
    for img_path, metric in result.metrics.items():
        rows.append({
            'Image': Path(img_path).name,
            'IQA': metric.iqa_score if metric.iqa_score else 'N/A',
            'AVA': metric.ava_score if metric.ava_score else 'N/A',
            'Sharpness': metric.sharpness if metric.sharpness else 'N/A',
            'Portrait': '✓' if metric.is_portrait else '✗',
            'Eyes Open': '✓' if metric.eyes_open else '✗' if metric.is_portrait else '-',
            'Smiling': '✓' if metric.is_smiling else '✗' if metric.is_portrait else '-',
            'Cluster': metric.cluster_id if metric.cluster_id is not None else 'Noise',
            'Selected': '⭐' if img_path in result.selected_images else '',
        })
    
    df = pd.DataFrame(rows)
    
    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )
    
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Metrics CSV",
        data=csv,
        file_name="album_metrics.csv",
        mime="text/csv"
    )
    
    st.subheader("Quality Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        iqa_scores = [m.iqa_score for m in result.metrics.values() if m.iqa_score]
        if iqa_scores:
            st.write("**IQA Scores**")
            st.bar_chart([iqa_scores])
    
    with col2:
        ava_scores = [m.ava_score for m in result.metrics.values() if m.ava_score]
        if ava_scores:
            st.write("**AVA Scores**")
            st.bar_chart([ava_scores])


def render_performance(result):
    """Render performance metrics and telemetry data."""
    import streamlit as st
    import json
    import pandas as pd
    from pathlib import Path
    
    st.subheader("⚡ Performance Metrics")
    
    if not result.telemetry:
        st.info("ℹ️ No telemetry data available")
        return
    
    telemetry = result.telemetry
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Time", f"{telemetry.total_duration_sec:.1f}s")
    
    with col2:
        st.metric("Operations", len(telemetry.timings))
    
    with col3:
        if telemetry.timings:
            slowest = max(telemetry.timings, key=lambda t: t.duration_sec)
            st.metric("Slowest Operation", slowest.name)
    
    # Timing breakdown table
    st.markdown("### Timing Breakdown")
    
    if telemetry.timings:
        timing_data = []
        for timing in telemetry.timings:
            timing_data.append({
                'Operation': timing.name,
                'Duration (s)': f"{timing.duration_sec:.2f}",
                'Count': timing.count,
                'Avg per Item (s)': f"{timing.avg_per_item:.3f}"
            })
        
        df = pd.DataFrame(timing_data)
        st.dataframe(df, use_container_width=True)
        
        # Bar chart of timings
        st.markdown("### Time Distribution")
        chart_data = pd.DataFrame({
            'Operation': [t.name for t in telemetry.timings],
            'Duration (seconds)': [t.duration_sec for t in telemetry.timings]
        })
        st.bar_chart(chart_data.set_index('Operation'))
    
    # Metadata
    if telemetry.metadata:
        st.markdown("### Run Metadata")
        
        meta_cols = st.columns(4)
        if 'total_images' in telemetry.metadata:
            with meta_cols[0]:
                st.metric("Total Images", telemetry.metadata['total_images'])
        if 'filtered_images' in telemetry.metadata:
            with meta_cols[1]:
                st.metric("Filtered Images", telemetry.metadata['filtered_images'])
        if 'num_clusters' in telemetry.metadata:
            with meta_cols[2]:
                st.metric("Clusters", telemetry.metadata['num_clusters'])
        if 'selected_images' in telemetry.metadata:
            with meta_cols[3]:
                st.metric("Selected", telemetry.metadata['selected_images'])
        
        # Preprocessing info
        if telemetry.metadata.get('preprocessing_enabled'):
            st.success("✅ Thumbnail preprocessing was enabled")
        else:
            st.warning("⚠️ Thumbnail preprocessing was disabled")
    
    # Download telemetry JSON
    if result.export_path:
        telemetry_file = result.export_path / f"telemetry_{result.run_id}.json"
        if telemetry_file.exists():
            st.download_button(
                "📥 Download Telemetry JSON",
                data=telemetry_file.read_text(),
                file_name=f"telemetry_{result.run_id}.json",
                mime="application/json"
            )


def render_export_info(result):
    """Render export information and download options."""
    import streamlit as st
    
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
        st.info("ℹ️ Images were not exported (no output directory specified)")
    
    st.subheader("Selected Images")
    
    selected_list = "\n".join([f"- {Path(p).name}" for p in result.selected_images])
    st.text_area(
        "Selected image filenames:",
        value=selected_list,
        height=200
    )
    
    st.download_button(
        "📥 Download Selected List",
        data=selected_list,
        file_name="selected_images.txt",
        mime="text/plain"
    )
