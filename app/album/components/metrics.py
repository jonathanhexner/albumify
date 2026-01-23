"""Metrics and performance components."""

from pathlib import Path
import streamlit as st
import pandas as pd

from sim_bench.album import WorkflowResult


def render_metrics(result: WorkflowResult):
    """Render detailed metrics and statistics."""
    st.subheader("Image Metrics")

    if not result.metrics:
        st.info("No metrics available")
        return

    filtered_out = set(getattr(result, 'filtered_out', []))

    rows = []
    for img_path, metric in result.metrics.items():
        if img_path in result.selected_images:
            status = '⭐ Selected'
        elif img_path in filtered_out:
            status = '🚫 Filtered'
        else:
            status = 'Clustered'

        total = metric.get_composite_score() if hasattr(metric, 'get_composite_score') else None

        rows.append({
            'Image': Path(img_path).name,
            'Status': status,
            'Score': f"{total:.2f}" if total is not None else 'N/A',
            'AVA': f"{metric.ava_score:.1f}" if metric.ava_score else 'N/A',
            'IQA': f"{metric.iqa_score:.2f}" if metric.iqa_score else 'N/A',
            'Sharp': f"{metric.sharpness:.2f}" if metric.sharpness else 'N/A',
            'Portrait': '✓' if metric.is_portrait else '',
            'Eyes': '👁️' if metric.eyes_open else '😑' if metric.is_portrait else '',
            'Smile': '😊' if metric.is_smiling else '😐' if metric.is_portrait else '',
            'Cluster': metric.cluster_id if metric.cluster_id is not None else '-',
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "album_metrics.csv", "text/csv")

    # Quality distribution
    st.subheader("Quality Distribution")
    col1, col2 = st.columns(2)

    with col1:
        iqa_scores = [m.iqa_score for m in result.metrics.values() if m.iqa_score]
        if iqa_scores:
            st.write("**IQA Scores**")
            st.bar_chart(iqa_scores)

    with col2:
        ava_scores = [m.ava_score for m in result.metrics.values() if m.ava_score]
        if ava_scores:
            st.write("**AVA Scores**")
            st.bar_chart(ava_scores)


def render_performance(result: WorkflowResult):
    """Render performance metrics and telemetry."""
    st.subheader("⚡ Performance Metrics")

    if not result.telemetry:
        st.info("No telemetry data available")
        return

    telemetry = result.telemetry

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Time", f"{telemetry.total_duration_sec:.1f}s")
    col2.metric("Operations", len(telemetry.timings))

    if telemetry.timings:
        slowest = max(telemetry.timings, key=lambda t: t.duration_sec)
        col3.metric("Slowest", slowest.name)

    # Timing breakdown
    st.markdown("### Timing Breakdown")

    if telemetry.timings:
        timing_data = [{
            'Operation': t.name,
            'Duration (s)': f"{t.duration_sec:.2f}",
            'Count': t.count,
            'Avg/Item (s)': f"{t.avg_per_item:.3f}"
        } for t in telemetry.timings]

        st.dataframe(pd.DataFrame(timing_data), use_container_width=True)

        # Bar chart
        st.markdown("### Time Distribution")
        chart_data = pd.DataFrame({
            'Operation': [t.name for t in telemetry.timings],
            'Duration': [t.duration_sec for t in telemetry.timings]
        })
        st.bar_chart(chart_data.set_index('Operation'))

    # Metadata
    if telemetry.metadata:
        st.markdown("### Run Metadata")
        cols = st.columns(4)

        meta_items = [
            ('total_images', "Total Images"),
            ('filtered_images', "Filtered"),
            ('num_clusters', "Clusters"),
            ('selected_images', "Selected")
        ]

        for idx, (key, label) in enumerate(meta_items):
            if key in telemetry.metadata:
                cols[idx].metric(label, telemetry.metadata[key])

        if telemetry.metadata.get('preprocessing_enabled'):
            st.success("✅ Thumbnail preprocessing was enabled")
        else:
            st.warning("⚠️ Thumbnail preprocessing was disabled")

    # Download
    if result.export_path:
        telemetry_file = result.export_path / f"telemetry_{result.run_id}.json"
        if telemetry_file.exists():
            st.download_button(
                "📥 Download Telemetry JSON",
                telemetry_file.read_text(),
                f"telemetry_{result.run_id}.json",
                "application/json"
            )
