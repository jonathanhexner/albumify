"""Workflow runner components."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import streamlit as st

from sim_bench.album import AlbumService, WorkflowResult
from sim_bench.album.domain.types import WorkflowStage


STAGE_DESCRIPTIONS = {
    WorkflowStage.DISCOVER: '🔍 Discovering images',
    WorkflowStage.PREPROCESS: '⚡ Generating thumbnails',
    WorkflowStage.ANALYZE: '📊 Analyzing quality and portraits',
    WorkflowStage.FILTER_QUALITY: '✂️ Filtering by quality',
    WorkflowStage.FILTER_PORTRAIT: '👤 Filtering portraits',
    WorkflowStage.EXTRACT_FEATURES: '🧬 Extracting features',
    WorkflowStage.CLUSTER: '🔗 Clustering images',
    WorkflowStage.SELECT: '⭐ Selecting best images',
    WorkflowStage.EXPORT: '📤 Exporting results',
    WorkflowStage.COMPLETE: '✅ Complete!',
}


def render_workflow_form() -> Optional[Tuple[Path, Path, str]]:
    """
    Render form for workflow parameters.
    
    Returns:
        Tuple of (source_dir, output_dir, album_name) or None if incomplete
    """
    st.header("📂 Album Selection")

    source_dir = st.text_input(
        "Source Directory",
        placeholder="e.g., C:/Users/Me/Photos/Vacation2024"
    )

    album_name = st.text_input(
        "Album Name",
        placeholder="e.g., Summer Vacation 2024"
    )

    output_dir = st.text_input(
        "Output Directory",
        placeholder="e.g., C:/Users/Me/Photos/Organized/Vacation2024"
    )

    if not source_dir or not album_name or not output_dir:
        st.warning("⚠️ Please fill in all fields")
        return None

    return Path(source_dir), Path(output_dir), album_name


def render_workflow_runner(
    source_directory: Path,
    output_directory: Path,
    album_name: str,
    config_overrides: Dict[str, Any]
) -> Optional[WorkflowResult]:
    """
    Run album workflow with progress display.
    
    Returns:
        WorkflowResult or None if not run yet
    """
    from sim_bench.config import get_global_config

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

    # Build config
    config = get_global_config().to_dict()
    config = _deep_merge(config, config_overrides)

    # Create service
    service = AlbumService(config)

    # Progress UI
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    detail_text = st.empty()

    stats_cols = st.columns(4)
    processed_metric = stats_cols[0].empty()
    rate_metric = stats_cols[1].empty()
    elapsed_metric = stats_cols[2].empty()
    eta_metric = stats_cols[3].empty()

    state = {'start': time.time(), 'processed': 0, 'last_stage': None}

    def progress_callback(stage: WorkflowStage, pct: float, detail: str = None):
        progress_bar.progress(pct)
        stage_desc = STAGE_DESCRIPTIONS.get(stage, str(stage))
        status_text.markdown(f"**{stage_desc}**")

        if detail:
            state['processed'] += 1
            detail_text.text(f"📄 {detail}")

        elapsed = time.time() - state['start']
        rate = state['processed'] / elapsed if elapsed > 0 else 0

        processed_metric.metric("Processed", state['processed'])
        rate_metric.metric("Rate", f"{rate:.1f} img/s")
        elapsed_metric.metric("Elapsed", f"{elapsed:.0f}s")
        eta_metric.metric("ETA", "...")

    # Execute
    result = service.organize_album(
        source_directory=source_directory,
        output_directory=output_directory,
        progress_callback=progress_callback
    )

    # Summary
    st.success(f"✅ Workflow complete!")

    cols = st.columns(4)
    cols[0].metric("Total", result.total_images)
    cols[1].metric("Filtered", result.filtered_images)
    cols[2].metric("Clusters", len(result.clusters))
    cols[3].metric("Selected", len(result.selected_images))

    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
