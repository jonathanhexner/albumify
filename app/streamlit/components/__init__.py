"""Reusable Streamlit UI components."""

from .sidebar import render_sidebar
from .album_selector import (
    render_album_selector,
    render_album_creator,
    render_album_list,
)
from .pipeline_runner import (
    render_pipeline_runner,
    render_pipeline_progress,
    render_step_list,
    poll_pipeline_status,
    DEFAULT_PIPELINE,
    MINIMAL_PIPELINE,
    STEP_DISPLAY_NAMES,
)
from .gallery import (
    render_image_gallery,
    render_image_card,
    render_cluster_gallery,
    render_image_comparison,
    render_selected_summary,
)
from .metrics import (
    render_pipeline_metrics,
    render_step_timings,
    render_quality_distribution,
    render_cluster_summary,
    render_people_summary,
    render_metric_card,
    render_results_table,
)
from .people_browser import (
    render_people_grid,
    render_person_card,
    render_person_card_selectable,
    render_person_detail,
    render_people_filter,
    render_people_summary_row,
    render_split_dialog,
    render_merge_dialog,
)
from .export_panel import (
    render_export_panel,
    render_quick_export,
)

__all__ = [
    # Sidebar
    "render_sidebar",
    # Album
    "render_album_selector",
    "render_album_creator",
    "render_album_list",
    # Pipeline
    "render_pipeline_runner",
    "render_pipeline_progress",
    "render_step_list",
    "poll_pipeline_status",
    "DEFAULT_PIPELINE",
    "MINIMAL_PIPELINE",
    "STEP_DISPLAY_NAMES",
    # Gallery
    "render_image_gallery",
    "render_image_card",
    "render_cluster_gallery",
    "render_image_comparison",
    "render_selected_summary",
    # Metrics
    "render_pipeline_metrics",
    "render_step_timings",
    "render_quality_distribution",
    "render_cluster_summary",
    "render_people_summary",
    "render_metric_card",
    "render_results_table",
    # People
    "render_people_grid",
    "render_person_card",
    "render_person_card_selectable",
    "render_person_detail",
    "render_people_filter",
    "render_people_summary_row",
    "render_split_dialog",
    "render_merge_dialog",
    # Export
    "render_export_panel",
    "render_quick_export",
]
