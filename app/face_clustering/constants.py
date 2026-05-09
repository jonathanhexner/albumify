"""Module-level constants shared across face_clustering app tabs."""
from __future__ import annotations

_RUN_FILE_DEFS = [
    {
        "file":        "pipeline_run.json",
        "format":      "JSON",
        "description": "Run record: config, per-stage status/timing, final summary, errors",
        "schema":      "run_id, source_album, started_at, config{distance_threshold,blur_min,...}, "
                       "stages{name:{status,elapsed_s}}, status, summary{n_faces,n_core,n_clusters,n_noise}",
    },
    {
        "file":        "faces.csv",
        "format":      "CSV",
        "description": "One row per detected face (core + holdout)",
        "schema":      "face_id, image_path, image_id, crop_path, cluster_id, is_core, "
                       "blur_score, area, yaw, pitch, roll",
    },
    {
        "file":        "clusters.csv",
        "format":      "CSV",
        "description": "One row per cluster",
        "schema":      "cluster_id, size, exemplar_face_ids, diameter",
    },
    {
        "file":        "embeddings.npy",
        "format":      "NumPy",
        "description": "L2-normalised ArcFace embeddings, float32, one row per face",
        "schema":      "shape (n_faces x 512) — row order matches faces.csv",
    },
    {
        "file":        "embedding_face_ids.npy",
        "format":      "NumPy",
        "description": "face_id for each row in embeddings.npy",
        "schema":      "shape (n_faces,) int32",
    },
    {
        "file":        "crop_manifest.json",
        "format":      "JSON",
        "description": "Mapping from face_id (str) to relative path of its aligned crop",
        "schema":      '{"0": "crops/face_0000_aligned.jpg", "1": "crops/face_0001_aligned.jpg", ...}',
    },
    {
        "file":        "export_summary.json",
        "format":      "JSON",
        "description": "High-level summary saved by the export stage",
        "schema":      "source_album, run_id, created_at, n_faces, n_core, n_clusters, n_noise, config",
    },
    {
        "file":        "crops/",
        "format":      "JPEG",
        "description": "Aligned face crops, 112x112 px (ArcFace input size), one per face",
        "schema":      "crops/face_XXXX_aligned.jpg  (XXXX = zero-padded face_id)",
    },
    {
        "file":        "logs/",
        "format":      "Text",
        "description": "Per-run pipeline log capturing all stages at DEBUG level",
        "schema":      "logs/run_YYYYMMDD_HHMMSS.log",
    },
]

_LOG_LIVE_LINES = 30
_LOG_MAX_STORED = 500

_STAGE_ORDER = [
    "discover", "embed", "quality", "crops", "cluster",
    "exemplars", "export", "split", "merge", "attach",
]

_STATUS_ICON = {
    "pending": "...",
    "running": ">>",
    "done":    "OK",
    "failed":  "!!",
}

_GALLERY_FILTERS   = ["All", "Merged", "Rejected", "Near Misses (3/4)", "Contested (1-3)"]
_GALLERY_SORTS     = ["Exemplar distance", "Gates passed (contested first)"]
_GALLERY_PAGE_SIZE = 10

_CLUSTER_THUMB_SIZE   = 80
_CLUSTER_SORT_OPTIONS = ["size", "diameter"]

_GROUP_FILTERS   = ["All", "Review Only", "Auto-Approve", "Auto-Reject"]
_GROUP_PAGE_SIZE = 10

_CONF_COLOR = {
    "auto_approve": "#4daa6e",
    "review":       "#e0a030",
    "auto_reject":  "#cc6666",
}
_CONF_LABEL = {
    "auto_approve": "AUTO-APPROVE",
    "review":       "REVIEW",
    "auto_reject":  "AUTO-REJECT",
}
