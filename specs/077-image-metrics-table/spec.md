# spec-077 — Per-image metrics table (controllable columns)

**Created**: 2026-06-05 · **Status**: Implemented · **Priority**: P2
**Predecessors**: spec-072 (ColumnSpec registry), spec-040 (per-run DB images table)
**Source**: user feedback #5 — "what gates apply to images? want a similar table there; columns controllable."

## Problem
Gates today are per-FACE; images carry their own scores (iqa / ava / sharpness / composite)
plus a `filter_passed` flag and `n_faces`. There's no per-image table. Also: the user wants
to choose which columns are shown.

## What we build
- `sim_bench/run_db/store.py`: `ImageRow` + `RunStore.list_images()` (reads the `images` table).
- `face_cluster/views/image_metrics.py`: `ImageMetricsService.list_images()` (passthrough) +
  `IMAGE_METRIC_COLUMNS` registry.
- `app/face_clustering_v2/tabs/images_tab.py`: NEW tab — a multiselect of columns + a sortable
  table (registry-driven). Default columns sensible; user picks the rest.
- `app/face_clustering_v2/main.py`: wire the "Images" tab.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | `list_images()` returns one row per image with scores + filter_passed + n_faces | unit test |
| 2 | Table renders; column multiselect controls shown columns; sortable | AppTest |
| 3 | Columns driven by `IMAGE_METRIC_COLUMNS` (ColumnSpec) | grep |
| 4 | Service Streamlit-free; tab ≤80 LOC, no SQL/FS | arch |
