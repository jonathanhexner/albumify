# spec-076 — Face Metrics: gate/reason column (+ confirm drill-in)

**Created**: 2026-06-05 · **Status**: Implemented · **Priority**: P2
**Predecessors**: spec-069 (Face Metrics), spec-070 (face overlays), spec-072 (metric registry)
**Source**: user feedback #4 — "know if the face was gate filtered and why; click thumbnail → face/image with bbox + pose."

## Problem
Face Metrics shows status (assigned/unassigned) but not *why* a face was held out. The
faces table has `rejection_reason` (154 rows on the reference run, e.g. `top_k_per_image`).
Drill-in already exists — the table's row-click seeds `selected_face_id` → Face Analysis
renders the bbox + head-pose overlay (spec-070). So this spec = surface the reason.

## What we build
- `face_cluster/views/face_metrics.py`: `FaceMetricRow.rejection_reason`; service populates it.
- `app/face_clustering_v2/tabs/face_metrics_tab.py`: add a "reason" structural column; caption noting "click a row → Face Analysis".

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | `FaceMetricRow.rejection_reason` populated from the faces table | unit test |
| 2 | "reason" column visible in the table | AppTest |
| 3 | row-click drill-in to Face Analysis works | existing (spec-070) — confirm |
| 4 | budapest path unaffected | AppTest 0 exc |

Note (thumbnail-click): Streamlit can't make a bare image clickable; the row-click (which
contains the thumbnail) is the supported equivalent and already works.
