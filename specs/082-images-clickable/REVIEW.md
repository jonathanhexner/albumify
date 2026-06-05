# REVIEW — spec-082 Clickable thumbnails (Images + Gallery)

**2026-06-05** · scope: spec-082 diff · ✅ no High findings.

## Files
- `face_cluster/views/image_metrics.py` — NEW `populated_columns(rows)` (drop
  all-None columns); Streamlit-free, unit-tested.
- `app/face_clustering_v2/tabs/images_tab.py` — populated-column defaults;
  `ImageColumn` thumbnails; parallel `_encode_thumb` + `st.spinner`; row-select
  → Image Analysis.
- `app/face_clustering_v2/components/cluster_strip.py` — per-face "Open" button
  → Face Analysis.
- `app/face_clustering_v2/_run_context.py` — NEW generic `cached_service`.
- NEW `app/face_clustering_v2/components/nearest_pairs.py` — extracted render.
- `app/face_clustering_v2/tabs/merged_clusters_tab.py` — uses the two helpers
  above; **118 → 73 LOC** (back under the 90 budget; the LOC guard had been
  red since spec-079).
- NEW `tests/face_clustering/views/test_image_populated_columns.py` (4).

## Checklist
| § | Finding |
|---|---|
| Correctness | Images: only non-None columns offered (verified screenshot — Composite/IQA/AVA/Sharpness gone, Faces/Gate/Width/Height kept). Row-click → Image Analysis (screenshot: "2 faces · 1 clustered · 1 noise · 0 filtered"). Gallery: button → `selected_face_id` + `navigate_to("Face Analysis")` (AppTest fid=6 → active_page switched, 0 exc). ✅ |
| Perf | Thumbnail cold-start **12 s → 3.3 s** via `ThreadPoolExecutor` (PIL drops the GIL in JPEG decode/encode); `st.spinner` removes the silent-blank window; cached per run dir. ✅ |
| Reuse | Gallery drill-in reuses the exact `selected_face_id` path the Face Metrics rows use; Image Analysis view reused unchanged; `cached_service` generalises the existing `cached_cluster_service`. ✅ |
| Layering | `populated_columns` + `nearest_pairs` render moved out of the tab into service/component; tab back to a thin orchestrator. Arch guards green (245 passed). ✅ |
| Tests | unit (4) + AppTest Images/Gallery/Merged-Clusters 0 exc + real-browser screenshots. Canvas row-pick not Playwright-`<img>`-addressable (SIGHTING-091) → verified by screenshot, as with Face Metrics/History. ✅ |
| Regression caught | the merged_clusters LOC guard (red since spec-079) fixed in the same pass rather than dismissed. ✅ |

**No High → Implemented** pending the budapest e2e baseline gate.
