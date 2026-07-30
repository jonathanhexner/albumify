# spec-074 — All-clusters summary table (restore the V1 overview)

**Created**: 2026-06-05
**Status**: Implemented
**Priority**: P2
**Predecessors**: spec-045 (Cluster Analysis), spec-072 (ColumnSpec metric registry)
**Source**: user manual-test feedback 2026-06-05 — "we had a clusters summary in V1; lost it. Want faces/diameter/distance-to-next per cluster, all at once."

---

## Problem

Cluster Analysis drills into ONE cluster. There's no at-a-glance table of **all**
clusters (size, diameter, spread, nearest neighbour + its size + distance). The data
for size/diameter/avg-intra exists cheaply (`get_cluster_rows`), but `nearest_cluster_*`
on `ClusterRow` are **placeholders** (only filled for the selected cluster) — so a
real "distance to next cluster" needs a one-time cross-cluster computation.

## What we build

| Layer | Change |
|---|---|
| `face_cluster/views/cluster_analysis.py` | NEW `ClusterSummaryRow` + `ClusterAnalysisService.cluster_summary()` — computes each cluster's nearest *other* cluster (min exemplar-to-exemplar cosine distance), its id, distance, **and size**. One pass, sub-second for typical runs. |
| `face_cluster/views/cluster_analysis.py` | NEW `CLUSTER_SUMMARY_COLUMNS: List[ColumnSpec]` registry |
| `app/face_clustering_v2/components/cluster_summary_table.py` | NEW `render_cluster_summary(rows) -> Optional[int]` — sortable table via `render_run_table`; returns the clicked cluster_id |
| `app/face_clustering_v2/tabs/cluster_analysis_tab.py` | render the summary above the picker; clicking a row → `_goto_cluster` (selects it in the picker, drills in) |

### `ClusterSummaryRow`
```
cluster_id · size · diameter · avg_intra_dist · n_exemplars
nearest_cluster_id · nearest_cluster_dist · nearest_cluster_size
```

`merge_candidate` flag = `nearest_cluster_dist < merge_candidate_threshold` (from run
metadata) — shown so the user can spot under-merges (ties into issue #1).

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `cluster_summary()` returns one `ClusterSummaryRow` per non-noise cluster, with a real nearest-cluster id/dist/size | synthetic test |
| AC2 | Summary table renders all clusters; sortable; clicking a row selects that cluster (drills in) | AppTest + manual |
| AC3 | Columns driven by `CLUSTER_SUMMARY_COLUMNS` (ColumnSpec) — no inline dict | grep |
| AC4 | budapest Scenario B (Cluster Analysis) stays green | pytest -m budapest -k b |
| AC5 | Service stays Streamlit-free; tab no SQL/FS | arch tests |

## Non-goals
- The "nearest cluster *pairs* below criteria" merge view (issue #1 — separate).
- A new top-level tab (summary lives in the Cluster Analysis tab).
