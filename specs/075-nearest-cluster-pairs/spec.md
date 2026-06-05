# spec-075 — Nearest cluster-pairs view (why isn't this merged?)

**Created**: 2026-06-05 · **Status**: Implemented · **Priority**: P2
**Predecessors**: spec-074 (cluster summary compute), spec-065 (Merged Clusters tab)
**Source**: user feedback #1 — "no merge candidates? at least show what was below criteria (N closest)."

## Problem
Merged Clusters shows only pairs that crossed the candidate threshold (3 rows, all
rejected). The user wants the **N closest cluster pairs regardless**, with why-rejected
when a pair was evaluated. The cross-cluster exemplar distances already exist (spec-074).

## What we build
- `face_cluster/views/cluster_analysis.py`: `NearestPairRow` + `NEAREST_PAIR_COLUMNS` +
  `ClusterAnalysisService.nearest_cluster_pairs(top_n=20)` — all cluster pairs sorted by
  exemplar distance, each with both sizes, `evaluated`/`merged`/`rejection_reason` joined
  from `merge_decisions`. Reuses a shared `_exemplar_matrices()` helper (refactored out of
  `cluster_summary`).
- `app/face_clustering_v2/tabs/merged_clusters_tab.py`: a "Nearest cluster pairs (closest N)"
  table above/below the merge-decisions table (via `ClusterAnalysisService`).

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | `nearest_cluster_pairs` returns pairs sorted by dist; evaluated pairs carry their merge verdict | synthetic test |
| 2 | Renders in Merged Clusters; sortable | AppTest |
| 3 | budapest Scenario E (Merged Clusters) green | pytest -m budapest -k e |
| 4 | Service Streamlit-free; ColumnSpec-driven | arch |
