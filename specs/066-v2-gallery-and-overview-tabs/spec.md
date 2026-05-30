# spec-066 — v2 Gallery tab + v2 Overview tab (grouped, last P3 pair)

**Created**: 2026-05-30
**Status**: Draft
**Priority**: P3 (both)
**Predecessors**: spec-042, spec-045, spec-064 (Face Analysis), spec-065 (Merged + Quality)
**Reference baseline**: must keep `tests/face_clustering/e2e_budapest/` green; adds Scenarios G + H. **This spec completes the v2 tab parity** — after this lands, every legacy tab has a v2 equivalent.

Grouped for the same reason as spec-065: both are aggregate views, both compose existing services, neither needs new compute or new schema.

---

## Problem

### Gallery tab

The user wants to browse all clusters cluster-by-cluster as scrollable thumbnail strips, not one-cluster-at-a-time as in Cluster Analysis. Useful for getting a feel for the album's people structure without drilling in.

### Overview tab

Aggregate run-level metrics — total runs in history, average n_clusters, gate pass rates over time, profile usage breakdown. The dashboard view legacy app had as its landing page.

## What we build

### Backend

Gallery composes existing services (no new methods):
- `ClusterAnalysisService.list_clusters()` → for each, fetch first N exemplar thumbnails
- Reuses face_grid component

Overview adds one new aggregate Service:
- `OverviewService.compute_dashboard() -> DashboardMetrics` — typed dataclass with run counts, mean/median n_clusters across all v2 runs, gate pass rates, top-3 profiles used. Reads from `RunHistoryRepository` (action_log) + per-run DB stats.

### Frontend

**`gallery_tab.py`** (~70 LOC):
- Top: filter bar (min cluster size, sort by size desc/asc)
- For each cluster: one row showing first 8 exemplar thumbnails + cluster id + size + "Open in Cluster Analysis" button
- Pagination: 10 clusters per page

**`overview_tab.py`** (~80 LOC):
- 4-metric headline strip (total runs, total faces ever, avg n_clusters, last run age)
- Per-album bar chart (total runs per album)
- Per-profile bar chart (which profiles get used most)
- Time-series chart of n_clusters over recent runs (Plotly)

### Tests

| Layer | File | Cases |
|---|---|---|
| Service synthetic | NEW `test_overview_service_synthetic.py` (5 cases) — Gallery has no new service | per-method |
| Service real | + 1 opt-in `slow` for Overview on the user's real action_log + run dirs |
| Architecture | both tabs in scan |
| Baseline e2e | **Scenario G** (Gallery): load reference run → tab → assert ≥ 1 cluster row with ≥ 1 thumbnail. **Scenario H** (Overview): same → assert 4-metric strip rendered |

## Locked decisions

1. **Gallery has no new Service.** It's pure composition of Cluster Analysis service. Avoids the duplicate-method problem.
2. **Overview aggregates across runs.** It's the first tab that reads from BOTH the global `action_log` AND per-run DBs. Pattern: `OverviewService` constructor takes `(history_repo, runs_dir)`; iterates over `history_repo.find(producer="fc_app_v2", limit=50)` and opens each run's DB. Cached on session.
3. **Sync only.** Aggregation across 50 runs is sub-second on the user's machine. spec-079 lesson holds.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `OverviewService.compute_dashboard()` returns typed `DashboardMetrics` | grep + test |
| AC2 | Both tab files ≤ 80 LOC; no SQL/FS/`cfg.get` | LOC + arch test |
| AC3 | 5 new synthetic service tests + 1 real-fixture pass | pytest |
| AC4 | Baseline e2e Scenarios G + H green | pytest -m budapest |
| AC5 | **v2 tab parity COMPLETE** — every legacy tab has a v2 equivalent | manual checklist in REVIEW |
| AC6 | spec-042 status flipped to Implemented (this spec is the last child) | grep spec-042/spec.md |

## Effort estimate

**~3-4 hours total** (both tabs): Overview service ~1h, two tabs ~1.5h, tests + scenarios ~1h.
