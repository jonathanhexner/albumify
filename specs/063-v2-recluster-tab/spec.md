# spec-063 — v2 Recluster tab

**Created**: 2026-05-30
**Status**: Draft
**Priority**: P1 (last P1 tab in spec-042's parity umbrella)
**Predecessors**: spec-042, spec-045 (Cluster Analysis), spec-059 (SQLAlchemy stack)
**Reference baseline**: `tests/face_clustering/test_v2_e2e_budapest_baseline.py` — every commit on this spec must keep it green.

---

## Problem

The legacy FC App lets the user re-cluster a prior run with different `K` / `distance_threshold` / merge params without re-running the producer chain (which is slow — face detection + alignment + embedding). The v2 app can't do this today. spec-042 lists Recluster as P1.

The clustering algorithm is already in `FCAppRunner` (the unified 8-step chain). What's missing is a *runner mode* that loads `context.face_records` from a prior run dir instead of running detect/align/embed, and a tab that orchestrates "pick prior run + tweak params + run."

## What we build

### Backend

**`FCAppRunner.recluster(prior_run_dir, step_configs) -> FCAppRunResult`**
- Loads `face_records` from the prior run via `RunStore(prior_run_dir).faces()`
- Pre-populates `context.face_records` (skipping the producer steps)
- Calls `self.run(context, step_configs=step_configs)` — same 8-step chain
- Sets `context.parent_run_id = prior_run_dir.name`
- Returns the same `FCAppRunResult` shape as `run()`

~10 LOC. Reuses everything `FCAppRunner.run` already does.

**Service: `ReclusterService`** in `face_cluster/views/recluster.py`
- `list_recent_runs(limit=20) -> list[RunPickerEntry]` — reuses spec-050's run-picker source
- `recluster_async(prior_run_dir, params: FCParams) -> AsyncHandle[ReclusterResult]` — wraps the runner. Spec-079 lesson: keep the heavy compute *off* the Streamlit thread, but DON'T use AsyncHandle in the UI; the tab uses `st.spinner` + sync.
- Actually: use synchronous `recluster(prior_run_dir, params)` returning a typed `ReclusterResult` (snapshot dir, n_clusters, n_faces, parent_run_id). spec-079's lesson is that async doesn't fit Streamlit.

### Frontend

**`app/face_clustering_v2/tabs/recluster_tab.py`** — orchestrator (≤ 80 LOC):
1. Prior-run picker (reuses spec-050's `render_run_picker`)
2. Params editor (reuses spec-041's `UI_SPEC` + `widget_factory.render_field` to render every FCParams field as a widget)
3. "Run recluster" button → `st.spinner` → `service.recluster(...)` → success toast + writes `current_run_dir` to session_state pointing at the new snapshot dir

### Tests

| Layer | File | Cases |
|---|---|---|
| Repository | n/a — reuses existing `ClusterAnalysisRepository` for prior-run reads | — |
| Service synthetic | `tests/face_clustering/views/test_recluster_service_synthetic.py` | 6: list_recent_runs, recluster with valid prior + default params, recluster with tightened K, missing prior_run_dir raises, snapshot dir created, parent_run_id set |
| Service real | + 1 in same file (opt-in `slow`): recluster the Budapest reference run with profile_4; assert n_clusters in [12, 18] (band around the 15-cluster baseline) |
| Architecture | extend existing `test_cluster_analysis_tab.py`-style guards to Recluster tab (no SQL / FS / `cfg.get` literals; ≤ 80 LOC) |
| Baseline gate | **MUST add a Scenario C to `test_v2_e2e_budapest_baseline.py`**: open Recluster tab → pick reference run → click Run → assert n_clusters in [12, 18] |

## Locked decisions

1. **Sync compute, not async.** spec-079's lesson: AsyncHandle doesn't fit Streamlit. Recluster on Budapest takes ~5-15 s (no producer chain) — `st.spinner` is the right shape.
2. **Snapshot output, not in-place mutation.** Reusing the spec-045 force-merge convention: write a sibling run dir, never mutate parent.
3. **FCParams as the single param surface.** No separate "recluster params" type. Reuses spec-041's `UI_SPEC` for widgets.
4. **No new schema.** Snapshot run dirs use the v5 layout (face_clustering.db top-level). RunExporter writes them.
5. **Parent linkage via `pipeline_run.json`'s `parent_run_id`.** History tab already shows this. No new column.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `FCAppRunner.recluster(prior_run_dir, step_configs)` exists with same return type as `.run()` | grep + test |
| AC2 | `recluster_tab.py` ≤ 80 LOC; no SQL / FS / `cfg.get` literals | LOC count + arch test |
| AC3 | Service synthetic suite: 6 cases green | pytest |
| AC4 | Service real suite (opt-in slow): reclustering the Budapest reference run with profile_4 yields n_clusters ∈ [12, 18] | pytest -m slow |
| AC5 | Baseline e2e Scenario C green | pytest -m budapest |
| AC6 | Snapshot dir's `pipeline_run.json` has `parent_run_id == <prior_run_id>` | test |
| AC7 | History tab shows the new snapshot run with parent linkage visible | manual + Scenario C assertion |

## Effort estimate

**~4-6 hours**: Service ~1h, runner mode ~1h, tab + widget wiring ~1.5h, tests ~1.5h, baseline e2e Scenario C ~1h.
