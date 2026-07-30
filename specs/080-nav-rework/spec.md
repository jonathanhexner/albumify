# spec-080 — Navigation rework (click-to-open) + single-page render

**Created**: 2026-06-05 · **Status**: Implemented · **Priority**: P2
**Source**: user feedback #2 — "clicking Open doesn't open Face Analysis."

## Problem
The app used `st.tabs`, which **cannot be switched programmatically** — so "Open"
buttons (face grid → Face Analysis, Gallery → Cluster Analysis, Face Metrics row →
Face Analysis) set state but never changed the view. Secondary issue: `st.tabs`
renders ALL 11 tab bodies every rerun (the eager-render heaviness behind earlier e2e
slowness).

## What we build
- NEW `app/face_clustering_v2/_nav.py` — `render_nav(pages)` (a horizontal radio bound
  to `session_state['active_page']`, renders ONLY the active page) + `navigate_to(name)`.
  `navigate_to` sets a PENDING flag that `render_nav` applies **before** the radio is
  created (Streamlit forbids mutating a live widget's key). Honors `?page=<name>`.
- `app/face_clustering_v2/main.py` — replace the `st.tabs` block with `render_nav([...])`.
- Wire `navigate_to` into the "Open" actions: `face_grid` → Face Analysis,
  `cluster_strip` → Cluster Analysis, `face_metrics_tab` row-select → Face Analysis
  (loop-safe via `_fm_last_pick`).
- e2e: `st.tabs` → radio means `get_by_role('tab')` no longer matches. New conftest
  helper `goto_page(page, name)`; all scenarios updated.

## Wins
- "Open" buttons now switch view (verified Gallery→Cluster Analysis, cluster 1 selected).
- Only the active page renders → much lighter reruns (removes the all-tabs eager cost;
  the hidden-tab `img`/`stMetric` selector hazards in the e2e disappear).

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | "Open" in Gallery/face-grid/Face-Metrics switches to the target view | browser + e2e |
| 2 | Only the active page's render fn runs per rerun | by construction |
| 3 | budapest B–I green with the new nav | pytest -m budapest |
| 4 | `?page=` deep-link opens a view | AppTest |
