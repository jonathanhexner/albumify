# spec-079 — Manual-test fixes batch (image aspect, disposition, merge clarity)

**Created**: 2026-06-05 · **Status**: Implemented · **Priority**: P2
**Source**: user manual-test feedback (round 2) — issues #1, #2, #4, #5.

## Fixes
- **#1 Face Analysis stretched image** — `face_bbox_overlay` set a fixed height +
  `use_container_width`, distorting aspect. Now sizes the figure to the image's
  proportions (capped 720px) + `scaleanchor="x"` so it can't distort.
- **#5 Merged Clusters unclear** — the nearest-pairs table rendered raw `read()` values
  (bools showed `True/False`, reasons blank). Now uses `display()`; columns are explicit:
  `Evaluated? yes/no`, `Merged? yes/no`, `Status / why not merged` (full reason, or
  "not a merge candidate (too far apart)").
- **#4 Face Metrics status ambiguous** — `unassigned` conflated *noise* with *gate-filtered*.
  Added a clear 3-way `disposition` (clustered / noise / filtered) on `FaceMetricRow`;
  table now shows `disposition` + `cluster` (C{id}) + `gate` (reason or "passed"); the Show
  filter uses the 3-way. (Reference run: clustered=107, noise=79, filtered=154.)
- **#2 (stopgap)** Cluster Analysis "Open" — `st.tabs` can't be switched programmatically;
  added an `st.toast` so the click gives feedback. **True auto-switch is deferred** (needs the
  `st.navigation` rework — see below).

## Deferred (own specs)
- **#2 proper** — replace `st.tabs` with `st.navigation`/page-state nav so "Open" buttons
  (face_grid, Gallery, cluster-summary) actually switch view. Ripples into the budapest e2e
  (which clicks `role=tab`). → its own spec.
- **#3 Image Analysis** — repurpose the Images tab into a per-image view: source photo with
  ALL face bboxes overlaid + per-face metrics + image metrics + filter status. → its own spec.

## Verification
disposition counts (107/79/154 = 340); nearest-pairs display formatted + clear; aspect-ratio
screenshot (`SHOT_face_analysis_aspect.png`); 11 unit tests + AppTest 0 exc + budapest D/E.
