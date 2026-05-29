# spec-064 — v2 Face Analysis tab (per-face popup)

**Created**: 2026-05-30
**Status**: Draft
**Priority**: P2
**Predecessors**: spec-042, spec-045 (Cluster Analysis — face_grid feeds the click target)
**Reference baseline**: must keep `test_v2_e2e_budapest_baseline.py` green; adds Scenario D.

---

## Problem

In legacy FC App, clicking a face thumbnail in Cluster Analysis opens a per-face popup with: bbox + landmarks visualized over the source image, all quality scores (blur, pose, area), nearest-clusters-and-faces for THIS face, and gate verdicts (why was it accepted / rejected). The v2 app has nowhere to send the user when they click a thumbnail.

The compute already exists: `face_cluster/views/face_view.py` has `FaceView.compute(result, face_id) -> FaceView` which produces everything above.

## What we build

### Backend

**Service: `FaceAnalysisService`** in `face_cluster/views/face_view.py` (extend existing module):
- `compute_face_detail(face_id) -> FaceView` — sync wrapper over `FaceView.compute`
- Reuses `ClusterAnalysisRepository` (already constructed for the current run by the Cluster Analysis tab — share via session_state cache)

### Frontend

**`app/face_clustering_v2/tabs/face_analysis_tab.py`** — orchestrator (≤ 80 LOC):
1. Face-id input (default from `st.session_state['selected_face_id']` set by Cluster Analysis's "Open Face Analysis" button on each thumbnail)
2. Main panel: large face crop + bbox/landmarks overlay (Plotly)
3. Quality scores strip (5 metrics: blur, pose yaw/pitch/roll, area)
4. Nearest-faces grid (8 cols, same component as Cluster Analysis's face_grid)
5. Gate verdicts table (which gates fired, which passed, which rejected)

**Cluster Analysis update**: face_grid's caption now includes a button "Open" that writes `selected_face_id` to session_state + auto-switches to the Face Analysis tab.

### Tests

| Layer | File | Cases |
|---|---|---|
| Service synthetic | `tests/face_clustering/views/test_face_view_service_synthetic.py` (NEW) | 6: compute_face_detail returns FaceView, unknown face_id raises, gate verdicts populate, nearest-faces non-empty, bbox/landmarks present, score fields populated |
| Service real | + 1 opt-in `slow` case on Budapest reference run |
| Architecture | tab in the LOC + no-DB-FS-cfg.get arch scan |
| Baseline gate | Add **Scenario D** to `test_v2_e2e_budapest_baseline.py`: load reference run → Cluster Analysis → click first face thumbnail's Open button → assert Face Analysis tab shows a face crop + ≥ 5 metric widgets |

## Locked decisions

1. **Reuses existing compute.** `FaceView.compute` already does the work. No new clustering math.
2. **Cross-tab navigation via session_state.** Cluster Analysis writes `selected_face_id`; Face Analysis reads it. No URL routing (Streamlit's tab model doesn't support deep-links cleanly).
3. **Sync, not async.** Per spec-079 lesson.
4. **Bbox/landmarks via Plotly.** Already imported by spec-045's cluster_debug; reuse.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `FaceAnalysisService.compute_face_detail(face_id) -> FaceView` exists; sync; typed | grep + test |
| AC2 | `face_analysis_tab.py` ≤ 80 LOC; no SQL / FS / `cfg.get` | LOC + arch test |
| AC3 | 6 synthetic service tests pass | pytest |
| AC4 | 1 real-fixture smoke passes on Budapest reference run | pytest -m slow |
| AC5 | Baseline e2e Scenario D green | pytest -m budapest |
| AC6 | "Open" button on face_grid thumbnail correctly populates session_state | covered by Scenario D |

## Effort estimate

**~3-4 hours**: Service ~30 min (most logic exists), tab + components ~1.5 h, tests ~1 h, baseline scenario ~30 min.
