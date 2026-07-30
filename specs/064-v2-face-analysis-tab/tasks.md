# Tasks — Face Analysis tab (064)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

## Phase 1 — Service (~30 min)
- [ ] T001 Add `FaceAnalysisService` class to `face_cluster/views/face_view.py` with `compute_face_detail(face_id) -> FaceView` (sync wrapper over existing `FaceView.compute`). Constructor takes `repo: ClusterAnalysisRepository`.
- [ ] T002 NEW `tests/face_clustering/views/test_face_view_service_synthetic.py` — 6 cases.

## Phase 2 — Tab + cross-tab nav (~1.5 h)
- [ ] T010 NEW `app/face_clustering_v2/tabs/face_analysis_tab.py` (≤ 80 LOC). Layout: face-id input + main panel (crop + bbox overlay) + scores strip + nearest-faces grid + gate verdicts table.
- [ ] T011 NEW component `app/face_clustering_v2/components/face_bbox_overlay.py` — Plotly figure with source image + bbox + landmarks dots.
- [ ] T012 UPDATE `face_grid.py` to add "Open" button per thumbnail; writes `selected_face_id` to session_state.
- [ ] T013 Wire into `app/face_clustering_v2/main.py`.

## Phase 3 — Baseline e2e Scenario D (~30 min)
- [ ] T020 Add `test_v2_scenario_d_face_analysis_drill_down` — clicks face_grid's Open button, asserts Face Analysis tab renders.

## Phase 4 — Close-out (~30 min)
- [ ] T030 Run `pytest -m budapest` → all 4 scenarios green.
- [ ] T031 `/code-review` → REVIEW.md. CHANGES_LOG. Status → Implemented. Commit + push.

**Total: ~3-4 h.**
