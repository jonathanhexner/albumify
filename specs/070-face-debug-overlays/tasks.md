# spec-070 — Tasks

**Spec**: [spec.md](spec.md) · **Status**: Code Review (core done; render step deferred)
**Predecessor**: SIGHTING-093 G3 (pose data) — RESOLVED 2026-06-05.

## Done

- [x] T01 — `face_cluster/overlays.py`: `pose_axes_2d(center, scale, yaw,
      pitch, roll)` + `draw_overlay(img, bbox, landmarks, pose)`. Streamlit-free,
      cv2 for drawing. Shared by UI + (future) render step. (AC1, AC2)
- [x] T02 — `tests/face_clustering/test_overlays.py`: 6 tests — frontal axes,
      yaw→Z sideways, pitch→Z vertical, anchor, draw changes pixels, None-pose
      safe. **All green.** (AC1, AC2, AC6)
- [x] T03 — Live UI overlay: `face_bbox_overlay.py` draws the 3 pose axes
      (X red / Y green / Z blue) anchored at the bbox centre, y-flipped for
      Plotly; `face_analysis_tab` passes `pose=record.pose`. (AC4)
- [x] T04 — Face Metrics drill-in: `st.dataframe(on_select="rerun",
      selection_mode="single-row")` → seeds `selected_face_id` + hint to open
      Face Analysis. (AC5)
- [x] T05 — Verification: AppTest vs `v2_budapest_20260605b` (340 faces,
      with_pose=340) → 0 exceptions; **visual check** rendered a near-profile
      face (yaw=-88) → blue forward-axis points the correct way
      (`_overlay_sample.png`). (AC4, AC5, AC6)

## Deferred (low value vs cost)

- [ ] D01 — Opt-in `render_face_overlays` pipeline step (save annotated JPGs
      to `run_dir/overlays/`). DEFERRED: the per-run output dir isn't cleanly
      available at step time (export derives `results/album/timestamp`, not the
      uuid run dir), and the **live UI overlay already provides the same
      "click → large image + bbox + pose fit"** interactively. Revisit if a
      no-app batch-QA artifact is needed; `draw_overlay` is ready for it.

## Close-out

- [ ] T06 — CHANGES_LOG (added); docs/architecture HTML note for the overlay
      component + helper; `/code-review` → REVIEW.md; flip Status →
      Implemented after no High findings + your visual sign-off in the app.
