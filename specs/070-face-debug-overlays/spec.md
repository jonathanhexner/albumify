# spec-070 — Face debug overlays (bbox + landmarks + pose axes)

**Created**: 2026-06-05
**Status**: Code Review (core implemented + verified 2026-06-05; opt-in render step deferred — see tasks.md)
**Priority**: P2 (debug tooling — high leverage going forward)
**Predecessors**: **SIGHTING-093** (pose data must be persisted first), spec-069
  (Face Metrics tab — the drill-in source), spec-068 (telemetry)
**Sighting**: SIGHTING-093 (pose persistence)

---

## Goal

Make it trivial to *see* what the detector/pose model decided for any face:
the bounding box, the 5 landmarks, and a **pose-fit gizmo** (3 axes showing
head yaw/pitch/roll) drawn on the face. Useful for debugging detection / pose
/ clustering going forward.

## Decision: coordinates are the source of truth; rendered photo is a derived, opt-in artifact

| Layer | What | Storage |
|---|---|---|
| **Coordinates (primary)** | bbox + landmarks (already in `faces`) + pose yaw/pitch/roll (added by SIGHTING-093). UI draws bbox + landmarks + pose axes **live** (Plotly). | none new |
| **Rendered photo (opt-in)** | a thin pipeline step draws the same overlay onto the source image and saves `run_dir/overlays/face_<id>.jpg`. For eyeballing a run without the app + for `verify_run.py`. | ~few MB / run, behind a flag |

Coordinates stay authoritative so the UI is interactive; the photo is a
convenience, never the only copy.

## What we build

### Shared helper (one source of the pose-axis math)
`face_cluster/overlays.py` (Streamlit-free, cv2/numpy):
- `pose_axes_2d(center, scale, yaw, pitch, roll) -> dict[str, (x0,y0,x1,y1)]`
  — rotate 3 unit axes by the head rotation, project to 2D, return the 3 line
  endpoints (X=red, Y=green, Z=blue). Pure math, deterministic, unit-testable.
- `draw_overlay(img, bbox, landmarks, pose) -> img` — draw rect + dots + axes
  with cv2 (used by the render step).

### Pipeline render step (opt-in)
`render_face_overlays` step: config `save_overlays: bool = False`. When on,
for each face writes `run_dir/overlays/face_<id:04d>.jpg` (source crop or
full image + overlay). Runs after detection + pose. ~thin step, calls the
helper.

### UI
- `face_bbox_overlay.py`: add the pose-axis trace (live, from yaw/pitch/roll
  via `pose_axes_2d`). Falls back gracefully when pose is None.
- Face Metrics table (spec-069) → **drill-in**: selecting a row sets
  `selected_face_id`; the Face Analysis tab shows the large image + overlay.

## Acceptance criteria

| # | Criterion | Verified by |
|---|-----------|-------------|
| AC1 | `pose_axes_2d` returns 3 axis lines; rotating yaw/pitch/roll moves them correctly (e.g. yaw=0 → X axis horizontal) | unit test |
| AC2 | `draw_overlay` produces an image with bbox + landmarks + axes (non-empty diff vs input) | unit test |
| AC3 | `render_face_overlays` writes `overlays/face_*.jpg` only when `save_overlays=True`; no-op otherwise | unit test |
| AC4 | Face Analysis overlay draws pose axes when pose present; no error when None | AppTest |
| AC5 | Face Metrics row-select drives the Face Analysis large view | AppTest |
| AC6 | Pose axes use the SIGHTING-093 yaw/pitch/roll (correct InsightFace `[pitch,yaw,roll]` remap) | unit test on a known pose |

## Out of scope

- 3D head mesh / full model fit (just 3 axes).
- Re-detecting or re-posing — reads persisted coordinates only.

## Effort estimate

~half day: helper + pose-axis math + render step + overlay trace + drill-in +
unit tests. Depends on SIGHTING-093 landing first (pose data).
