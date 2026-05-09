# Feature Specification: Face Detail Popup

**Spec**: `015-face-detail-popup`
**Created**: 2026-04-24
**Status**: Draft
**Resolves**: SIGHTING-026 (quality gate opaque — poor faces pass unnoticed); Feature Request 2026-04-24

---

## Problem Statement

The Face Analysis tab is the only place in the app where per-face detail is visible, but it requires the user to leave their current tab, find the face in a selectbox, wait for the async worker to compute, and then return. When inspecting faces inside a cluster gallery, merge pair, or co-image group, the round-trip is disruptive and slow.

Additionally there is no way to annotate a face — e.g., mark it as "falsely filtered", "bad detection", or "confirmed correct" — without leaving the data somewhere external.

## Objective

Any face crop shown anywhere in the app (cluster gallery, Merge Analysis pairs, Face Analysis neighbor grids, co-image strip) can be clicked to open a modal dialog showing **the full contents of the current Face Analysis tab**, plus a free-text comment field that is persisted to disk.

The Face Analysis tab itself is **not removed** — it remains as the primary full-width view. The popup is a quick-access overlay for in-context inspection.

---

## User Stories

### Story 1 — Popup from cluster gallery (P0)
**Given** I am viewing Clusters (Base) or Clusters (Merged) and I see a face crop in a cluster row,
**When** I click the "Detail" button under that face,
**Then** a modal popup opens showing the full face detail without navigating away from the Clusters tab.

### Story 2 — Popup from Merge Analysis (P0)
**Given** I am reviewing a merge pair in Merge Analysis and I want to know more about a specific face,
**When** I click "Detail" under that face crop,
**Then** the same popup opens with all face attributes and neighbors.

### Story 3 — Popup from co-image strip and neighbor grids (P1)
**Given** I am in any tab that renders a face crop — including the closest-same-cluster strip, closest-other-clusters strip, or co-image strip in Face Analysis itself —
**When** I click "Detail" on any face,
**Then** the popup opens for that face.

### Story 4 — Comment annotation (P1)
**Given** I am viewing a face in the popup,
**When** I type a comment and click Save,
**Then** the comment is persisted to `{output_dir}/face_comments.json` and is visible the next time the popup is opened for that face (within the same run or any later load of that run).

### Story 5 — Navigate to Face Analysis from popup (P2)
**Given** I want deeper analysis than the popup shows,
**When** I click "Open in Face Analysis tab" inside the popup,
**Then** the popup closes, the app navigates to the Face Analysis tab, and the face is pre-selected.

---

## Content — What the Popup Shows

The popup renders **identical content to the current Face Analysis tab**, in a condensed layout:

| Section | Content |
|---------|---------|
| **Header** | `face_XXXX` label, CORE/HOLDOUT badge (green/red) |
| **Rejection reason** | If holdout: the gate that failed (e.g., "blur: 32.4 < 50.0") |
| **Attributes** | Blur, Area, Pose (yaw/pitch/roll), det_score, Cluster ID, source image + rank |
| **Quality Report** | Per-gate table: gate name, value, threshold, PASS/FAIL badge; det_score; d10_score |
| **Closest — Same Cluster** | Up to 5 face crops with distances (each with its own Detail button) |
| **Closest — Other Clusters** | Up to 5 face crops with cluster label + distance (each with Detail button) |
| **Co-image faces** | All other faces from the same source image with cluster labels |
| **Comment** | Editable text area + Save button |
| **Navigation** | "Open in Face Analysis" button |

No information is added or removed relative to the existing Face Analysis tab.

---

## Data Contract: face_comments.json

Writer: the popup comment Save action.
Reader: the popup comment loader (on open).

```json
{
  "42": "falsely filtered — sharp frontal face, blur threshold too strict",
  "107": "bad detection — partial face on image edge"
}
```

Schema: `{ face_id_str: comment_str }` — keys are string representations of face_id integers.
Location: `{output_dir}/face_comments.json`
Max comment length: 1024 characters.
Missing file → treated as empty dict (no error).

---

## Architecture

### Trigger mechanism

Any tab that renders a face crop adds a small `Detail` button immediately below it. When clicked:
```python
st.session_state.face_popup_id = face_id
st.rerun()
```

### Dialog rendering

`main.py` calls `maybe_show_face_popup(result)` unconditionally at the bottom of every render cycle. This function checks `st.session_state.face_popup_id`:
- If set: opens `st.dialog("Face Detail")` and renders the popup content
- The dialog's close action clears `face_popup_id` from session_state

### Shared helper

New file: `app/face_clustering/face_popup.py`
- `maybe_show_face_popup(result: PipelineResult) -> None` — called from `main.py`
- `_face_detail_btn(face_id: int, key: str) -> None` — renders the small Detail button; any tab imports this instead of duplicating the trigger logic
- `_load_comments(output_dir: Path) -> dict[str, str]`
- `_save_comment(output_dir: Path, face_id: int, text: str) -> None`

### Computation

`FaceView.compute()` is called synchronously inside the dialog, wrapped in `st.spinner("Analysing...")`. The result is cached in `st.session_state.face_popup_cache[face_id]` so repeated opens of the same face are instant. Cache is cleared when a new run is loaded (inside `_invalidate_run_caches()`).

### No new backend modules

`FaceView.compute()` already exists and computes everything needed. No changes to `face_cluster/` are required for this feature.

---

## Edge Cases

| Case | Handling |
|------|---------|
| Face not in result (e.g., stale popup_id) | Show error in dialog: "Face {id} not found in current run" |
| Embeddings absent (legacy run) | Closest-same / closest-other sections show "Embeddings not available" (matches current tab behavior) |
| Holdout face — no cluster | Cluster field shows "holdout/noise"; closest-same is empty with info message |
| Comment > 1024 chars | Save button shows `st.error` inline; does not write |
| face_comments.json write fails | Show `st.warning`; non-fatal |
| Popup opens another popup (Detail inside popup) | Same `face_popup_id` mechanism; dialog re-renders for new face |

---

## Acceptance Criteria

1. Clicking "Detail" under any face in Clusters (Base) opens a popup showing the face's blur, area, pose, gate result, rejection reason, closest-same-cluster faces, closest-other-cluster faces, and co-image faces — without leaving the Clusters tab.
2. Clicking "Detail" under any face in Merge Analysis opens the same popup for that face.
3. A comment typed and saved in the popup persists in `face_comments.json` and reappears when the popup is reopened for that face (within the same Streamlit session and on reload of the run).
4. "Open in Face Analysis" navigates to the Face Analysis tab with the face pre-selected.
5. The Face Analysis tab is unchanged — the popup is additive, not a replacement.
6. No `face_cluster/` library code is modified (popup is UI-only).
7. The Detail button appears consistently: cluster gallery (Base and Merged), Merge Analysis crop strips, Face Analysis neighbor and co-image strips.
8. All output uses ASCII characters (no Unicode badges that break Windows console encoding).

---

## Out of Scope

- Replacing the Face Analysis tab
- Bulk face annotation / export of comments
- Comment history / versioning
- Face comparison between two faces side-by-side (separate feature)
