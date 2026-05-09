# Implementation Tasks: Face Detail Popup (spec-015)

**Design summary**:
- New `app/face_clustering/face_popup.py` — dialog renderer + comment store
- `main.py` calls `maybe_show_face_popup(result)` unconditionally on every render
- Every face-rendering site gets a `_face_detail_btn(face_id, key)` helper call below the crop
- No changes to `face_cluster/` library; no new pipeline stages

**Pre-condition**: Verify Streamlit version supports `st.dialog` (≥1.28).

---

## Task Checklist

### Phase 1 — Core popup infrastructure

- [ ] **T1**: Check Streamlit version in `pyproject.toml` / venv; confirm `st.dialog` is available. If not, identify upgrade path.

- [ ] **T2**: Create `app/face_clustering/face_popup.py` with:
  - `_load_comments(output_dir: Path) -> dict[str, str]` — reads `face_comments.json`, returns `{}` if missing
  - `_save_comment(output_dir: Path, face_id: int, text: str) -> None` — writes to `face_comments.json`; enforces 1024-char limit; shows `st.error` inline if exceeded
  - `_face_detail_btn(face_id: int, key: str) -> None` — renders a small "Detail" button; sets `st.session_state.face_popup_id = face_id` on click
  - `maybe_show_face_popup(result: PipelineResult) -> None` — checks `st.session_state.face_popup_id`; if set, opens the dialog

- [ ] **T3**: Implement the dialog body inside `maybe_show_face_popup`:
  - Header: `face_XXXX`, gate badge (CORE/HOLDOUT), rejection reason if holdout
  - Attributes row: blur, area, pose (yaw/pitch/roll), det_score (if present)
  - Cluster assignment
  - Source image + rank in image
  - Quality Report section (reuse `_render_quality_report` from `quality_panels.py`)
  - Closest — Same Cluster: up to 5 crops with distance captions; each has its own `_face_detail_btn`
  - Closest — Other Clusters: up to 5 crops with cluster label + distance; each has its own `_face_detail_btn`
  - Co-image faces strip: all faces from same source image; each has its own `_face_detail_btn`
  - Comment section: `st.text_area`, Save button calling `_save_comment`
  - "Open in Face Analysis" button: sets `selected_face`, clears `face_popup_id`, triggers navigation

- [ ] **T4**: Add `FaceView` result cache in session_state:
  - Key: `(face_id, str(output_dir))`
  - Stored in `st.session_state.face_popup_cache` (dict)
  - Populated synchronously with `st.spinner("Analysing face_XXXX...")` on first open
  - Cache cleared in `_invalidate_run_caches()` in `state.py`

- [ ] **T5**: Wire `maybe_show_face_popup(result)` into `main.py` — call unconditionally after tab rendering, so the dialog is available regardless of which tab is active.

- [ ] **T6**: Add `face_popup_id` and `face_popup_cache` to the session_state initialisation block in `state.py`.

---

### Phase 2 — Surface Detail buttons at all face-rendering sites

- [ ] **T7**: `gallery_panels.py` → `_render_cluster_view()` — add `_face_detail_btn` below each face in the "All Faces" grid and the Exemplars grid.

- [ ] **T8**: `gallery_panels.py` → `_render_cluster_gallery()` — add `_face_detail_btn` below each thumbnail in the cluster summary row (3 exemplar thumbnails per row).

- [ ] **T9**: `tabs/cluster_analysis_tab.py` — add `_face_detail_btn` below faces in:
  - Exemplars grid (top of page)
  - All Faces in Cluster grid
  - Nearest Clusters exemplar thumbnails

- [ ] **T10**: `tabs/face_analysis_tab.py` — add `_face_detail_btn` below faces in:
  - Closest — Same Cluster strip
  - Closest — Other Clusters strip
  - Other Faces in same image strip

- [ ] **T11**: `_merge_decisions_panel.py` — add `_face_detail_btn` below face crops in merge pair crop strips (exemplar faces for cluster A and cluster B).

---

### Phase 3 — Navigation and state

- [ ] **T12**: "Open in Face Analysis" inside the popup:
  - Set `st.session_state.selected_face = face_id`
  - Set `st.session_state.face_worker = None` (clears stale worker)
  - Clear `st.session_state.face_popup_id`
  - Navigate to Face Analysis tab — set the active tab index in session_state (verify how tab selection is done in `main.py`)

- [ ] **T13**: Popup close (X button / clicking outside) must clear `face_popup_id` — verify `st.dialog` handles this automatically via its built-in close mechanism. If not, add cleanup on dialog exit.

---

### Phase 4 — Testing

- [ ] **T14**: Unit test for `_load_comments` / `_save_comment`:
  - Write comment → read back → matches
  - Comment > 1024 chars → raises / shows error without writing
  - Missing file → returns `{}`
  - Multiple face_ids → all persisted and readable

- [ ] **T15**: Unit test for `_face_detail_btn` key uniqueness — verify the key scheme is collision-free for the same face appearing in multiple tabs simultaneously (e.g., `f"popup_btn_{face_id}_{key_suffix}"`).

- [ ] **T16**: Verify `_render_quality_report` is importable from `face_popup.py` (same sys.path context). Check if the import path works from the Streamlit app root.

- [ ] **T17**: Manual smoke test on Windows — run app, open cluster gallery, click Detail on a face, verify popup appears with correct data, save a comment, close, reopen, verify comment persists.

---

### Phase 5 — Cleanup and docs

- [ ] **T18**: Verify no Unicode characters were introduced in popup output (ASCII-only rule from CLAUDE.md).

- [ ] **T19**: Update `docs/FEATURE_REQUESTS.md` — mark the face popup request as IMPLEMENTED.

- [ ] **T20**: Update `CHANGES_LOG.md` with [FEATURE] entry.

- [ ] **T21**: Update `docs/architecture.md` if the popup introduces any new session_state keys or file contracts not previously documented.
