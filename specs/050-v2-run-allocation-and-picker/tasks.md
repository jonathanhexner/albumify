# Tasks: spec-050 v2 App Run Allocation + Picker

**Status**: Implemented
**Estimated effort**: ~3 hours

7 phases. Each phase ends in a checkpoint that can ship independently if the rest slips.

---

## Phase 0 — Run-dir allocator

- [ ] T000 Create `face_cluster/run_layout.py`:
  ```python
  from __future__ import annotations
  from pathlib import Path
  from uuid import uuid4

  def allocate_run_dir(base_dir: Path, album_slug: str) -> tuple[Path, str]:
      base_dir.mkdir(parents=True, exist_ok=True)
      run_id = uuid4().hex
      run_dir = base_dir / run_id
      run_dir.mkdir(parents=True, exist_ok=False)  # collision = crash, not silent reuse
      return run_dir, run_id
  ```
- [ ] T001 Create `tests/face_cluster/test_run_layout.py`:
  - Allocator returns unique paths across 10 successive calls.
  - Returned dir exists on disk.
  - `run_id` matches the dir name and is 32 lowercase hex chars.
  - `album_slug` is accepted but not used in path (documentation test — guards against someone "improving" the layout by adding the album as a subdir, which would defeat the flat-uuid policy).

**Check**: 4/4 pass.

---

## Phase 1 — Run tab: album input + early session state

- [ ] T010 In `app/face_clustering_v2/tabs/run_tab.py`:
  - Replace the "Output directory" text input with two fields side-by-side: **Album name** (required) and **Base runs directory** (collapsed under an "Advanced" expander, defaulted to `~/.sim_bench/runs`).
  - Replace `out = ...` defaulting logic with a call to `allocate_run_dir(base_dir, album)` inside the `if st.button("Run", ...)` branch.
  - Disable the Run button if `not (src and album)`.
  - **Write `st.session_state.v2_last_run_dir = str(run_dir)` BEFORE calling `run_v2_pipeline`**, not after success.
  - Pass `album=album` and `run_dir=run_dir` to `run_v2_pipeline` (signature change in Phase 2).
- [ ] T011 Drop the `v2_out_dir` session-state key entirely; replace any remaining reads with `v2_last_run_dir` (it now serves the same purpose AND survives failures).

**Check**: Manual smoke — empty album → button disabled. Enter album + src → button enabled. Click Run → see fresh `runs/<uuid>/` materialize on disk before the spinner appears.

---

## Phase 2 — Pipeline: accept allocated run_dir + album

- [ ] T020 In `app/face_clustering_v2/pipeline.py::run_v2_pipeline`:
  - Add `album: str` and `run_dir: Path` parameters; remove the `output_dir` parameter (caller now allocates).
  - Use `run_id = run_dir.name` (the UUID) — delete the `datetime.utcnow().strftime(...)` line.
  - Pass `source_album=album` (not `src_dir.name`) into the `start_action` payload.
  - Verify `output_dir=str(run_dir)` flows into the payload too.
- [ ] T021 Update the 2 other callers of `run_v2_pipeline`:
  - `scripts/run_v2.py` — wire up an `--album` flag + the allocator, OR keep a legacy mode flag (decide based on what `run_v2_script` tests cover).
  - Any test that constructs the pipeline directly — update signature.
- [ ] T022 Add `tests/face_clustering/test_v2_pipeline_run_allocation.py` — runs a 3-jpg fixture via the new signature, asserts:
  - action_log row exists with `run_id == run_dir.name`, `source_album == "test_album"`, `producer == "fc_app_v2"`.
  - `run_dir / "face_clustering.db"` exists.

**Check**: New test passes; `scripts/run_v2.py` smoke OK.

---

## Phase 3 — Run picker component

- [ ] T030 Create `app/face_clustering_v2/components/run_picker.py`:
  - `RunPickerEntry` dataclass (per §Data contracts in spec).
  - `render_run_picker(*, label, key, limit) -> Optional[RunPickerEntry]`:
    - Query: `RunHistoryRepository().find(RunHistoryCriteria(producer="fc_app_v2", limit=limit))`.
    - Build options: `[f"{r.started_at[:16]} — {r.source_album} — {r.n_faces or '-'}f/{r.n_clusters or '-'}c — {r.status} ({r.run_id[:8]})" for r in rows]`.
    - Default-select the entry matching `st.session_state.v2_last_run_dir` if present; else the first (newest).
    - Return the chosen `RunPickerEntry` (None if the dropdown is empty).
- [ ] T031 Unit test `tests/face_clustering/test_v2_run_picker.py`:
  - Seed a temp action_log DB with 3 v2 rows + 1 non-v2 row.
  - Construct the Repository with the temp DB (via fixture).
  - Render the picker via `AppTest.from_function(...)` — assert 3 options visible, ordered newest-first.
  - Assert the non-v2 row is filtered out.

**Check**: 4 assertions in unit test pass.

---

## Phase 4 — Clusters tab rewrite

- [ ] T040 Rewrite `app/face_clustering_v2/tabs/clusters_tab.py`:
  ```python
  def render_clusters_tab() -> None:
      st.subheader("Clusters — v2 run")

      entry = render_run_picker(label="Pick a run from history")
      run_dir_default = entry.output_dir if entry else st.session_state.get("v2_last_run_dir", "")

      with st.expander("Advanced: override run directory", expanded=False):
          override = st.text_input("Run directory (full path)",
                                    value=str(run_dir_default), key="v2_clusters_dir")
      run_dir = Path(override or run_dir_default)
      if not run_dir or not (run_dir / "face_clustering.db").exists():
          st.info("Pick a run above, or paste a run directory in the Advanced section.")
          return

      st.caption(f"Loaded: `{run_dir}`  ·  album: **{entry.album if entry else '(unknown)'}**  ·  status: **{entry.status if entry else '?'}**")

      from face_cluster.run_store import RunStore
      try:
          store = RunStore(run_dir)
          result = store.clusters("latest")
          faces = store.faces()
      except Exception as e:
          st.error(f"Could not read run: {e}")
          return

      if not result.clusters:
          st.info("Run produced no clusters (all noise).")
          return

      faces_by_index = {i: f for i, f in enumerate(faces)}
      c1, c2, c3 = st.columns(3)
      c1.metric("Clusters", result.n_clusters)
      c2.metric("Faces", len(faces) - result.n_noise)
      c3.metric("Noise", result.n_noise)

      for cid, face_indices in sorted(result.clusters.items(), key=lambda kv: -len(kv[1])):
          with st.expander(f"Cluster {cid} — {len(face_indices)} face(s)"):
              cols = st.columns(4)
              for i, fidx in enumerate(face_indices):
                  face = faces_by_index.get(fidx)
                  with cols[i % 4]:
                      crop = store.crop_path(face.face_id) if face else None
                      if crop and crop.exists():
                          st.image(str(crop), width=120)
                      st.caption(f"face_id={face.face_id if face else '?'}")
  ```
- [ ] T041 Delete the unused `list_clusters` / `list_assignments` call sites elsewhere (grep first).

**Check**: Manual smoke against the user's existing v2_latest run (or a fresh new-layout run).

---

## Phase 5 — AppTest E2E

- [ ] T050 Create `tests/face_clustering/test_v2_run_picker_e2e.py`:
  - Use `streamlit.testing.v1.AppTest.from_file("app/face_clustering_v2/main.py")`.
  - Monkeypatch `run_v2_pipeline` to a fast stub that writes a minimal `face_clustering.db` with 1 cluster (2 faces) into the allocated run_dir.
  - Drive the Run tab: type album, type src dir, click Run.
  - Switch to Clusters tab.
  - Assert: picker has 1 option; first cluster expander renders; `st.error` was NOT called.
  - Drive a second Run with a different album; switch back to Clusters; assert the picker now has 2 options and is sorted newest-first.

**Check**: AppTest passes.

---

## Phase 6 — Sweep + sighting close

- [ ] T060 Run `pytest tests/face_clustering tests/architecture` — confirm no new failures introduced. (The 13 pre-existing ones remain — out of scope per spec.)
- [ ] T061 Manual restart of the v2 app:
  - Run with album="Budapest" → confirm `runs/<uuid>/` materializes, action_log row visible in History tab.
  - Switch to Clusters → confirm picker shows the new run, expanders render, thumbnails visible.
  - Run a second time with album="Other" → confirm both runs in picker; switching between them works.
  - Paste an old `v2_latest`-style path into the Advanced override → confirm it still works.
- [ ] T062 File and immediately resolve **SIGHTING-075** covering all 3 symptoms as a single MVP-completeness gap. Cross-link to spec-050.
- [ ] T063 `CHANGES_LOG.md` entry under `[BUGFIX]`.
- [ ] T064 Run `/code-review` → `REVIEW.md`. Flip spec status → Implemented.

**Check**: All AC1–AC9 green; manual smoke clean.

---

## Sequencing rationale

- Phase 0 before 1: allocator is the only piece run_tab needs.
- Phase 1 before 2: UI proves the field shape before pipeline wires it through.
- Phase 2 before 3: picker needs real per-run action_log rows to test against; Phase 2 produces them.
- Phase 3 before 4: clusters_tab depends on the picker component.
- Phase 5 last among code: AppTest needs every other layer working.
- Phase 6 closes out: full sweep + sighting + review gate. Mandatory.

---

## Test delta

| Phase | New tests | Type |
|---|---|---|
| 0 | 4 | unit (allocator) |
| 2 | 1 | integration (pipeline + action_log) |
| 3 | 4 | unit (picker) |
| 5 | 2 | AppTest E2E |
| **Total** | **11** | |

No tests deleted. Coverage gap that allowed the 3 reported bugs to ship is closed by Phase 5 specifically.
