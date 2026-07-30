# Tasks: Click every button — Playwright E2E gate (062)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

---

## Phase 0 — Playwright + Streamlit subprocess plumbing (~2 h)

- [ ] **T001** Add a session-scoped fixture `v2_streamlit_server` in `tests/face_clustering/conftest.py`:
  - Subprocess: `streamlit run app/face_clustering_v2/main.py --server.port 8889 --server.headless true`
  - Wait until `http://localhost:8889` returns 200 (timeout 20 s)
  - Yields `"http://localhost:8889"`
  - Teardown: `proc.terminate(); proc.wait(5)`
- [ ] **T002** Add `playwright_page` function-scoped fixture: launches headless Chromium, returns a fresh page per test; saves screenshot to `tests/face_clustering/_failure_artifacts/<test_name>.png` on failure (via pytest hook).
- [ ] **T003** Add `seeded_v2_run` session fixture: reuses `_build_synthetic_run_dir` + `_add_no_op_merge_round` + seeds an action_log row pointing at that dir. Returns `(run_dir, action_id)`.
- [ ] **T004** Add `@pytest.mark.browser` marker registration in `pyproject.toml` (`markers = ["browser: ..."]`).

**Validation gate (Phase 0)**: a placeholder test that opens the page, asserts the title "Face Clustering — v2" is visible, then exits. Passes cleanly; subprocess cleans up.

---

## Phase 1 — Run tab (~1 h)

- [ ] **T010** `test_run_tab_inputs_enable_button_when_both_filled`: type source dir + album → "Run pipeline" enables.
- [ ] **T011** `test_run_tab_empty_source_keeps_button_disabled`: leave source empty → button stays disabled.
- [ ] **T012** `test_run_tab_profile_selectbox_persists_selection`: pick a non-default profile → next rerun shows it still selected.

**Validation gate (Phase 1)**: 3 Run-tab tests pass against the seeded fixture.

---

## Phase 2 — Cluster Analysis tab — read paths (~2 h)

- [ ] **T020** `test_cluster_analysis_empty_state_when_no_run_loaded`: clear session_state → info banner appears.
- [ ] **T021** `test_cluster_analysis_picker_populated_when_run_loaded`: seed run → selectbox shows ≥1 option.
- [ ] **T022** `test_cluster_analysis_metrics_strip_renders`: 5 `<div class="stMetric">` elements visible after picking a cluster.
- [ ] **T023** `test_cluster_analysis_face_grid_renders_images`: at least one `<img>` element with src containing `/crops/` is visible (this is the test that would have caught today's thumbnail bug).
- [ ] **T024** `test_cluster_analysis_face_grid_renders_captions_even_when_crops_missing`: delete crop files mid-test → page does NOT crash; captions still visible.
- [ ] **T025** `test_cluster_analysis_changing_cluster_re_renders_grid`: switch picker → grid contents change.
- [ ] **T026** `test_cluster_analysis_nearest_clusters_expander_renders`: click expander → ≥1 row visible.
- [ ] **T027** `test_cluster_analysis_go_to_button_switches_cluster`: click "Go to" on a nearest cluster → picker selection updates.

**Validation gate (Phase 2)**: 8 Cluster Analysis read-path tests pass.

---

## Phase 3 — Cluster Analysis tab — force merge + graph debug (~2 h)

- [ ] **T030** `test_force_merge_expander_renders_dropdowns`: click expander → two cluster selectboxes + "Preview Merge" button visible.
- [ ] **T031** `test_force_merge_preview_shows_three_gate_badges`: pick A + B, click Preview → 3 `<div class="stMetric">` (Exemplar / Support / Diameter) with PASS or FAIL visible.
- [ ] **T032** `test_force_merge_confirm_writes_snapshot_and_switches_run`: click Confirm → on-disk snapshot dir exists; session_state's `current_run_dir` updated.
- [ ] **T033** `test_graph_debug_expander_renders_metrics`: click "Graph debug" → 4 metric widgets visible.
- [ ] **T034** `test_graph_debug_heatmap_renders_when_cluster_has_embeddings`: cluster with ≥2 faces → Plotly heatmap iframe visible.

**Validation gate (Phase 3)**: 5 force-merge + graph-debug tests pass.

---

## Phase 4 — History tab (~2 h)

- [ ] **T040** `test_history_runs_table_renders_seeded_run`: seeded v2 run visible as a row.
- [ ] **T041** `test_history_clicking_run_row_shows_detail_panel`: click row → detail panel + "Load into analysis tabs" button visible.
- [ ] **T042** `test_history_load_button_enabled_for_v2_run`: seeded v2 run → button is enabled (NOT the disabled "missing artifacts" state — would have caught SIGHTING-080).
- [ ] **T043** `test_history_load_button_enabled_for_v4_transitional_run`: seed a run with `_v4/face_clustering.db` → button enabled.
- [ ] **T044** `test_history_load_button_disabled_for_incomplete_run`: seed a run with status="failed" → button disabled with clear warning.
- [ ] **T045** `test_history_load_button_click_populates_session_state`: click → session_state has `current_run_dir`; success toast visible.
- [ ] **T046** `test_history_album_filter_filters_table`: type an album name → table shows only matching rows.
- [ ] **T047** `test_history_comment_input_persists_to_db`: type a comment → reload page → comment still there.

**Validation gate (Phase 4)**: 8 History-tab tests pass.

---

## Phase 5 — Drift guard (~1 h)

- [ ] **T050** Create `tests/architecture/test_v2_click_coverage.py`. Greps every `app/face_clustering_v2/{tabs,components}/*.py` for `st.button(`, `st.selectbox(`, `st.dataframe(`, `st.expander(`, etc. Builds a set of expected interactive elements. Greps `tests/face_clustering/test_v2_click_every_button.py` for test names; asserts every interactive element has at least one test naming it. **Fails CI when a new button lands without a test.**

**Validation gate (Phase 5)**: drift guard passes; a deliberate "remove a test" experiment makes it fail.

---

## Phase 6 — Wire into the gate (~1 h)

- [ ] **T060** Update CLAUDE.md §"Delivery Quality": name `pytest -m browser tests/face_clustering/test_v2_click_every_button.py` as the pre-handoff check for any v2 UI change.
- [ ] **T061** Update spec-implementer agent prompt (or document the manual step) to run this test before flipping any "considerable change" spec to Implemented.
- [ ] **T062** Cross-reference from spec-060 (E2E gold-standard) and spec-061 (read-path audit).

**Validation gate (Phase 6)**: CLAUDE.md updated; agent prompt diff present.

---

## Phase 7 — Close-out (~1 h)

- [ ] **T070** CHANGES_LOG entry summarizing: ≥24 element tests + 1 drift guard.
- [ ] **T071** `/code-review` → REVIEW.md.
- [ ] **T072** Flip status Draft → Code Review → Implemented.
- [ ] **T073** Commit + push.

**Final gate**: spec-062 status = Implemented; running `pytest -m browser` against the v2 app produces 24+ green tests in ≤ 3 min; the drift guard guarantees future buttons get covered.

---

## Total estimate

**~8–12 hours.** Phase 0 plumbing dominates the first few hours; the rest is mostly templated tests once one passes cleanly.
