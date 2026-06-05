# spec-067 — Tasks

**Spec**: [spec.md](spec.md) · **Status**: In Progress · **Started**: 2026-06-03

Drives SIGHTING-091 to resolution. Unblocks budapest e2e Scenarios B/C/D/E/F
(all time out on the canvas-rendered `st.dataframe` row-pick). Approach:
seed `current_run_dir` via query param so the harness skips the one
untestable click; every downstream tab already resolves its data from that
session key, so no tab code changes.

## Confirmed before drafting (grep evidence)

- All four analysis tabs (`cluster_analysis_tab`, `face_analysis_tab`,
  `merged_clusters_tab`, `quality_tab`) resolve via
  `for key in ("current_run_dir", "v2_last_run_dir", "active_run_dir")` →
  build service **from the dir**. None read a pre-loaded `pipeline_result`.
  ⇒ seeding `current_run_dir` is sufficient; no `load_pipeline_result` call
  needed in the seeder.
- `load_button.py` Load click sets `current_run_dir` + `active_run_dir`
  (+ `pipeline_result`, `current_source_album` which nothing downstream reads).
  Seeder mirrors the two dir keys.

## Tasks

- [ ] T01 — Seeder in `app/face_clustering_v2/main.py`.
      `_seed_session_state_from_query_params()`: once per session (sentinel
      `_qp_seeded`), map `?current_run_dir=<path>` → session `current_run_dir`
      **and** `active_run_dir` (mirror the Load button). Allowlist also
      `?selected_face_id=<int>` (guarded int-parse). Call it **before**
      `st.tabs(...)`. Opt-in: no params ⇒ no writes. (AC1, AC6) | ~15 LOC

- [ ] T02 — Arch/unit test for the seeder (AppTest).
      `tests/face_clustering/test_v2_query_param_seed.py`:
      (a) `?current_run_dir=X` ⇒ `session_state["current_run_dir"] == X`;
      (b) no params ⇒ key absent (no behaviour change);
      (c) re-run idempotent (sentinel set). (AC1, AC6)

- [ ] T03 — Fixture `page_with_reference_run_loaded` in
      `tests/face_clustering/e2e_budapest/conftest.py`. Skip if
      `REFERENCE_RUN_DIR` missing. `page.goto(APP_URL + "?current_run_dir=" +
      urllib.parse.quote(str(REFERENCE_RUN_DIR)))` (URL-encode the Windows
      path: backslashes + `C:`). Wait for `h1`. Return page. (AC2)

- [ ] T04 — Convert Scenarios C/D/E/F to the fixture: swap `(page)` →
      `(page_with_reference_run_loaded)`, delete the 4-line History-tab +
      row-pick + Load preamble, keep every downstream assertion. (AC3)

- [ ] T05 — Convert Scenario B. Its old job (Load button enabled + detail
      panel non-blank for a v2 run, SIGHTING-080/089) cannot survive a
      row-pick bypass. Repoint B at an analysis tab (Cluster Analysis) and
      assert the seeded run renders the baseline cluster shape. Add a
      module docstring note that the Load-flow regression now lives in
      `test_v2_app_smoke.py::test_history_tab_recognizes_v2_run_as_loadable`
      + `HistoryService` unit tests. (AC3, AC4)

- [ ] T06 — Run `pytest -m budapest tests/face_clustering/e2e_budapest/ -v`.
      All 6 scenarios green. Capture output for REVIEW.md. (AC4)
      **PARTIAL**: B, D green. E/F/C blocked — see T09–T12 (added 2026-06-05).

- [ ] T07 — Confirm AC5: `git diff --stat` shows `run_table.py` and
      `actions_table.py` untouched. History tab UI unchanged.

## Follow-up tasks (added 2026-06-05, from "Test coverage strategy" section)

Unblocking the seed exposed never-validated assertions in C–F. See spec.md
§"Test coverage strategy per tab" + §"Resolution plan".

- [x] T09 — **Server fragility (root blocker).** Shared session-scoped
      Streamlit server served only the first browser test; next `page.goto`
      timed out. Made `streamlit_server` function-scoped + port-free guard in
      `e2e_budapest/conftest.py`. **DONE** — B and D pass.

- [x] T10 — **D selector fix.** `input[type=number].first` grabbed a
      hidden-tab input → `get_by_role("spinbutton", name="Face id")`. **DONE.**

- [x] T11 — **E re-point off canvas.** DONE. Dropped `[role='gridcell']`
      count + canvas row-click detail assertion; browser now asserts only the
      `stDataFrame` container renders (`:visible`-scoped to the active tab).
      Field/row correctness covered by
      `test_merged_clusters_service_synthetic.py` (incl. real budapest fixture).
      **B/D/E green** (`pytest -m budapest -k "load_reference or face_analysis
      or merged_clusters"` -> 3 passed).

- [x] T12 — **F data decision: filed SIGHTING-092.** Reference run has 0
      filter_decisions (telemetry `tab.done name=quality n_items=0`); F's
      "220–240 rejected" is unmeetable. Resolution options (a) v2 reference
      run with quality data or (b) redefine F to empty-state — **needs user
      decision**, do NOT fudge. Sighting written.

- [x] T13 — **C: replaced by a headless endpoint test (DONE).** Per "test
      the endpoint" decision. New `test_recluster_reference_run_lands_in_
      expected_band` in `views/test_recluster_service_synthetic.py` reclusters
      the real reference run via `ReclusterService` and asserts parent linkage
      + the 12-18 band. **Passes in 5s** (vs the 5-min flaky browser path).
      Surfaced a real fact: raw `FCParams()` -> 22 clusters; only **profile_4**
      params -> the ~15 baseline, so the band is profile_4-specific (band
      assumption in e2e conftest was never validated because browser C never
      ran past the row-pick).
      REMAINING (separate, not blocking C's coverage):
        - T14: demote/skip the browser Scenario C (slow-paint) with a pointer
          to this endpoint test.
        - T15: file a sighting for the Recluster tab painting too slowly
          in-browser (heaviest tab; ~30 widgets re-rendered every run).

- [ ] T08 — Docs + close-out: CHANGES_LOG entry; flip SIGHTING-091 to
      RESOLVED (note: a11y gap on the dataframe stays as a known limitation,
      per spec Out-of-scope); update v2 e2e README if preamble wording
      referenced the row-pick; run `/code-review` → REVIEW.md; flip spec
      Status → Implemented only after REVIEW.md has no High findings.

## Acceptance criteria → task map

| AC | Task(s) |
|----|---------|
| AC1 honors `?current_run_dir` on first render | T01, T02 |
| AC2 fixture exists | T03 |
| AC3 B/C/D/E/F use fixture, row-pick removed | T04, T05 |
| AC4 all 6 budapest scenarios green | T05, T06 |
| AC5 History UI unchanged (`run_table`/`actions_table`) | T07 |
| AC6 seeder opt-in (no-param no-op) | T01, T02 |

## Open decision for the user

T05 changes Scenario B's character (load-flow assertion → analysis-tab
render assertion). Spec §"Coverage we lose" pre-approved this, but flagging
since it's the one scenario whose *purpose* shifts rather than just its
preamble.
