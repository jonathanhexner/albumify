# Tasks: ML Merge Interface (010)

**Input**: `specs/010-ml-merge-interface/` — spec.md, plan.md, data-model.md, contracts/
**Feature**: Two-phase delivery — Phase 1: three-state decisions; Phase 2: ML mode

---

## Phase 1: Setup

**Purpose**: No new files needed — all changes are additive to existing modules.

- [ ] T001 Verify `.venv/Scripts/python -m pytest tests/face_clustering/ -v` passes clean before any changes in `tests/face_clustering/`

---

## Phase 2: Foundational

**Purpose**: Backend data model and API extensions that both US1 and US2 depend on. Must complete before any UI work.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T002 Add `ml_prob: Optional[float] = None` and `ml_pred: Optional[int] = None` fields to `MergeDecisionRow` dataclass in `face_cluster/analysis_views.py`
- [ ] T003 Add `ml_threshold: Optional[float] = None` field to `MergeAnalysisView` dataclass in `face_cluster/analysis_views.py`
- [ ] T004 [P] Implement `compute_ml_merge_view(result, model_payload, threshold, candidate_threshold)` function in `face_cluster/analysis_views.py` per contract in `specs/010-ml-merge-interface/contracts/analysis-views-api.md` — includes probability-to-gate mapping, feature compute, predict, group, and return `MergeAnalysisView`
- [ ] T005 [P] Implement `compute_pair_feature_contributions(pair_features_dict, model_payload, top_n=3)` function in `face_cluster/analysis_views.py` per contract — logistic regression exact, XGBoost/MLP proxy with "(global)" suffix
- [ ] T006 [P] Write unit tests for `compute_ml_merge_view` in `tests/face_clustering/test_merge_analysis.py`: `ut_MLMergeView` class with tests: `test_compute_returns_merge_analysis_view`, `test_high_prob_pairs_in_merges_list`, `test_low_prob_pairs_in_rejections_list`, `test_borderline_pairs_in_near_misses`, `test_feature_version_mismatch_raises`, `test_probability_to_gate_mapping`
- [ ] T007 [P] Write unit tests for `compute_pair_feature_contributions` in `tests/face_clustering/test_merge_analysis.py`: `ut_FeatureContributions` class with tests: `test_logistic_regression_exact`, `test_xgboost_proxy_returns_top3`

**Checkpoint**: Run `pytest tests/face_clustering/test_merge_analysis.py -v` — all new tests pass, existing tests unchanged.

---

## Phase 3: User Story 1 — Three-State Decision Refactor (P1)

**Goal**: All merge candidates start Undecided. Users explicitly approve or reject. Training data only includes human decisions. Smart Approve and Smart Reject are separate explicit buttons.

**Independent Test**: Load a run in heuristic mode, verify all pairs start undecided, click Smart Approve, verify only auto_approve groups change, click Apply + Remerge with zero approvals (button disabled), approve two pairs, apply — only those two are merged.

### Tests

- [ ] T008 [P] [US1] Add `test_merge_analysis_starts_all_undecided` to `tests/face_clustering/test_streamlit_app.py`: after loading a run, assert `st.session_state.merge_approval_decisions == {}`
- [ ] T009 [P] [US1] Add `test_smart_approve_only_sets_high_confidence` to `tests/face_clustering/test_streamlit_app.py`: Smart Approve does not change review or auto_reject group decisions
- [ ] T010 [P] [US1] Add `test_smart_reject_only_sets_low_confidence` to `tests/face_clustering/test_streamlit_app.py`: Smart Reject does not change review or auto_approve group decisions
- [ ] T011 [P] [US1] Add `test_apply_button_disabled_with_no_approvals` to `tests/face_clustering/test_streamlit_app.py`: Apply + Remerge button disabled when `merge_approval_decisions` is empty
- [ ] T012 [P] [US1] Add three-state training data tests to `tests/face_clustering/test_merge_analysis.py`: `ut_ThreeState` class with `test_undecided_excluded_from_training_data`, `test_human_override_included_in_training_data`, `test_ml_suggested_approve_not_in_training_data`

### Implementation

- [ ] T013 [US1] Remove auto-pre-fill block (lines ~2505-2518) from `render_merge_analysis_tab()` in `app/face_clustering.py` — all pairs now start absent from `merge_approval_decisions`
- [ ] T014 [US1] Initialize `st.session_state.merge_decision_sources = {}` alongside `merge_approval_decisions` in `app/face_clustering.py` — add to all init/clear/invalidate locations
- [ ] T015 [US1] Update all decision writes in `app/face_clustering.py` to also write `merge_decision_sources[pair_key]`: user button clicks → `"human"`, Smart Approve/Reject → `"human"`, pair clear → remove from both dicts
- [ ] T016 [US1] Split "Smart Approve" button into two in `_render_approval_controls()` in `app/face_clustering.py`: "Smart Approve" (sets only auto_approve groups, source="human") and "Smart Reject" (sets only auto_reject groups, source="human") — remove the side-effect that set `merge_group_filter = "Review Only"`
- [ ] T017 [US1] Add three-state summary bar above gallery in `app/face_clustering.py`: `st.columns(3)` showing Approved / Rejected / Undecided counts with `st.metric()`
- [ ] T018 [US1] Gate "Apply + Remerge" button on `n_approved >= 1` in `_render_approval_controls()` in `app/face_clustering.py`: disable button with message "Approve at least one pair to merge." when no approvals; show info "N pairs undecided — they will be re-evaluated in the next round." when `n_undecided > 0`
- [ ] T019 [US1] Update `_collect_all_merge_labels()` signature and logic in `app/face_clustering.py` to accept `sources: Dict` and filter: only write to training DB when `sources.get(pair_key) == "human"` per contract in `specs/010-ml-merge-interface/contracts/training-data-discipline.md`
- [ ] T020 [US1] Update session step metadata in `app/face_clustering.py` Apply + Remerge handler: pass `merge_mode`, `n_approved_human`, `n_approved_ml`, `n_rejected_human`, `n_rejected_ml`, `n_undecided`, `n_training_samples_saved` in `params` dict to `SessionManager.append_step()`
- [ ] T021 [US1] Update gallery card rendering in `_render_grouped_merge_gallery()` in `app/face_clustering.py`: show "ML" badge on decisions where `merge_decision_sources[pair_key] == "ml"`; neutral grey styling for Undecided; clicking Approve/Reject on an ML-suggested pair changes source to "human" and removes ML badge

**Checkpoint**: Run `pytest tests/face_clustering/ -v` — all tests pass. Manually verify: load a run, confirm all pairs start undecided, Smart Approve sets only high-confidence groups.

---

## Phase 4: User Story 2 — ML Model as Merge Proposer (P1)

**Goal**: Mode selector in Merge Analysis tab. Select an ML model, click "Apply threshold", model pre-fills high-confidence pairs. Borderline pairs stay undecided. Apply + Remerge executes ML-suggested and human approvals; only human decisions saved to training DB.

**Independent Test**: Load a run, switch to ML mode, select a model, click "Apply threshold", verify probability badges on cards, override one ML-suggested reject to approve, apply — overridden pair saved as training sample, untouched ML suggestions not saved.

**Depends on**: Phase 3 complete (three-state session state infrastructure).

### Tests

- [ ] T022 [P] [US2] Add `test_mode_selector_shows_model_dropdown_in_ml_mode` to `tests/face_clustering/test_streamlit_app.py`: switching mode radio to "ML Model" reveals model selectbox
- [ ] T023 [P] [US2] Add `test_ml_prefill_sets_source_ml` to `tests/face_clustering/test_streamlit_app.py`: after ML pre-fill, `merge_decision_sources` contains "ml" for pre-filled pairs

### Implementation

- [ ] T024 [US2] Add mode selector radio ("Heuristic (4-gate)" / "ML Model") at top of `render_merge_analysis_tab()` in `app/face_clustering.py`; store in `st.session_state.merge_mode` (default "heuristic")
- [ ] T025 [US2] Add ML controls row in `app/face_clustering.py` (visible only when merge_mode == "ml_model"): model dropdown (`st.selectbox` from `load_model_records()`), threshold slider (0.1–0.9, default 0.5, step 0.05, key "ml_threshold"), "Wider candidates" checkbox (sets `ml_candidate_threshold` to 0.65), "Apply threshold" button
- [ ] T026 [US2] Implement async prediction worker in `app/face_clustering.py` triggered by "Apply threshold": creates `_AsyncState` worker that calls `compute_ml_merge_view(result, payload, threshold, candidate_threshold)`, stores in `st.session_state.ml_predict_worker`; on completion stores view in `st.session_state.merge_analysis_view` and applies ML pre-fill to `merge_approval_decisions` + `merge_decision_sources`
- [ ] T027 [US2] Implement ML pre-fill logic in `app/face_clustering.py`: after prediction completes, for each pair: `prob >= threshold` → decisions[key]="approve", sources[key]="ml"; `prob < (1 - threshold)` → decisions[key]="reject", sources[key]="ml"; else leave absent (Undecided); applied once and only if pair has no prior human decision
- [ ] T028 [US2] Branch gallery card rendering in `_render_grouped_merge_gallery()` in `app/face_clustering.py` on `merge_mode`: in ML mode replace gate-count badge with probability badge (colour-coded: green >=0.8, amber 0.5-0.8, red <0.5, format "[MERGE: 85%]" or "[REJECT: 12%]")
- [ ] T029 [US2] Add ML detail section inside expandable group/pair card in `app/face_clustering.py`: per-pair shows `prob=0.92 (threshold=0.50)`; feature contributions from `compute_pair_feature_contributions()` as 3-row table with feature name, value, direction arrow; heuristic gates shown read-only below as "Heuristic reference: N/4 gates"
- [ ] T030 [US2] Replace "Apply to Current Run" flat table in `render_ml_training_tab()` Section 7 in `app/face_clustering.py` with: compact 5-row preview table (sorted by ml_prob desc) + "Open in Merge Analysis" button that sets `st.session_state.merge_mode = "ml_model"`, `st.session_state.ml_selected_model = sel_model_name`, and switches active tab to Merge Analysis

**Checkpoint**: Run `pytest tests/face_clustering/ -v` — all tests pass. Manually verify: select a model, apply threshold, confirm probability badges, override one ML decision, apply — only overridden pair in training DB.

---

## Phase 5: User Story 3 — ML Probability Overview Panel (P2)

**Goal**: Probability histogram and model summary shown above the gallery in ML mode. Threshold slider updates counts. Low-separation warning when distribution is flat.

**Independent Test**: Apply ML predictions, verify histogram renders, move threshold slider and click Apply, verify summary counts update.

**Depends on**: Phase 4 complete (ML prediction data available).

- [ ] T031 [US3] Implement `_render_ml_overview_panel(view, model_metadata, threshold)` in `app/face_clustering.py`: shows model name + AUC + F1 + N samples row; plotly histogram of ml_prob values (bins 0.1); summary counts "Merge: N | Reject: M | Borderline: K"; group tier counts "N high-conf | M review | K low-conf"
- [ ] T032 [US3] Add low-separation warning in `_render_ml_overview_panel()` in `app/face_clustering.py`: compute distribution entropy or variance; if flat (< 0.1 variance or > 0.8 entropy) show `st.warning("Model shows low confidence separation — consider retraining with more data.")`
- [ ] T033 [US3] Wire `_render_ml_overview_panel()` into ML mode render path in `render_merge_analysis_tab()` in `app/face_clustering.py`: shown between mode controls and gallery when predictions are loaded

**Checkpoint**: Overview panel appears in ML mode with histogram and counts.

---

## Phase 6: User Story 4 — Per-Pair Feature Contributions (P2)

**Goal**: In ML mode, expanded pair detail shows top-3 feature contributions with name, value, and direction. Logistic regression is exact; XGBoost/MLP shows global proxy with disclaimer.

**Independent Test**: Expand a pair card in ML mode, verify 3 feature rows with names, values, and direction arrows render.

**Depends on**: Phase 4 (T029 skeleton already in place — this completes it).

- [ ] T034 [US4] Complete feature contributions rendering in the ML detail section (T029 skeleton) in `app/face_clustering.py`: fetch per-pair features from `st.session_state.ml_pair_features` (stored alongside predictions); call `compute_pair_feature_contributions()`; render as 3-column table (Feature | Value | Direction); add "(global)" suffix to direction for non-LR models
- [ ] T035 [US4] Store per-pair `ClusterPairFeatures` alongside ML predictions in `app/face_clustering.py`: when prediction worker completes, also store `st.session_state.ml_pair_features: Dict[Tuple[int,int], dict]` from `FeatureComputer().compute_all_pairs()` output (already computed during `compute_ml_merge_view`)

**Checkpoint**: Expand any pair card in ML mode — top-3 contributing features shown with values and direction.

---

## Phase 7: User Story 5 — Heuristic vs ML Comparison (P3)

**Goal**: Optional comparison mode overlay. Each pair shows both ML and heuristic decisions with agreement indicator. "Disagreements only" filter available.

**Independent Test**: Enable comparison checkbox, verify agreement indicators appear on all cards, filter to Disagreements only, verify only disagreeing pairs shown.

**Depends on**: Phase 4 (ML predictions) + run must have existing heuristic merge_log.

- [ ] T036 [US5] Add "Show heuristic comparison" checkbox in ML mode controls row in `app/face_clustering.py` (only visible when merge_mode=="ml_model" AND `result.merge_log` is non-empty); store in `st.session_state.ml_show_comparison`
- [ ] T037 [US5] Add agreement indicator to gallery card header in `_render_grouped_merge_gallery()` in `app/face_clustering.py` when comparison enabled: compute agreement per pair using heuristic `action` field vs ML `ml_pred`; display "Both merge" / "Both reject" / "ML: merge, Heuristic: reject" / "ML: reject, Heuristic: merge"
- [ ] T038 [US5] Add "Disagreements only" option to the existing gallery filter dropdown in `app/face_clustering.py` (visible only when comparison mode active); filters to pairs where ML and heuristic disagree

**Checkpoint**: Comparison mode shows agreement badges; Disagreements filter works.

---

## Phase 8: User Story 6 — Wider Candidate Discovery (P3)

**Goal**: "Wider candidates" checkbox raises candidate threshold to 0.65. Extended-range pairs shown with visual marker. Count preview before applying.

**Independent Test**: Enable wider candidates, verify additional pairs appear with "Extended range" badge; verify standard pairs (within 0.45) have no badge.

**Depends on**: Phase 4 (candidate_threshold already threaded through prediction path).

- [ ] T039 [US6] Wire "Wider candidates" checkbox state in `app/face_clustering.py` T025 (already added): when checked, set `st.session_state.ml_candidate_threshold = 0.65`; show info before applying: "This will evaluate N additional pairs (beyond the default 0.45 threshold)"
- [ ] T040 [US6] Add "Extended range" visual badge to gallery cards for pairs where `exemplar_dist > 0.45` in `_render_grouped_merge_gallery()` in `app/face_clustering.py` (only visible when wider candidates is active)

**Checkpoint**: Wider candidates toggle produces additional pairs with Extended range badge.

---

## Phase 9: Polish

- [ ] T041 [P] Update `CHANGES_LOG.md` with `[FEATURE]` entry covering all phases implemented
- [ ] T042 [P] Update `docs/FEATURE_REQUESTS.md`: mark spec 010 as "Done" once all phases complete
- [ ] T043 Run full test suite `.venv/Scripts/python -m pytest tests/face_clustering/ -v` and confirm all tests pass on Windows
- [ ] T044 Append learning to `docs/LEARNINGS.md` for any non-obvious discoveries during implementation

---

## Dependencies & Execution Order

```
T001 (baseline)
  └─> T002, T003, T004, T005 (foundational dataclass + backend functions)
       └─> T006, T007 (backend tests — can run once T002–T005 done)
            └─> T008–T012 (US1 tests)
                 └─> T013–T021 (US1 implementation)
                      └─> T022–T030 (US2 tests + implementation)
                           ├─> T031–T033 (US3 overview panel)
                           ├─> T034–T035 (US4 feature contributions)
                           ├─> T036–T038 (US5 comparison)
                           └─> T039–T040 (US6 wider candidates)
                                └─> T041–T044 (polish)
```

### Parallel opportunities

- T004 and T005 can be developed in parallel (different functions in same file — coordinate to avoid merge conflicts)
- T006 and T007 can be written in parallel
- T008–T012 can all be written in parallel (each is a different test class/function)
- T031–T033 (US3), T034–T035 (US4), T036–T038 (US5), T039–T040 (US6) can all proceed in parallel after Phase 4 (US2) is complete

---

## Implementation Strategy

### MVP (Phase 1–3 only = US1 three-state refactor)

1. T001 — verify clean baseline
2. T002–T007 — foundational dataclass + backend (even if not yet used by UI)
3. T008–T021 — US1 three-state UI
4. **STOP AND VALIDATE**: Run tests, manually verify undecided default, Smart Approve/Reject separation, Apply gating

### Full Delivery

After MVP validation:
5. T022–T030 — US2 ML mode
6. T031–T033 — US3 overview panel
7. T034–T035 — US4 feature contributions
8. T036–T038 — US5 comparison (optional, P3)
9. T039–T040 — US6 wider candidates (optional, P3)
10. T041–T044 — polish

---

## Notes

- All tests are in `tests/face_clustering/` — use `AppTest` for Streamlit tests, plain pytest for backend
- No new module files — all changes are additive to `face_cluster/analysis_views.py` and `app/face_clustering.py`
- Threshold change (T027) only re-applies pre-fill from cached predictions — does NOT re-run the model
- T029 creates the feature contributions skeleton; T034–T035 complete it — coordinate file edits
- Windows: no Unicode characters in any new log/progress output
