# Tasks: ML Model Expansion & Merge Interpretability (011)

**Input**: `specs/011-ml-model-expansion/` — spec.md, plan.md, data-model.md, contracts/
**Feature**: Six phases — new models, merge evidence, importance dashboard, manual merge, harvest negatives, polish

---

## Phase 1: Backend — New Models + Importance Infrastructure

**Purpose**: All 6 model types trainable with importance extraction and config mapping.

- [ ] T001 Verify `.venv/Scripts/python -m pytest tests/face_clustering/ -v` passes clean before any changes
- [ ] T002 Add `decision_tree` branch to `_make_estimator()` in `face_cluster/ml_trainer.py`: `DecisionTreeClassifier(max_depth=5, class_weight="balanced")`
- [ ] T003 Add `random_forest` branch to `_make_estimator()` in `face_cluster/ml_trainer.py`: `RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", n_jobs=-1)`
- [ ] T004 Add `catboost` branch to `_make_estimator()` in `face_cluster/ml_trainer.py` with try/except ImportError guard (same pattern as xgboost): `CatBoostClassifier(iterations=100, depth=4, verbose=0, auto_class_weights="Balanced")`
- [ ] T005 Add `decision_tree`, `random_forest`, `catboost` branches to `_extract_importance()` in `face_cluster/ml_trainer.py` — all three use `model.feature_importances_`
- [ ] T006 Add `FEATURE_CONFIG_MAP: Dict[str, Optional[Dict]]` constant to `face_cluster/ml_trainer.py` per data-model.md — maps feature names to config field, default value, and hint
- [ ] T007 [P] Write unit tests in `tests/face_clustering/test_ml_trainer.py`: `ut_NewModelTypes` class with tests for each new model type — train, predict_proba, feature_importances_, save/load roundtrip. Use synthetic 2-class data (no real embeddings needed).

**Checkpoint**: `pytest tests/face_clustering/test_ml_trainer.py -v` — all tests pass.

---

## Phase 2: Backend — Merge Evidence + Negative Harvesting + DB Migration

**Purpose**: Standalone functions for computing pair evidence and harvesting negatives; training DB supports `source` column.

- [ ] T008 Implement `evaluate_pair_evidence(cluster_id_a, cluster_id_b, cluster_result, distance_matrix, config) -> Dict` as module-level function in `face_cluster/merge.py` per contract in `contracts/merge-evidence-api.md` — wraps ConservativeMerger._evaluate_merge_evidence()
- [ ] T009 Implement `harvest_negative_pairs(cluster_result, distance_matrix, candidate_threshold, min_dist, max_count, exclude_pairs, random_state) -> List[Tuple[int,int,float]]` in `face_cluster/merge.py` per contract
- [ ] T010 Add `source` TEXT column migration to `init_training_table()` in `face_cluster/training_db.py` — check `PRAGMA table_info`, ALTER TABLE if absent, DEFAULT 'human'
- [ ] T011 Update `_COLS` list in `face_cluster/training_db.py` to include `"source"`; update `training_data_summary()` to return `by_source` breakdown
- [ ] T012 [P] Write unit tests in `tests/face_clustering/test_merge.py`: `ut_EvaluatePairEvidence` class — test evidence for a candidate pair (all gates), test for a non-candidate pair (exemplar gate fails), test return dict keys
- [ ] T013 [P] Write unit tests: `ut_HarvestNegativePairs` — test count limit, test min_dist floor, test exclude_pairs, test empty result when all pairs are candidates
- [ ] T014 [P] Write unit tests in `tests/face_clustering/test_training_db.py`: `ut_SourceColumn` — test source column created on fresh DB, test upsert with source='auto_negative', test load_training_data returns source column, test training_data_summary by_source

**Checkpoint**: `pytest tests/face_clustering/test_merge.py tests/face_clustering/test_training_db.py -v` — all tests pass.

---

## Phase 3: UI — Feature Importance Dashboard (US2)

**Purpose**: After training, show ranked feature importance chart and config mapping table.

**Depends on**: Phase 1 complete (FEATURE_CONFIG_MAP, new model types).

- [ ] T015 Add `_render_feature_importance_chart(train_result: TrainResult)` function in `app/face_clustering.py`: Plotly horizontal bar chart sorted by abs(importance); LR: signed coefficients with green (merge) / red (reject) coloring; tree-based: unsigned Gini/gain single color; MLP: proxy with "(approx)" note
- [ ] T016 Add config mapping table below the chart in `app/face_clustering.py`: reads `FEATURE_CONFIG_MAP` from ml_trainer, shows top-10 features with [Feature | Importance | Config Field | Current Default | Adjustment Hint]; features with no config mapping shown as "—"
- [ ] T017 Wire `_render_feature_importance_chart()` into the ML Training tab training results section in `app/face_clustering.py` — render after successful train, inside an expander "Feature Analysis"

**Checkpoint**: Train any model type in ML Training tab — importance chart and config table render.

---

## Phase 4: UI — Manual Merge Widget (US3)

**Purpose**: Select two clusters, preview with full merge metrics (same as Merge Analysis), confirm merge.

**Depends on**: Phase 2 complete (evaluate_pair_evidence).

- [ ] T018 Add "Manual Merge" expander section in Merge Analysis tab in `app/face_clustering.py`: two `st.selectbox` for cluster IDs (populated from `result.cluster_result.clusters.keys()`), "Preview" button
- [ ] T019 Implement preview rendering in `app/face_clustering.py`: on Preview click, call `evaluate_pair_evidence()`, render exemplar crops from both clusters (reuse `_crop_for_face`), render 4-gate pass/fail badges (same HTML as `_render_grouped_merge_gallery` gate section), show warning banner if not a candidate, show cluster sizes and total faces
- [ ] T020 Implement "Confirm Merge" button in `app/face_clustering.py`: calls `save_manual_merge_snapshot(faces, merged_cluster_result, approved_pairs=[(cid_a, cid_b)], ...)`, saves training sample to DB (source="human", label=1), appends session step with action="manual_merge", reloads result via `_invalidate_run_caches()`

**Checkpoint**: Select two non-candidate clusters, preview shows all 4 gates, confirm merge applies it.

---

## Phase 5: UI — Harvest Negatives (US4)

**Purpose**: Button in ML Training tab to sample non-candidate pairs as negative training data.

**Depends on**: Phase 2 complete (harvest_negative_pairs, DB source column).

- [ ] T021 Add "Harvest Negatives" section in ML Training tab in `app/face_clustering.py` (after training data summary section): count slider (10–500, default 100), min_dist slider (0.45–1.0, default 0.5, step 0.05), "Harvest" button
- [ ] T022 Implement harvest logic in `app/face_clustering.py`: on Harvest click, load existing human-labeled pairs from DB as exclude set, call `harvest_negative_pairs()`, compute features for harvested pairs via `FeatureComputer.compute_pair_features()`, upsert to training DB with source='auto_negative' and label=0
- [ ] T023 Show harvest summary in `app/face_clustering.py`: N pairs harvested, exemplar distance range (min–max), training data balance before (pos/neg counts) and after, refresh training data summary display

**Checkpoint**: Harvest 50 negatives — training data summary shows updated counts with "auto_negative" source.

---

## Phase 6: Polish

- [ ] T024 Update `CHANGES_LOG.md` with `[FEATURE]` entry covering all phases
- [ ] T025 Update `docs/FEATURE_REQUESTS.md`: mark spec 011 status
- [ ] T026 Run full test suite `.venv/Scripts/python -m pytest tests/face_clustering/ -v` and confirm all tests pass on Windows
- [ ] T027 Append learnings to `docs/LEARNINGS.md`

---

## Dependencies & Execution Order

```
T001 (baseline)
  └─> T002–T006 (new models + FEATURE_CONFIG_MAP)
       └─> T007 (model tests)
            └─> T008–T011 (merge evidence + harvest + DB)
                 └─> T012–T014 (backend tests)
                      ├─> T015–T017 (US2 importance dashboard)
                      ├─> T018–T020 (US3 manual merge)
                      └─> T021–T023 (US4 harvest negatives)
                           └─> T024–T027 (polish)
```

### Parallel opportunities

- T002, T003, T004 can be done in one pass (same function, sequential branches)
- T015–T017 (US2), T018–T020 (US3), T021–T023 (US4) can all proceed in parallel after Phase 2
- T012, T013, T014 can all be written in parallel (different test classes)
