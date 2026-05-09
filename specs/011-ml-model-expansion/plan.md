# Implementation Plan: ML Model Expansion & Merge Interpretability

**Spec**: `specs/011-ml-model-expansion/spec.md`
**Created**: 2026-04-22
**Status**: Ready for tasks

## Technical Context

- **Language**: Python 3.10+
- **UI Framework**: Streamlit (async compute via `_AsyncState` pattern)
- **ML Stack**: scikit-learn, xgboost (optional), catboost (optional)
- **DB**: SQLite via `face_cluster/training_db.py`
- **Existing infrastructure**: `face_cluster/ml_trainer.py` (train/predict/save/load), `face_cluster/merge.py` (ConservativeMerger, propose_merge_candidates), `face_cluster/manual_merge_snapshot.py` (union-find snapshot writer)

## Design Decisions

### D1: New model types — extension point in `_make_estimator()`

Add three elif branches to `_make_estimator()` for `decision_tree`, `random_forest`, `catboost`. CatBoost import guarded with try/except (same as XGBoost). All new models expose `predict_proba()` and `feature_importances_`.

**Files**: `face_cluster/ml_trainer.py`

### D2: Feature importance chart — rendered in ML Training tab after training

After `MergeTrainer.train()` returns a `TrainResult`, the app renders a Plotly bar chart sorted by `abs(importance)`. LR shows signed coefficients (color-coded: green=toward merge, red=toward reject). Tree-based models show unsigned Gini/gain importance. MLP shows weight-magnitude proxy.

Below the chart: a table mapping top-N features to `PipelineConfig` fields and their current defaults, sourced from `FEATURE_CONFIG_MAP`.

**Files**: `face_cluster/ml_trainer.py` (FEATURE_CONFIG_MAP), `app/face_clustering.py` (chart render)

### D3: Manual merge — new UI widget + existing snapshot infrastructure

A "Manual Merge" section is added to the **Merge Analysis** tab. Two selectboxes for cluster IDs -> "Preview" button -> renders evidence (same 4-gate display as Merge Analysis pair cards, including exemplar distance, support, margin, diameter) + exemplar crops -> "Confirm Merge" button -> calls `save_manual_merge_snapshot()` with `approved_pairs=[(cid_a, cid_b)]`.

Gate evidence is computed via a new `evaluate_pair_evidence()` module-level function in `merge.py` (wraps `_evaluate_merge_evidence()`).

**Files**: `face_cluster/merge.py` (evaluate_pair_evidence), `app/face_clustering.py` (manual merge widget)

### D4: Negative harvesting — new function + UI button

`harvest_negative_pairs()` in `merge.py` enumerates non-candidate pairs, filters by min_dist, samples up to N. Returns `(cid_a, cid_b, exemplar_dist)` tuples.

The ML Training tab gets a "Harvest negatives" section with count slider and min_dist slider. Button calls `harvest_negative_pairs()`, computes features via `FeatureComputer.compute_pair_features()`, and calls `upsert_training_samples()` with `source='auto_negative'`.

**Files**: `face_cluster/merge.py` (harvest function), `face_cluster/training_db.py` (source column migration), `app/face_clustering.py` (UI)

### D5: Training DB `source` column — schema migration

`init_training_table()` checks `PRAGMA table_info(merge_training_data)` for `source` column. If absent, runs `ALTER TABLE`. Default `'human'` for existing rows.

**Files**: `face_cluster/training_db.py`

## Phase Plan

### Phase 1: Backend — New Models + Importance Infrastructure

**Goal**: All 6 model types trainable, importance extraction works, FEATURE_CONFIG_MAP defined.

1. Add `decision_tree`, `random_forest`, `catboost` branches to `_make_estimator()` in `ml_trainer.py`.
2. Add the same three branches to `_extract_importance()`.
3. Add `FEATURE_CONFIG_MAP` constant to `ml_trainer.py`.
4. Update `compute_pair_feature_contributions()` in `analysis_views.py` if needed (verify `hasattr(model, 'feature_importances_')` path covers new types).
5. Unit tests: train each new model type, verify `predict_proba`, `feature_importances_`, save/load roundtrip.

**Files modified**: `face_cluster/ml_trainer.py`, `tests/face_clustering/test_ml_trainer.py`

### Phase 2: Backend — Merge Evidence + Negative Harvesting

**Goal**: `evaluate_pair_evidence()` and `harvest_negative_pairs()` available as standalone functions.

1. Extract `evaluate_pair_evidence()` as a module-level function in `merge.py`.
2. Implement `harvest_negative_pairs()` in `merge.py`.
3. Add `source` column migration to `training_db.py`.
4. Update `_COLS`, `upsert_training_samples()`, `training_data_summary()`.
5. Unit tests: evidence for candidate and non-candidate pairs, harvesting with exclusion, DB source column roundtrip.

**Files modified**: `face_cluster/merge.py`, `face_cluster/training_db.py`, `tests/face_clustering/test_merge.py`, `tests/face_clustering/test_training_db.py`

### Phase 3: UI — Feature Importance Dashboard (US2)

**Goal**: Bar chart + config mapping table rendered after training in ML Training tab.

1. After `MergeTrainer.train()` in the training section, render feature importance chart (Plotly horizontal bar, sorted by abs value, colored by sign for LR, single color for tree-based).
2. Below chart: table with columns [Feature, Importance, Config Field, Default, Hint].
3. Model type indicator determines chart styling (signed vs unsigned).

**Files modified**: `app/face_clustering.py`

### Phase 4: UI — Manual Merge Widget (US3)

**Goal**: Select two clusters, preview full merge evidence (same metrics as Merge Analysis pair cards), confirm merge.

1. Add "Manual Merge" section to Merge Analysis tab (below the gallery, or as a collapsed expander).
2. Two `st.selectbox` for cluster IDs (populated from loaded result's cluster list).
3. "Preview" button -> calls `evaluate_pair_evidence()` -> renders:
   - Exemplar crops from both clusters (same as pair card layout)
   - 4-gate pass/fail badges (exemplar, support, margin, diameter) with values and deltas
   - Warning banner if exemplar dist > candidate threshold: "Not a merge candidate (dist=X > threshold=Y)"
   - Summary: total faces, cluster sizes
4. "Confirm Merge" button -> calls `save_manual_merge_snapshot()` -> saves training sample (source="human", label=1) -> reloads result.

**Files modified**: `app/face_clustering.py`

### Phase 5: UI — Harvest Negatives (US4)

**Goal**: Button in ML Training tab to sample non-candidate pairs as negatives.

1. Add "Harvest Negatives" section in ML Training tab.
2. Controls: count slider (10–500, default 100), min_dist slider (0.45–1.0, default 0.5).
3. "Harvest" button -> calls `harvest_negative_pairs()` -> computes features -> upserts to DB with source='auto_negative'.
4. Show summary: N pairs harvested, distance range, training data balance before/after.

**Files modified**: `app/face_clustering.py`

### Phase 6: Polish

1. Update `CHANGES_LOG.md`.
2. Update `docs/FEATURE_REQUESTS.md`.
3. Full test run.
4. Log learnings.

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CatBoost not installed in env | Training fails with cryptic error | Import guard with clear error message; disable option in UI |
| Small training set + complex model (RF/CatBoost) | Overfitting, misleading metrics | Default `max_depth=5`; cross-validation enforced; show train vs test metrics |
| Auto-negative pairs too easy | Model learns trivially, poor generalization near boundary | `min_dist` floor (default 0.5) keeps negatives above candidate threshold but not trivially distant |
| Manual merge bypasses all gates | User merges dissimilar clusters | Preview shows all gate results with warnings; merge is reversible via session branching |

## Dependencies

```
Phase 1 (new models + FEATURE_CONFIG_MAP)
  |
  v
Phase 2 (merge evidence + harvest + DB migration)
  |
  +---> Phase 3 (importance chart)     -- depends on FEATURE_CONFIG_MAP
  +---> Phase 4 (manual merge widget)  -- depends on evaluate_pair_evidence()
  +---> Phase 5 (harvest negatives UI) -- depends on harvest_negative_pairs() + DB source column
         |
         v
       Phase 6 (polish)
```

Phases 3, 4, 5 are independent of each other and can be parallelized.
