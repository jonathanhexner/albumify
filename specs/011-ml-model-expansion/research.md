# Research: ML Model Expansion & Merge Interpretability

## R1: New Model Types — Implementation Feasibility

**Decision**: Add `decision_tree`, `random_forest`, `catboost` to `_make_estimator()` in `ml_trainer.py`.

**Rationale**: All three have `predict_proba()` and `feature_importances_` (sklearn API compliant). CatBoost requires an optional import (same pattern as XGBoost). DecisionTree/RandomForest are in sklearn — zero new dependencies.

**Alternatives considered**:
- LightGBM: Similar to XGBoost/CatBoost but less interpretable; deferred.
- Extra Trees: Marginal benefit over RandomForest; not requested.

**Key implementation notes**:
- `DecisionTreeClassifier`: `max_depth=5` default (prevent extreme overfitting on small datasets).
- `RandomForestClassifier`: `n_estimators=100, max_depth=5, class_weight="balanced"`.
- `CatBoostClassifier`: `iterations=100, depth=4, verbose=0`. Import guarded like XGBoost.
- All three expose `feature_importances_` (Gini impurity). `_extract_importance()` needs one new elif branch for each.
- `compute_pair_feature_contributions()` already has a proxy path for tree models (uses `feature_importances_`). DecisionTree, RandomForest, CatBoost share this path — no new logic needed.

---

## R2: Feature Importance Extraction — Current State

**Decision**: Extend `_extract_importance()` with new model types and expose a standalone `get_feature_importance_chart_data()` function for the UI.

**Current state** (`_extract_importance` in ml_trainer.py):
- `logistic_regression`: `model.coef_[0]` — signed coefficients (positive = toward merge).
- `xgboost`: `model.feature_importances_` — unsigned Gini/gain.
- `mlp`: `np.abs(model.coefs_[0]).sum(axis=1)` — first-layer weight magnitude proxy.

**New model types** follow the XGBoost pattern:
- `decision_tree`: `model.feature_importances_` — unsigned Gini.
- `random_forest`: `model.feature_importances_` — unsigned (averaged across trees).
- `catboost`: `model.feature_importances_` — unsigned (prediction values contribution).

---

## R3: Feature-to-Config Parameter Mapping

**Decision**: Maintain a static `FEATURE_CONFIG_MAP` dict in `ml_trainer.py`.

| Feature | Config Field | Default | Notes |
|---------|-------------|---------|-------|
| `min_exemplar_dist` | `merge_exemplar_threshold` | 0.35 | Exemplar gate threshold |
| `support_fraction` | `merge_support_frac` | 0.3 | Min support fraction |
| `n_cross_pairs_below_threshold` | `merge_support_min` | 2 | Min absolute support count |
| `post_merge_diameter` | `merge_diameter_expansion_factor` | 1.5 | Max allowed diameter expansion |
| `dist_to_threshold_ratio` | `merge_exemplar_threshold` | 0.35 | Ratio feature — same config |
| `blur_min_a`, `blur_min_b` | `blur_min` | 50.0 | Quality gating (indirect) |
| `frontal_frac_a`, `frontal_frac_b` | `yaw_max` / `pitch_max` | 30.0 / 25.0 | Pose gating (indirect) |
| `t_global` | (computed, not configurable) | — | Global threshold from exemplar P90 |
| `merge_candidate_threshold` | `merge_candidate_threshold` | 0.45 | Candidate discovery gate |

Features without a direct config parameter (e.g., `diameter_ratio`, `cross_dist_iqr`, `area_ratio`) are mapped to `null` — they're useful for ML but don't correspond to a tunable knob.

---

## R4: Computing Merge Evidence for Arbitrary Pairs

**Decision**: Extract gate evaluation from `ConservativeMerger._evaluate_merge_evidence()` into a standalone helper function.

**Current state**: `_evaluate_merge_evidence()` is a method on `ConservativeMerger`. It requires `cluster_result`, `distance_matrix`, `cluster_thresholds`, `global_threshold`. It returns a dict with all 4 gate results.

**Problem**: It's an instance method — needs a `ConservativeMerger` with a config. For US3 (manual merge preview), we need the same evidence for any pair.

**Solution**: Create `evaluate_pair_evidence(cluster_id_a, cluster_id_b, cluster_result, distance_matrix, config)` as a module-level function in `merge.py`. Internally instantiates a temporary `ConservativeMerger` and calls `_evaluate_merge_evidence()`. This avoids duplicating logic while making it callable from the app layer.

---

## R5: Training DB — Source Field

**Decision**: Add a `source` TEXT column to the `merge_training_data` table.

**Current state**: The training DB schema has no `source` column. The `upsert_training_samples()` function accepts any dict keys that match `_COLS`. Source tracking was added in spec 010 at the session-state level (`merge_decision_sources`) but never persisted to the training DB.

**Migration**: Add column with `ALTER TABLE merge_training_data ADD COLUMN source TEXT DEFAULT 'human'`. Existing rows default to `'human'` (correct — they predate ML pre-fill). New auto-negative samples will be `source='auto_negative'`.

---

## R6: Manual Merge Snapshot Reusability

**Decision**: Reuse `save_manual_merge_snapshot()` for US3. No new infrastructure needed.

**Current state**: `save_manual_merge_snapshot(faces, merged_cluster_result, approved_pairs, ...)` applies union-find, writes pipeline-compatible output. Fully UI-agnostic.

**US3 integration**: The app layer builds a single-pair `approved_pairs=[(cid_a, cid_b)]` and calls the existing function. The preview UI computes evidence via the new `evaluate_pair_evidence()` helper. Lineage tracking is built-in.
