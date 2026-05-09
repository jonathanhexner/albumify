# Feature Specification: ML Model Expansion & Merge Interpretability

**Feature Branch**: `011-ml-model-expansion`
**Created**: 2026-04-22
**Status**: Draft
**Input**: User description: "Add decision tree, catboost and random forest to ML models. Provide interpretability/analysis of ML model to understand which features matter and inform better default heuristic values. Add manual merge for non-candidate clusters. Use non-candidate cluster pairs as negative training samples to improve class balance."

## User Scenarios & Testing

### User Story 1 — Additional ML Model Types (Priority: P1)

The user trains merge classifiers in the ML Training tab. Currently only logistic regression, XGBoost, and MLP are available. The user wants to also train and evaluate decision tree, random forest, and CatBoost models. These tree-based models offer different bias/variance trade-offs and may outperform the current options on small training sets. The user selects the new model type from the same dropdown, trains, compares metrics, and saves the best model.

**Why this priority**: Directly expands the model zoo at minimal risk — purely additive to the existing training pipeline.

**Independent Test**: Select "random_forest" from the model type dropdown in ML Training tab, click Train, verify that training completes with AUC/F1 metrics displayed, and the model can be saved and loaded for prediction.

**Acceptance Scenarios**:

1. **Given** the ML Training tab is open, **When** the user selects "decision_tree" from model type, **Then** training produces a fitted DecisionTreeClassifier with metrics (AUC, F1, accuracy).
2. **Given** a random forest model is saved, **When** the user loads it in ML merge mode, **Then** predictions and probability badges render correctly.
3. **Given** a CatBoost model is trained, **When** feature importance is requested, **Then** the model returns feature importances (native `feature_importances_`).
4. **Given** any of the new model types, **When** `compute_pair_feature_contributions()` is called, **Then** it returns top-3 features using global importance proxy (same as XGBoost path).

---

### User Story 2 — Feature Importance & Interpretability Dashboard (Priority: P1)

After training a model, the user wants to understand which features drive merge/reject decisions and by how much. This serves two purposes: (a) validate that the model is learning meaningful patterns (not overfitting to noise), and (b) identify whether specific heuristic defaults (e.g., `merge_candidate_threshold`, `blur_min`, `merge_support_min`) could be adjusted to improve clustering quality without ML.

The ML Training tab shows a feature importance chart after training, ranked by impact. For logistic regression, the chart shows signed coefficients (positive = toward merge). For tree-based models, it shows Gini/gain importance. A summary table maps the top features to the corresponding heuristic config parameters and their current default values, so the user can see: "min_exemplar_dist matters most → current merge_candidate_threshold = 0.45 → consider adjusting."

**Why this priority**: High diagnostic value — helps the user decide whether to use ML at all or simply tune heuristic defaults.

**Independent Test**: Train a logistic regression model, verify the importance chart renders with all features ranked, and the config mapping table shows at least the top-5 features with their corresponding `PipelineConfig` fields.

**Acceptance Scenarios**:

1. **Given** a model has just been trained, **When** the user views the "Feature Analysis" section in ML Training, **Then** a bar chart of feature importances (sorted by absolute value, color-coded by direction) is shown.
2. **Given** the importance chart is visible, **When** the user looks at the config mapping table, **Then** each feature is mapped to the corresponding `PipelineConfig` / `ConservativeMergerConfig` field and current default value (e.g., `min_exemplar_dist → merge_candidate_threshold = 0.45`).
3. **Given** a tree-based model (XGBoost, random forest, CatBoost, decision tree), **When** the importance chart renders, **Then** importances are unsigned (Gini/gain — direction is not available for tree splits) and labeled accordingly.
4. **Given** an MLP model, **When** the importance chart renders, **Then** a note states "First-layer weight proxy — treat as approximate."

---

### ~~User Story 3 — Manual Merge for Non-Candidate Clusters~~

> **Moved to `specs/014-force-merge/spec.md`** — this feature has no ML dependency and belongs as a standalone spec.

---

### User Story 4 — Non-Candidate Pairs as Negative Training Samples (Priority: P2)

The training dataset is typically imbalanced — most labeled pairs come from merge candidates (pairs close enough to be proposed), which biases toward the decision boundary. Pairs that are clearly different people (exemplar distance well above the candidate threshold) are never labeled because they're never shown. The user wants to automatically harvest a configurable number of these "obvious reject" pairs as negative training samples.

The ML Training tab gains a "Harvest negatives" button. When clicked, the system samples cluster pairs that were NOT candidates (exemplar dist > candidate_threshold) and adds them to the training DB as `label=0, source="auto_negative"`. The user controls the count (e.g., "sample up to N negative pairs") and can set a minimum exemplar distance floor to avoid borderline cases.

**Why this priority**: Improves training data balance without manual effort, but depends on having runs with embeddings loaded — slightly less standalone than US1-US3.

**Independent Test**: Load a run with 50+ clusters, click "Harvest negatives" with N=100, verify that ~100 new label=0 rows appear in the training DB, all with exemplar_dist > candidate_threshold.

**Acceptance Scenarios**:

1. **Given** a loaded run with embeddings, **When** the user clicks "Harvest negatives" with count=50 and min_dist=0.5, **Then** up to 50 cluster pairs with exemplar_dist >= 0.5 are sampled and saved to the training DB as `label=0, source="auto_negative"`.
2. **Given** auto-negative samples exist in the training DB, **When** the user trains a model, **Then** the auto-negative samples are included in training alongside human-labeled samples.
3. **Given** "Harvest negatives" is clicked, **When** the system generates pairs, **Then** it avoids pairs that already have a human label in the training DB (no overwriting).
4. **Given** the training data summary is viewed, **When** the user checks the breakdown, **Then** it shows counts by source: `human`, `auto_negative`.

---

### User Story 5 — Decision Boundary Visualization (Priority: P3)

For advanced users, provide a 2D projection (UMAP or PCA) of the feature space colored by model prediction (merge/reject) with the decision boundary overlaid. This helps visualize whether the model has clean separation or is struggling with overlapping clusters of samples.

**Why this priority**: Nice-to-have diagnostic for advanced users; lower priority since feature importance (US2) covers the primary interpretability need.

**Independent Test**: After training a model, click "Decision boundary" in the analysis section, verify a 2D scatter plot renders with points colored by predicted class and the threshold line visible.

**Acceptance Scenarios**:

1. **Given** a trained model and feature data, **When** the user clicks "Decision boundary plot", **Then** a 2D PCA projection scatter plot renders with points colored by predicted class (merge=green, reject=red).
2. **Given** the plot is shown, **When** the user hovers over a point, **Then** a tooltip shows the cluster pair IDs and ML probability.

---

### Edge Cases

- **CatBoost not installed**: If `catboost` is not in the environment, the "catboost" option should be disabled with a tooltip "Install catboost to enable" rather than raising an import error at train time.
- **Harvest negatives with very few clusters**: If fewer than 5 non-candidate pairs exist, show a warning rather than harvesting a biased micro-sample.
- **Manual merge of already-merged clusters**: If the user tries to merge two clusters that are already in the same merged group, show a warning and do nothing.
- **Feature importance with zero-variance features**: If a feature has zero variance across all training samples, it should be excluded from the importance chart (not shown as zero importance, which could confuse).
- **Model saved without feature_importances_**: Some exotic sklearn wrappers may not expose `feature_importances_`. Fall back to permutation importance or show a note.

## Requirements

### Functional Requirements

- **FR-001**: System MUST support training with model types: `decision_tree`, `random_forest`, `catboost` (in addition to existing `logistic_regression`, `xgboost`, `mlp`).
- **FR-002**: All new model types MUST produce probability estimates (`predict_proba`) for the ML merge mode to function.
- **FR-003**: System MUST display a ranked feature importance chart after training any model, using model-native importance (coefficients for LR, Gini/gain for tree-based, first-layer weight proxy for MLP).
- **FR-004**: System MUST provide a config-mapping table linking top features to their corresponding `PipelineConfig` / `ConservativeMergerConfig` fields and current default values.
- **FR-005**: System MUST allow manual merge of any two clusters regardless of whether they were merge candidates, and MUST show the same merge metrics as Merge Analysis pair cards (exemplar distance, support, 4-gate pass/fail, margin gap, post-merge diameter).
- **FR-006**: Manual merges MUST be recorded as human-labeled positive training samples (source="human", label=1).
- **FR-007**: System MUST support harvesting non-candidate cluster pairs as negative training samples with configurable count and minimum distance floor.
- **FR-008**: Auto-negative samples MUST be labeled `source="auto_negative"` and MUST NOT overwrite existing human labels.
- **FR-009**: `compute_pair_feature_contributions()` MUST support all new model types using the global importance proxy path.
- **FR-010**: CatBoost support MUST degrade gracefully if the `catboost` package is not installed.

### Key Entities

- **TrainConfig.model_type**: Extended enum — adds `"decision_tree"`, `"random_forest"`, `"catboost"` to existing values.
- **Feature-to-Config mapping**: Static lookup table mapping feature names (e.g., `min_exemplar_dist`) to config fields (e.g., `PipelineConfig.merge_candidate_threshold`) with current defaults.
- **Auto-negative training sample**: A training DB row with `source="auto_negative"`, `label=0`, generated from non-candidate cluster pairs.
- **Manual merge step**: A session step with `action="manual_merge"`, recording the two cluster IDs and the merge result.

## Success Criteria

### Measurable Outcomes

- **SC-001**: User can train and save all 6 model types (LR, XGBoost, MLP, decision tree, random forest, CatBoost) and use any of them in ML merge mode.
- **SC-002**: Feature importance chart renders within 1 second after training completes for any model type.
- **SC-003**: Config mapping table covers at least the top-10 features with corresponding config fields identified.
- **SC-004**: Manual merge workflow (select two clusters → preview → confirm) completes in under 30 seconds.
- **SC-005**: Harvesting 200 negative samples from a 100-cluster run completes in under 5 seconds.
- **SC-006**: After harvesting negatives, training data class balance improves (measured as ratio of label=0 to label=1 moving closer to 1.0).

## Assumptions

- CatBoost is an optional dependency — the system works without it, just with reduced model options.
- Feature importance for tree-based models uses native `feature_importances_` (Gini/gain), which is fast and does not require permutation importance.
- The feature-to-config mapping is maintained as a static dict — it does not auto-discover config fields from feature names.
- Manual merge uses the same `save_manual_merge_snapshot` infrastructure already in place, just exposed through a different UI path.
- Auto-negative pair sampling is uniform random from non-candidate pairs; stratified sampling by distance bucket is a potential future enhancement.
