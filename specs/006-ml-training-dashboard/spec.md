# Feature Specification: Labeling Review + ML Training Dashboard

**Feature**: 006-ml-training-dashboard
**Created**: 2026-04-17
**Status**: Draft

## Overview

Two new sections in the face clustering app:
1. **Labeling Review** — audit labeled data quality, spot disagreements, edit labels, get labeling suggestions
2. **ML Training** — configure, train, and evaluate merge classifiers from the UI

Both read from the central `merge_training_data` table in `~/.sim_bench/sim_bench.db`.

---

## Part A: Labeling Review

### User Story A1 — Audit label quality (P1)

A user has labeled merge decisions across several runs. Before training, they want to check:
are any labels suspicious? Did I reject a pair that the heuristic strongly approved (all 4 gates passed), or approve one it strongly rejected?

**Acceptance Scenarios:**

1. **Given** labeled data exists, **When** user opens the Labeling Review section, **Then** a table shows every labeled pair with: label, n_gates_passed, exemplar_dist, thumbnails, and a "heuristic agreed" flag.
2. **Given** a pair where human label disagrees with heuristic (>= 3 gates passed but rejected, or 0 gates but approved), **Then** row is highlighted as "disagreement".
3. **Given** the user clicks "Flip label" on a pair, **Then** the label is toggled (approve <-> reject) and saved to the DB immediately.

### User Story A2 — Active labeling suggestions (P2)

A user wants to maximize label value — the system should suggest which unlabeled pairs to label next (borderline cases are most informative).

**Acceptance Scenarios:**

1. **Given** unlabeled pairs exist in the DB, **When** user opens "Suggested next labels", **Then** pairs are ranked by uncertainty (closest to the decision boundary — e.g., 2/4 gates passed, or exemplar_dist near the merge threshold).

### Edge Cases
- No labeled data: show "No labels yet" message.
- Labels from different feature versions: show warning badge, still display.
- Conflicting labels for the same identity pair across runs: flag as "Conflict".

---

## Part B: ML Training Dashboard

### User Story B1 — Train a classifier from the UI (P1)

A data scientist has collected 100+ labeled pairs. They want to train a Logistic Regression model, see results, and save it — without writing code or running scripts.

**Acceptance Scenarios:**

1. **Given** the user selects runs and split percentages, **When** they click "Train", **Then** a model trains in the background with progress feedback.
2. **Given** training completes, **Then** the user sees: accuracy, precision, recall, F1, AUC, confusion matrix, ROC curve, and feature importance.
3. **Given** the user clicks "Save Model", **Then** the model + scaler + metadata are persisted and appear in the model history.

### User Story B2 — Compare models (P2)

After training several models (different model types, features, data), the user wants to compare.

**Acceptance Scenarios:**

1. **Given** 2+ saved models, **When** user opens "Compare", **Then** a side-by-side table shows key metrics for each model.

### User Story B3 — Apply model to current run (P2)

The user has a trained model and wants to see what it predicts for the current run's merge candidates.

**Acceptance Scenarios:**

1. **Given** a trained model and a loaded run with merge candidates, **When** user clicks "Predict", **Then** each candidate pair gets a predicted label + confidence score, shown alongside the heuristic decision.

### Edge Cases
- Fewer than 30 labeled samples: block training with clear message.
- Extreme class imbalance (> 10:1): show warning, still allow training.
- Feature mismatch (model trained on V3 features, current run has V2): show error, don't predict.
- XGBoost not installed: disable that option with install hint.

---

## Requirements

### Functional Requirements

**Labeling Review:**
- **FR-A01**: Show all labeled pairs with: label, heuristic decision (n_gates_passed), exemplar_dist, exemplar thumbnails.
- **FR-A02**: Highlight disagreements between human label and heuristic (>= 3 gates but rejected, or <= 1 gate but approved).
- **FR-A03**: Allow inline label flip (approve <-> reject) with immediate DB write.
- **FR-A04**: Show active labeling suggestions ranked by uncertainty (distance to decision boundary).
- **FR-A05**: Filter by run, by label, by agreement status.

**ML Training:**
- **FR-B01**: Dataset panel: select runs (multiselect), train/val/test split sliders (%), split strategy (random-stratified vs. by-run).
- **FR-B02**: Model panel: select model type (Logistic Regression, XGBoost, MLP), per-model hyperparameters.
- **FR-B03**: Feature panel: checkboxes per feature group (A, B+C, D, G), show selected feature count.
- **FR-B04**: Training runs async (background thread, `_AsyncState` pattern). Shows progress.
- **FR-B05**: Results panel: metrics table, confusion matrix heatmap, ROC curve, feature importance bar chart.
- **FR-B06**: Save model: persists pickle + metadata to `~/.sim_bench/models/`. Records in `trained_models` DB table.
- **FR-B07**: Model history: table of all saved models with metrics, load button, delete button.
- **FR-B08**: Compare: select 2+ models, show metrics side-by-side.
- **FR-B09**: Predict: apply a saved model to the current run's candidates, show predicted label + probability.

### Non-Functional
- Training must not block the Streamlit render thread (async pattern).
- Minimum 30 labeled samples required to start training.
- Model files are self-contained (pickle with model + scaler + feature_names + metadata).

---

## Key Entities

- **`TrainedModel`**: pickle file containing `{model, scaler, feature_names, metadata}`.
- **`trained_models` DB table**: `id, name, model_type, model_path, feature_version, n_train, n_test, accuracy, f1, auc, created_at, metadata_json`.
- **`MergeTrainer`**: class in `face_cluster/ml_trainer.py` — wraps sklearn/xgboost training, evaluation, prediction.

---

## Success Criteria

- **SC-001**: User can train a model, view results, and save it without touching the CLI.
- **SC-002**: Saved model can be loaded and applied to a new run's candidates from the UI.
- **SC-003**: Labeling review identifies all human-heuristic disagreements.
- **SC-004**: Training completes in < 10s for 500 samples (album-scale).
