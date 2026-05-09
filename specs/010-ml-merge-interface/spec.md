# Feature Specification: ML Merge Interface

**Feature**: `010-ml-merge-interface`
**Created**: 2026-04-21
**Status**: In Progress

## Overview

Replace the flat-table "Apply to Current Run" in the ML Training tab with a full visual merge interface integrated into the existing Merge Analysis tab. The ML model becomes an alternative merge proposer — predicting approve/reject for every candidate pair — shown through the same grouped gallery with face crops, confidence tiers, and Apply + Remerge workflow. Simultaneously, introduce a three-state decision model (Approve / Reject / Undecided) that improves both the heuristic and ML workflows.

---

## User Scenarios & Testing

### User Story 1 — Three-State Merge Decisions (Priority: P1)

A user opens the Merge Analysis tab to review merge candidates. Instead of seeing pre-filled approve/reject decisions (which they never verified), all candidates start as **Undecided**. The user explicitly approves or rejects pairs they've reviewed. Undecided pairs are not merged and not saved as training data — they carry forward to the next remerge round for re-evaluation.

**Why this priority**: Foundational change that improves data quality for both heuristic and ML workflows. Without this, ML-as-first-guess would inherit the training data pollution problem from heuristic pre-filling.

**Independent Test**: Can be tested entirely within the existing heuristic workflow (no ML model needed). Load a run, verify all pairs start undecided, approve some, apply + remerge, verify undecided pairs reappear in the next round.

**Acceptance Scenarios**:

1. **Given** a loaded run with merge candidates, **When** the user opens the Merge Analysis tab, **Then** all candidate pairs/groups show "Undecided" state (no pre-filled approve/reject).

2. **Given** a pair in "Undecided" state, **When** the user clicks "Approve" or "Reject", **Then** the pair's state changes to the selected decision and the summary bar updates (e.g., "5 approved | 2 rejected | 41 undecided").

3. **Given** 10 approved, 3 rejected, and 35 undecided pairs, **When** the user clicks "Apply + Remerge", **Then** only the 10 approved pairs are merged. The 3 rejected pairs are recorded as explicit rejections (not shown again). The 35 undecided pairs are re-evaluated in the next round with updated evidence.

4. **Given** the user clicks "Apply + Remerge", **When** training data is saved, **Then** only the 10 approved and 3 rejected pairs are written to `merge_training_data` (label=1 and label=0 respectively). The 35 undecided pairs are NOT saved as training samples.

5. **Given** merge candidates exist, **When** the user clicks "Smart Approve", **Then** only high-confidence groups (all pairs with 4/4 gates in heuristic mode) are set to "Approve". All other groups remain in their current state (undecided or previously set).

6. **Given** merge candidates exist, **When** the user clicks "Smart Reject", **Then** only low-confidence groups (all pairs with <= 1 gate in heuristic mode) are set to "Reject". All other groups remain unchanged.

---

### User Story 2 — ML Model as Merge Proposer (Priority: P1)

A user has a trained ML merge model and wants to apply it to a run's merge candidates. They select the ML model in the Merge Analysis tab. The model scores every candidate pair with a probability. The gallery shows ML predictions with face crops, confidence badges, and the same approve/reject/undecided workflow. The ML model serves as the "first guess" — pre-filling high-confidence predictions while leaving borderline cases undecided for human review.

**Why this priority**: This is the core value proposition — reducing manual merge review from 200+ pairs to ~10 borderline cases by leveraging the trained model.

**Independent Test**: Load a run, select a trained model, verify predictions appear with probability scores. Override one prediction, apply + remerge, verify overrides are saved as training data but untouched ML suggestions are not.

**Acceptance Scenarios**:

1. **Given** a loaded run and at least one saved ML model, **When** the user switches the merge source to "ML Model" and selects a model, **Then** features are computed for all candidate pairs and the model predicts merge probability for each pair.

2. **Given** ML predictions are computed, **When** the gallery renders, **Then** each pair/group shows: ML probability badge (color-coded), face crops for both clusters, per-pair probability, and the heuristic's 4-gate status (read-only, for reference).

3. **Given** a probability threshold slider (default 0.5), **When** the user adjusts the threshold and clicks "Apply threshold", **Then** pairs are re-classified: prob >= threshold → pre-filled as "Approve (ML)", prob < (1 - threshold) → pre-filled as "Reject (ML)", otherwise → "Undecided". Group counts update accordingly.

4. **Given** ML pre-filled decisions, **When** the user changes an ML-suggested decision (e.g., overrides "Approve ML" to "Reject"), **Then** the pair is now marked as a human decision (ML badge removed). When saved as training data, this override IS included as a training sample.

5. **Given** ML pre-filled decisions, **When** the user clicks "Apply + Remerge" without reviewing some ML suggestions, **Then** ML-suggested "Approve" pairs ARE merged (trusted by default). ML-suggested "Reject" pairs are NOT merged. But neither untouched ML suggestion is saved as training data — only human-touched decisions are saved.

6. **Given** a model trained on feature version 3, **When** the current run has feature version 3, **Then** prediction proceeds normally. **Given** a feature version mismatch, **Then** the system shows an error and blocks prediction.

---

### User Story 3 — ML Probability Overview and Threshold Tuning (Priority: P2)

A user wants to understand the ML model's confidence distribution across all candidates before reviewing individual pairs. They see a probability histogram showing how many pairs fall into high-confidence merge, high-confidence reject, and borderline zones. A threshold slider lets them tune the merge cutoff, with live count updates.

**Why this priority**: Enhances user confidence in the ML model and enables threshold tuning without trial-and-error on individual pairs.

**Independent Test**: Load a run, select ML model, verify histogram renders and threshold slider updates group counts.

**Acceptance Scenarios**:

1. **Given** ML predictions are computed, **When** the overview panel renders, **Then** it shows: model name, training metrics (AUC, F1), candidate count, a probability histogram, and a summary line ("Merge: N | Reject: M | Borderline: K").

2. **Given** the threshold slider is set to 0.5, **When** the user drags it to 0.7 and clicks "Apply threshold", **Then** fewer pairs are pre-filled as "Approve" (only those with prob >= 0.7), more pairs become "Undecided", and the group counts in the summary update.

3. **Given** a bimodal probability distribution (clear merge vs reject), **Then** the histogram shows two peaks with a valley near the threshold. **Given** a flat/unimodal distribution, **Then** a warning badge appears: "Model shows low separation — consider retraining with more data."

---

### User Story 4 — Per-Pair Feature Contributions (Priority: P2)

A user reviewing an ML prediction wants to understand WHY the model predicted merge or reject for a specific pair. Expanding a pair's detail section shows the top contributing features with their values and direction.

**Why this priority**: Builds trust in ML decisions and helps users make informed overrides.

**Independent Test**: Expand a pair detail, verify top-3 features are shown with values and direction.

**Acceptance Scenarios**:

1. **Given** a pair with ML prediction, **When** the user expands its detail section, **Then** the top 3 contributing features are shown with: feature name, value, and direction arrow (e.g., `min_exemplar_dist = 0.22 -> merge`, `size_ratio = 8.5 -> reject`).

2. **Given** a logistic regression model, **When** feature contributions are computed, **Then** contribution = coefficient * scaled_feature_value. For XGBoost/MLP, use global feature importance as a proxy.

3. **Given** both ML and heuristic data, **When** the detail section renders, **Then** the 4-gate heuristic status is shown below the ML features as read-only context (e.g., "Heuristic: 3/4 gates pass — margin gate failed").

---

### User Story 5 — Heuristic vs ML Comparison (Priority: P3)

A user wants to see where the ML model and heuristic disagree to understand if the model is better or worse at borderline cases.

**Why this priority**: Useful for model evaluation and debugging but not required for the core merge workflow.

**Independent Test**: Toggle comparison mode, verify disagreement rows are highlighted with both decisions visible.

**Acceptance Scenarios**:

1. **Given** both ML and heuristic decisions, **When** the user enables "Show heuristic comparison", **Then** each pair shows both decisions and an agreement indicator (Agree / ML-only merge / Heuristic-only merge / Both reject).

2. **Given** disagreements exist, **When** the user filters to "Disagreements only", **Then** only pairs where ML and heuristic disagree are shown, sorted by ML probability (most uncertain first).

---

### User Story 6 — Wider Candidate Discovery (Priority: P3)

A user suspects the heuristic's candidate threshold (0.45) is too restrictive and some valid merges are being missed. They enable "Wider candidates" to let the ML model evaluate pairs up to a higher distance threshold.

**Why this priority**: Advanced feature for power users; the default threshold works for most cases.

**Independent Test**: Enable wider candidates, verify additional pairs appear that weren't in the default candidate list.

**Acceptance Scenarios**:

1. **Given** the default merge_candidate_threshold is 0.45, **When** the user enables "Wider candidates" (threshold raised to e.g. 0.60), **Then** additional candidate pairs are discovered and scored by the ML model.

2. **Given** wider candidates are enabled, **When** the gallery renders, **Then** pairs beyond the default threshold are visually marked as "Extended range" so the user knows these wouldn't have been proposed by the heuristic.

---

### Edge Cases

- **No ML model saved**: ML mode selector is disabled with message "Train and save a model in the ML Training tab first."
- **Feature version mismatch**: Model trained on V3 features but current FeatureComputer is V4 → block prediction, show error with model's version and current version.
- **All pairs high-confidence**: If model predicts all pairs > 0.9, show info "Model is very confident — consider spot-checking a few predictions before applying."
- **All pairs undecided at Apply time**: "Apply + Remerge" button disabled with message "Approve at least one pair to merge."
- **Model predicts 0 merges**: Show info "Model recommends no merges at this threshold. Try lowering the threshold or reviewing undecided pairs."
- **Empty merge_log**: Run has no merge stage → show "This run has no merge data. Run the pipeline with merge_enabled=True."
- **Remerge produces same candidates**: After Apply + Remerge, if the same undecided pairs reappear with identical evidence, show info "No new evidence — remaining candidates are stable."

---

## Requirements

### Functional Requirements

**Three-State Decisions (US1):**
- **FR-001**: All merge candidate pairs MUST start in "Undecided" state when the Merge Analysis tab loads. No auto-pre-filling on load.
- **FR-002**: Each pair/group MUST support three explicit states: Approve (green), Reject (red), Undecided (grey).
- **FR-003**: The summary bar MUST show counts for all three states at all times: "N approved | M rejected | K undecided".
- **FR-004**: "Smart Approve" MUST only set high-confidence groups to Approve. It MUST NOT change Reject or Undecided states of other groups.
- **FR-005**: "Smart Reject" MUST be a separate button that only sets low-confidence groups to Reject. It MUST NOT change Approve or Undecided states of other groups.
- **FR-006**: "Apply + Remerge" MUST only merge Approved pairs. Undecided and Rejected pairs MUST NOT be merged.
- **FR-007**: Training data MUST only include pairs with explicit human decisions (Approve or Reject). Undecided pairs MUST be excluded from training data.
- **FR-008**: Undecided pairs MUST carry forward across remerge rounds: if their clusters still exist as candidates after merging, they reappear for review. Rejected pairs MUST NOT reappear.

**ML Mode (US2):**
- **FR-009**: The Merge Analysis tab MUST have a mode selector: "Heuristic (4-gate)" or "ML Model".
- **FR-010**: When ML mode is selected, the user MUST be able to choose from saved models via a dropdown.
- **FR-011**: The system MUST compute features for all candidate pairs using `FeatureComputer` and run `MergeTrainer.predict()` to get probabilities.
- **FR-012**: ML predictions MUST pre-fill decisions based on the probability threshold: prob >= threshold → "Approve (ML)", prob < (1 - threshold) → "Reject (ML)", otherwise → "Undecided".
- **FR-013**: ML-suggested decisions MUST be visually distinct from human decisions (e.g., "ML" badge).
- **FR-014**: When the user changes an ML-suggested decision, it MUST become a human decision (ML badge removed).
- **FR-015**: ML-suggested "Approve" pairs that the user did not review MUST be merged when "Apply + Remerge" is clicked (trusted execution).
- **FR-016**: ML-suggested decisions that the user did not explicitly confirm or change MUST NOT be saved as training data.
- **FR-017**: Human overrides of ML suggestions MUST be saved as training data (highest-value signal).

**Threshold and Overview (US3):**
- **FR-018**: A probability threshold slider (range 0.1–0.9, default 0.5) MUST be shown in ML mode with an "Apply threshold" button to recompute pre-fills.
- **FR-019**: A probability distribution panel MUST show: model name, key metrics (AUC, F1), candidate count, histogram of probabilities, and merge/reject/borderline counts.
- **FR-020**: A warning MUST be shown when the probability distribution indicates poor model separation.

**Feature Contributions (US4):**
- **FR-021**: In ML mode, each pair's expandable detail section MUST show the top 3 contributing features with name, value, and direction.
- **FR-022**: The 4-gate heuristic status MUST be shown as read-only context in ML mode detail sections.

**Comparison (US5):**
- **FR-023**: A "Show heuristic comparison" toggle MUST show both ML and heuristic decisions per pair with an agreement indicator.
- **FR-024**: A "Disagreements only" filter MUST be available when comparison mode is active.

**Wider Candidates (US6):**
- **FR-025**: A "Wider candidates" toggle MUST allow raising the candidate threshold beyond the default (e.g., 0.45 → 0.60).
- **FR-026**: Extended-range pairs MUST be visually distinguished from default-range pairs.

### Key Entities

- **MergeDecision**: A candidate pair's state — one of Approve, Reject, Undecided — with source indicator (Human or ML).
- **MLPrediction**: Model output for a candidate pair — probability, predicted class, top feature contributions.
- **MergeMode**: The active merge proposer — Heuristic (4-gate ConservativeMerger) or ML Model (trained classifier).

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: All merge candidate pairs start in Undecided state on tab load — zero pre-filled decisions.
- **SC-002**: Training data contains only explicit human decisions — no auto-filled or ML-suggested labels saved without user interaction.
- **SC-003**: When an ML model is applied, the user can complete merge review for 200+ candidate pairs in under 5 minutes (vs 30+ minutes with pair-by-pair heuristic review), by reviewing only borderline cases (~10-20 pairs).
- **SC-004**: ML predictions are shown with face crops in the same gallery format as heuristic merge analysis — no flat tables.
- **SC-005**: The threshold slider updates group counts within 2 seconds of clicking "Apply threshold" for datasets up to 500 candidate pairs.
- **SC-006**: Undecided pairs from round N reappear in round N+1 after remerge (carry-forward works).
- **SC-007**: Human overrides of ML suggestions are saved as training data and can be used for model retraining.

---

## Assumptions

- At least one ML model has been trained and saved via the ML Training tab before ML mode can be used.
- The existing `FeatureComputer` (version 3) and `MergeTrainer` classes provide the necessary compute and predict APIs — no new ML infrastructure is needed.
- The session chain model (spec 009) will record merge step metadata including mode (heuristic/ML), model name, and threshold.
- The existing `group_merge_candidates()` function can be reused for ML-mode grouping by passing probability-derived gate counts.
- Feature contribution display uses model coefficients (logistic regression) or global feature importance (XGBoost/MLP) — not per-prediction SHAP values (too expensive for Streamlit).
- The "Wider candidates" threshold is bounded (max 0.70) to prevent evaluating obviously irrelevant pairs.
