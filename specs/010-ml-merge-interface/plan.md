# Implementation Plan: ML Merge Interface

**Feature**: `010-ml-merge-interface`
**Date**: 2026-04-21
**Spec**: [spec.md](spec.md)

---

## Summary

Introduce a three-state merge decision model (Approve / Reject / Undecided) into the existing Merge Analysis tab, then layer ML-model-as-first-guess on top. Phase 1 removes auto-pre-fill and adds explicit Smart Approve/Smart Reject buttons. Phase 2 adds a mode selector where the user picks a trained ML model, computes probabilities for all candidate pairs, and uses ML predictions as the initial state — leaving borderline cases undecided for human review.

All changes are confined to `face_cluster/analysis_views.py` (two new functions + two field additions) and `app/face_clustering.py` (UI logic changes). No new modules, no DB schema changes.

---

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Streamlit (UI), scikit-learn / XGBoost (ML), NumPy, pandas (features), plotly (histogram)
**Storage**: Session state (in-memory), `~/.sim_bench/sim_bench.db` (training data, model registry — unchanged schema)
**Testing**: pytest, `streamlit.testing.v1.AppTest`
**Target Platform**: Windows 10, Streamlit 1.x
**Performance Goals**: Threshold re-application in < 2s for 500 candidate pairs; ML predict in < 5s for 500 pairs
**Constraints**: No new module files; no DB schema changes; no blocking the render thread

---

## Constitution Check

*(No constitution.md found — check skipped. Validating against CLAUDE.md rules manually.)*

| Rule | Status | Notes |
|---|---|---|
| No algorithm code in `app/` | PASS | New `compute_ml_merge_view()` and `compute_pair_feature_contributions()` go in `face_cluster/analysis_views.py` |
| No blocking UI on render thread | PASS | ML prediction uses `_AsyncState` pattern |
| Tests must use production-default config | PASS | Test fixtures use default `PipelineConfig` values |
| File format contracts tested end-to-end | PASS | Training data save discipline contract has explicit tests |
| No test uses real album path | PASS | All tests use `tmp_path` or synthetic fixtures |
| Windows-safe output | PASS | No Unicode in new CLI/log output |

---

## Project Structure

### Documentation (this feature)

```
specs/010-ml-merge-interface/
├── spec.md
├── plan.md                   (this file)
├── research.md
├── data-model.md
├── contracts/
│   ├── analysis-views-api.md
│   └── training-data-discipline.md
└── tasks.md                  (next step)
```

### Source Code Changes

```
face_cluster/
└── analysis_views.py         # +2 new functions, +2 new fields on existing dataclasses

app/
└── face_clustering.py        # UI logic: mode selector, three-state, ML overview panel

tests/face_clustering/
├── test_merge_analysis.py    # +tests for compute_ml_merge_view, three-state filtering
└── test_streamlit_app.py     # +tests for mode selector, undecided default, source tracking
```

---

## Phase 1: Three-State Decision Refactor (US1)

Standalone — delivers value without any ML model.

### 1.1 Remove auto-pre-fill on load

**File**: `app/face_clustering.py`, lines ~2505-2518

Remove the block that pre-fills `merge_approval_decisions` with auto_approve → "approve" and auto_reject → "reject" on first tab load.

After removal, all pairs start absent from `merge_approval_decisions` (= Undecided).

### 1.2 Add `merge_decision_sources` session state

**File**: `app/face_clustering.py`

Initialize `st.session_state.merge_decision_sources = {}` alongside `merge_approval_decisions`.

Update every write to `merge_approval_decisions` to also write the corresponding source:
- User-initiated writes (button clicks): source = "human"
- Smart Approve/Smart Reject: source = "human" (explicit user action)
- ML pre-fill (Phase 2): source = "ml"

### 1.3 Split Smart Approve → Smart Approve + Smart Reject

**File**: `app/face_clustering.py`, `_render_approval_controls()`

Replace single "Smart Approve" button with two:
- **Smart Approve** (green): sets only `auto_approve`-confidence groups to "approve" (source="human"). Leaves all others unchanged.
- **Smart Reject** (red/secondary): sets only `auto_reject`-confidence groups to "reject" (source="human"). Leaves all others unchanged.

Remove the `st.session_state.merge_group_filter = "Review Only"` side-effect from Smart Approve — that behaviour was tied to the old auto-pre-fill assumption.

### 1.4 Three-state summary bar

**File**: `app/face_clustering.py`

Add a persistent summary metric row above the gallery:

```python
total = len(all_candidate_pairs)
n_approved = sum(1 for v in decisions.values() if v == "approve")
n_rejected  = sum(1 for v in decisions.values() if v == "reject")
n_undecided = total - n_approved - n_rejected

col1, col2, col3 = st.columns(3)
col1.metric("Approved", n_approved)
col2.metric("Rejected",  n_rejected)
col3.metric("Undecided", n_undecided)
```

### 1.5 Per-card state display (three states)

**File**: `app/face_clustering.py`, `_render_grouped_merge_gallery()` and `_render_flat_merge_gallery()`

Group/pair cards render the current decision state:
- Undecided: grey neutral badge (no colour)
- Approved: green badge, possibly with "ML" indicator if source="ml"
- Rejected: red badge, possibly with "ML" indicator if source="ml"

Radio buttons or Approve/Reject/Clear buttons allow the user to set or clear decisions.

### 1.6 Gate "Apply + Remerge" on ≥1 approved pair

**File**: `app/face_clustering.py`, `_render_approval_controls()`

Disable "Apply + Remerge" button and show info message "Approve at least one pair to merge." when `n_approved == 0`.

Show info when many pairs are undecided: "N pairs undecided — they will be re-evaluated in the next round."

### 1.7 Training data discipline

**File**: `app/face_clustering.py`, `_collect_all_merge_labels()`

Add `sources: Dict` parameter. Filter: only write to training DB when `sources.get(pair_key) == "human"`. Update all call sites to pass `st.session_state.merge_decision_sources`.

### 1.8 Carry-forward undecided pairs (no code change needed)

Undecided pairs are absent from `pending_labels.json` → not merged → their clusters survive → remerge re-evaluates them → they appear in the next round's candidates. This is already the correct behaviour (absent = not in approved list = not merged). Verified by test.

---

## Phase 2: ML Mode Integration (US2–US6)

Builds on Phase 1. Requires at least one saved ML model.

### 2.1 New backend functions

**File**: `face_cluster/analysis_views.py`

#### `compute_ml_merge_view(result, model_payload, threshold, candidate_threshold)`

Steps:
1. Build `MergeFeatureContext` from `result.cluster_result`, `result.faces`, `result.distance_matrix` (load from disk if None).
2. `FeatureComputer().compute_all_pairs(ctx, candidate_threshold)` → `pair_features: Dict[Tuple[int,int], ClusterPairFeatures]`
3. `FeatureComputer().to_dataframe(pair_features)` → `df`
4. Validate `model_payload["metadata"]["feature_version"] == FeatureComputer.VERSION` → raise ValueError if mismatch.
5. `MergeTrainer().predict(model_payload, df)` → `df` with `ml_prob`, `ml_pred` columns
6. For each pair: build `MergeDecisionRow` with `action="proposed_merge"` if ml_pred==1 else `"proposed_reject"`, `ml_prob=row.ml_prob`, `ml_pred=row.ml_pred`. Populate `exemplar_face_ids_a/b` from `result.cluster_result`.
7. Map `ml_prob` to gate-count equivalent (research.md Decision 1).
8. Call `group_merge_candidates(candidate_pairs, gate_count_equiv)` → `CandidateGroup` list.
9. Build `MergeAnalysisView` with `merges` (ml_pred==1), `rejections` (ml_pred==0), `merge_groups`, `ml_threshold=threshold`, `near_misses` (0.4 < prob < 0.6).
10. Return view.

#### `compute_pair_feature_contributions(pair_features_dict, model_payload, top_n=3)`

For logistic regression:
```python
scaled = scaler.transform([feature_values])
contributions = coef[0] * scaled[0]
```
For XGBoost/MLP: use `model_payload["metadata"]["feature_importance"]` as proxy.

Return sorted list of `(name, value, direction_str)`.

### 2.2 UI: Mode selector

**File**: `app/face_clustering.py`, `render_merge_analysis_tab()` top section

```python
mode = st.radio(
    "Merge source",
    options=["Heuristic (4-gate)", "ML Model"],
    horizontal=True,
    key="merge_mode_selector",
)
```

When ML Model selected:
- Dropdown: `st.selectbox("Model", model_options, key="ml_selected_model")`
- Threshold slider: `st.slider("Merge threshold", 0.1, 0.9, 0.5, 0.05, key="ml_threshold")`
- Optional: "Wider candidates" checkbox → sets `ml_candidate_threshold` to 0.65 if checked
- Button: "Apply threshold" → triggers async prediction worker

When Heuristic selected: existing `MergeAnalysisView.compute()` path (unchanged except auto-pre-fill removal).

### 2.3 UI: Async ML prediction worker

**File**: `app/face_clustering.py`

```python
def _run_ml_predict():
    payload = MergeTrainer().load_model(model_path)
    return compute_ml_merge_view(result, payload, threshold, candidate_threshold)

worker = _AsyncState()
st.session_state.ml_predict_worker = worker
worker.start(_run_ml_predict)
st.rerun()
```

On completion: store result in `st.session_state.merge_analysis_view`, apply ML pre-fills to `merge_approval_decisions` + `merge_decision_sources`.

### 2.4 UI: ML pre-fill logic

Once prediction completes, apply initial decisions:

```python
for pair_key, row in ml_predictions.items():
    prob = row["ml_prob"]
    if prob >= threshold:
        decisions[pair_key] = "approve"
        sources[pair_key] = "ml"
    elif prob < (1.0 - threshold):
        decisions[pair_key] = "reject"
        sources[pair_key] = "ml"
    # else: leave absent (Undecided)
```

Applied once. If user subsequently changes a decision: source becomes "human", ML badge removed.

### 2.5 UI: ML overview panel

**File**: `app/face_clustering.py`, new `_render_ml_overview_panel(view, threshold)`

```
Model: lr_austria_v3  |  AUC: 0.94  |  F1: 0.88  |  N: 128 training samples
48 candidate pairs at threshold 0.50
[probability histogram: plotly bar chart, bins of 0.1]
Merge: 31  |  Reject: 17  |  Borderline: 5
Groups: 8 high-confidence | 3 review | 2 low-confidence
```

If distribution entropy is high (flat histogram): show warning "Model shows low confidence separation — consider retraining."

### 2.6 UI: Gallery cards in ML mode

Modify `_render_grouped_merge_gallery()` to branch on `merge_mode`:

**ML mode card header**:
- Replaces gate count badge with probability badge: colour-coded green (>=0.8), amber (0.5-0.8), red (<0.5)
- Shows `[MERGE: 92%]` or `[REJECT: 12%]`

**ML mode card detail section** (in expander):
- Per-pair: `prob=0.92  (threshold=0.50)` instead of gate pass/fail row
- Feature contributions: top 3 from `compute_pair_feature_contributions()`
- Heuristic gates (read-only): collapsible sub-section "Heuristic reference: 3/4 gates"

### 2.7 UI: Comparison mode (US5)

**File**: `app/face_clustering.py`

Add checkbox: "Show heuristic comparison" (only visible when merge_mode == "ml_model" AND heuristic merge_log exists in result).

When enabled:
- Each pair shows both decisions in the card header
- Agreement indicator: "✓ Both merge" / "✓ Both reject" / "⚠ ML: merge, Heuristic: reject" / "⚠ ML: reject, Heuristic: merge"
- Add "Disagreements only" filter option

### 2.8 ML Training tab: "Open in Merge Analysis" button

**File**: `app/face_clustering.py`, `render_ml_training_tab()` Section 7

Replace the current flat table display with:

```python
if st.button("Open in Merge Analysis tab"):
    st.session_state.merge_mode_selector = "ML Model"
    st.session_state.ml_selected_model = sel_model_name
    # Switch to Merge Analysis tab (tab index)
    st.session_state.active_tab = <merge_analysis_tab_index>
    st.rerun()
```

Keep a minimal summary table as a preview (top 5 rows by ml_prob), but the primary action is "Open in Merge Analysis".

---

## Test Plan

### New unit tests in `test_merge_analysis.py`

| Test | What it verifies |
|---|---|
| `ut_MLMergeView::test_compute_returns_merge_analysis_view` | `compute_ml_merge_view` returns a `MergeAnalysisView` with ml_prob populated |
| `ut_MLMergeView::test_high_prob_pairs_in_merges_list` | Pairs with prob >= threshold appear in `view.merges` |
| `ut_MLMergeView::test_low_prob_pairs_in_rejections_list` | Pairs with prob < (1-threshold) appear in `view.rejections` |
| `ut_MLMergeView::test_borderline_pairs_in_near_misses` | Pairs with 0.4 < prob < 0.6 appear in `view.near_misses` |
| `ut_MLMergeView::test_feature_version_mismatch_raises` | `ValueError` on version mismatch |
| `ut_MLMergeView::test_probability_to_gate_mapping` | Prob 0.85 → gate count 4; prob 0.25 → gate count 0 |
| `ut_FeatureContributions::test_logistic_regression_exact` | LR contributions sum to expected log-odds shift |
| `ut_FeatureContributions::test_xgboost_proxy_returns_top3` | XGBoost proxy returns 3 entries with direction |
| `ut_ThreeState::test_undecided_excluded_from_training_data` | `_collect_all_merge_labels` excludes undecided and ML-source pairs |
| `ut_ThreeState::test_human_override_included_in_training_data` | Human-overridden ML suggestion → saved as training sample |
| `ut_ThreeState::test_ml_suggested_approve_not_in_training_data` | ML-source approve, untouched → excluded from training |

### New app tests in `test_streamlit_app.py`

| Test | What it verifies |
|---|---|
| `test_merge_analysis_starts_all_undecided` | After loading a run, `merge_approval_decisions` is empty |
| `test_smart_approve_only_sets_high_confidence` | Smart Approve doesn't change review/auto_reject groups |
| `test_smart_reject_only_sets_low_confidence` | Smart Reject doesn't change review/auto_approve groups |
| `test_apply_button_disabled_with_no_approvals` | "Apply + Remerge" disabled when `n_approved == 0` |
| `test_mode_selector_shows_model_dropdown_in_ml_mode` | ML mode selector reveals model dropdown |

---

## Risks

| Risk | Mitigation |
|---|---|
| `distance_matrix` not loaded in `PipelineResult` (large albums may skip it) | `compute_ml_merge_view` loads from `{output_dir}/distance_matrix.npy` if `result.distance_matrix is None` — same pattern as existing analysis views |
| Feature computation slow for large albums (100+ clusters, 500+ pairs) | Async `_AsyncState` pattern; show spinner with elapsed time |
| ML model not calibrated (overconfident near 0/1) | Histogram with bimodal-check warning; user always sees and can override |
| Threshold change re-triggers full prediction | Cache predictions per (model, candidate_threshold); threshold change only re-applies the decision pre-fill locally without re-running the model |
| Wider candidates produce too many pairs | Bounded at 0.65; show count before applying: "This will evaluate N additional pairs" |
