# Research: ML Merge Interface (010)

**Feature**: `010-ml-merge-interface`
**Phase**: 0 — Resolve unknowns before design

---

## Decision 1: How to adapt `group_merge_candidates()` for ML mode

**Question**: The existing `group_merge_candidates(candidate_pairs, gate_counts, ...)` takes integer gate counts (0–4) per pair. ML mode produces continuous probabilities. Can we reuse the same grouping function without changes?

**Decision**: Map ML probability to a gate-count equivalent before passing to `group_merge_candidates()`.

```
prob >= 0.8  → 4  (auto_approve)
0.6 <= prob < 0.8 → 3  (review, cohesion-promoted if group avg >= 0.8)
0.4 <= prob < 0.6 → 2  (review)
0.3 <= prob < 0.4 → 1  (low)
prob < 0.3   → 0  (auto_reject)
```

**Rationale**: No changes to `group_merge_candidates()`. The 0–4 mapping is semantically equivalent (probability deciles map naturally to gate counts). The confidence tiers (auto_approve / review / auto_reject) emerge correctly from this mapping at the same thresholds used for gates.

**Alternatives considered**:
- New classification function that works directly on probabilities: rejected — duplicates logic already working well in `group_merge_candidates()`.
- Expose a new parameter to `group_merge_candidates()` accepting raw probabilities: rejected — unnecessary complexity, would break existing callers.

---

## Decision 2: Three-state session state representation

**Question**: How to represent Approve/Reject/Undecided in `st.session_state.merge_approval_decisions` without breaking existing usage?

**Decision**: Keep `merge_approval_decisions: Dict[Tuple[int,int], str]` with values `"approve"` | `"reject"`. Absence from the dict = Undecided. Add a **new** parallel dict `merge_decision_sources: Dict[Tuple[int,int], str]` with values `"human"` | `"ml"` to track decision provenance.

Training data filtering:
- Save to `merge_training_data` only pairs where `source == "human"`.
- ML-suggested pairs (source `"ml"`) that the user did not touch are excluded.

**Rationale**: Minimal change to existing code. `merge_approval_decisions` keeps its semantics (only contains explicit decisions). The new `merge_decision_sources` dict is additive. The "undecided" state is represented by absence, which is already how unreviewed groups work today.

**Alternatives considered**:
- Explicit "undecided" key in the dict: rejected — would require updating every caller that checks `decisions.get(key) == "approve"`.
- Full new dataclass per pair: rejected — over-engineered for what is essentially two parallel dicts.

---

## Decision 3: Feature contributions per prediction

**Question**: Per-pair feature contributions for transparency. SHAP values would be ideal but are computationally expensive. What proxy is practical in Streamlit?

**Decision**: Use coefficient-based contributions for logistic regression; global feature importance (from `TrainResult.feature_importance`) as a proxy for XGBoost/MLP.

For logistic regression (exact):
```python
scaled = scaler.transform([pair_feature_values])
contributions = model.coef_[0] * scaled[0]  # per-feature contribution to log-odds
top3 = sorted(zip(feature_names, contributions), key=lambda x: abs(x[1]), reverse=True)[:3]
```

For XGBoost/MLP (proxy — global importance, not per-prediction):
- Use `TrainResult.feature_importance` rank order.
- Display with caveat: "Importance shown is global, not per-prediction."

**Rationale**: Logistic regression supports exact per-prediction attribution at negligible cost. For tree/neural models, per-prediction SHAP is too slow for interactive Streamlit (0.1–1s per prediction call vs 10ms target). Global importance with a disclaimer is honest and useful.

**Alternatives considered**:
- Full SHAP integration: rejected for interactive use — computationally prohibitive in Streamlit render loop.
- Skip feature contributions for non-LR models: rejected — users with XGBoost models also deserve transparency.

---

## Decision 4: Wider candidates threshold bound

**Question**: When the user enables "Wider candidates", what upper bound prevents evaluating irrelevant pairs?

**Decision**: Cap at 0.65 (vs default 0.45). This adds ~50% more candidate distance range while remaining within a range where some true merges still exist. Beyond 0.65, cosine distances are high enough that clusters are clearly different identities in most ArcFace embedding spaces.

**Rationale**: ArcFace same-person pairs typically cluster below 0.45. Between 0.45–0.65, some valid merges exist when image quality or pose variation is high. Above 0.65, pairs are almost exclusively different identities — evaluating them wastes ML inference time and floods the user with low-value candidates.

**Alternatives considered**:
- Unlimited: rejected — would produce hundreds of irrelevant candidates on large albums.
- Fixed at 0.55: rejected — may be too conservative for some datasets.
- User-configurable up to 0.80: rejected — too much rope; default UX should be safe.

---

## Decision 5: Where `compute_ml_merge_view()` lives

**Question**: Should the ML view computation function live in `analysis_views.py` (alongside `MergeAnalysisView.compute()`) or in a new `ml_merge_view.py`?

**Decision**: In `analysis_views.py` as a module-level function alongside `MergeAnalysisView.compute()`. Returns a `MergeAnalysisView` instance (same type, same rendering code).

**Rationale**: The output is a `MergeAnalysisView` — the rendering path is identical to the heuristic view. Putting the builder function in the same file as the type keeps the module cohesive. A new file would be premature separation for one function.

**Alternatives considered**:
- New `ml_merge_view.py`: rejected — two files for what is essentially an alternative constructor.
- Static method on `MergeAnalysisView`: rejected — would make `MergeAnalysisView` import `MergeTrainer` and `FeatureComputer`, creating a circular dependency chain.

---

## Decision 6: ML mode async computation

**Question**: Computing features + running prediction for 500 candidate pairs involves `FeatureComputer.compute_all_pairs()` (distance matrix operations) and `MergeTrainer.predict()`. Should this be async?

**Decision**: Yes — use the existing `_AsyncState` pattern. Feature computation can take 1–5 seconds for large albums (200+ faces, 48+ candidate pairs). The probability histogram and overview panel render only after completion.

The worker result is stored in `st.session_state.ml_predict_worker`. Once done, store predictions in `st.session_state.ml_predictions` (a `Dict[Tuple[int,int], float]` of probabilities).

**Rationale**: Consistent with all other heavy operations in the app. Keeps the render thread responsive per the non-blocking UI rule in CLAUDE.md.

---

## Decision 7: Auto-pre-fill removal timing

**Question**: The current code auto-pre-fills decisions on tab load (lines 2505-2518 of `app/face_clustering.py`). Should this be removed or made opt-in?

**Decision**: Remove the auto-pre-fill entirely. Replace with two explicit buttons:
- "Smart Approve" (green): sets high-confidence groups to Approve
- "Smart Reject" (red): sets low-confidence groups to Reject

In ML mode, "Smart Approve" and "Smart Reject" are replaced by the ML pre-fill (which runs once when the model is applied, not on every tab load).

**Rationale**: Auto-pre-fill on load is the root of the training data pollution problem. The Smart Approve/Reject buttons preserve the convenience without the "invisible pre-fill" problem.
