# Contract: analysis_views.py API Extensions (010)

**Module**: `face_cluster/analysis_views.py`
**Change type**: Extensions (existing interfaces unchanged)

---

## Extended: MergeDecisionRow

```python
@dataclass
class MergeDecisionRow:
    # ... all existing fields unchanged ...

    # NEW: ML prediction fields (optional — None in heuristic mode)
    ml_prob: Optional[float] = None   # ML merge probability (0.0–1.0)
    ml_pred: Optional[int] = None     # ML class prediction: 1=merge, 0=reject
```

**Invariants**:
- If `ml_prob` is set, `ml_pred` must also be set and vice versa.
- `ml_prob` is in range [0.0, 1.0].
- `ml_pred` is 1 if `ml_prob >= threshold_at_prediction_time`, else 0.
- In heuristic mode, both fields are None. In ML mode, both are populated.

---

## Extended: MergeAnalysisView

```python
@dataclass
class MergeAnalysisView:
    # ... all existing fields unchanged ...

    # NEW: ML mode metadata (None in heuristic mode)
    ml_threshold: Optional[float] = None  # Threshold used when view was built in ML mode
```

---

## New Function: compute_ml_merge_view

```python
def compute_ml_merge_view(
    result: PipelineResult,
    model_payload: Dict,          # Output of MergeTrainer.load_model()
    threshold: float = 0.5,       # Merge probability cutoff
    candidate_threshold: float = 0.45,  # Exemplar distance cutoff for candidate discovery
) -> MergeAnalysisView:
```

**Inputs**:
- `result`: A loaded `PipelineResult` with `cluster_result`, `faces`, and `distance_matrix` (or loadable from output_dir).
- `model_payload`: Dict with keys `"model"`, `"scaler"`, `"feature_names"`, `"metadata"`.
- `threshold`: Pairs with `ml_prob >= threshold` are classified as "proposed_merge". Pairs with `ml_prob < (1 - threshold)` are "proposed_reject". Others are borderline.
- `candidate_threshold`: Maximum exemplar distance to consider a pair as a merge candidate.

**Output**: `MergeAnalysisView` instance with:
- `merges`: `MergeDecisionRow` list for pairs where `ml_pred == 1`, with `ml_prob` and `ml_pred` populated, `action = "proposed_merge"`.
- `rejections`: `MergeDecisionRow` list for pairs where `ml_pred == 0`, with `action = "proposed_reject"`.
- `merge_groups`: Grouped via `group_merge_candidates()` with probability-to-gate mapping.
- `ml_threshold`: The threshold value used.
- `n_auto_approve`, `n_review`, `n_auto_reject`: Derived from group confidence tiers.
- `gate_rejection_counts`, `gate_sole_blocker_counts`: Empty dicts (not applicable in ML mode).
- `near_misses`: Pairs with `ml_prob` between 0.4 and 0.6 (borderline cases).

**Raises**:
- `ValueError("Feature version mismatch: model requires V{x}, current is V{y}")` if `model_payload["metadata"]["feature_version"] != FeatureComputer.VERSION`.
- `ValueError("No candidate pairs found. Check that cluster_result has clusters.")` if feature computation finds no pairs.

**Performance**: For 500 candidate pairs, completes in < 3 seconds (feature computation dominated).

---

## New Function: compute_pair_feature_contributions

```python
def compute_pair_feature_contributions(
    pair_features: Dict,           # Single ClusterPairFeatures dict
    model_payload: Dict,           # Loaded model
    top_n: int = 3,
) -> List[Tuple[str, float, str]]:
```

**Output**: List of `(feature_name, contribution_value, direction)` tuples where:
- `contribution_value` is signed (positive = towards merge, negative = towards reject).
- `direction` is "→ merge" if positive, "→ reject" if negative.
- Sorted by `abs(contribution_value)` descending.
- Length is min(top_n, len(feature_names)).

**Model-specific behavior**:
- Logistic regression: `contribution = coef[i] * scaled_value[i]` (exact).
- XGBoost/MLP: `contribution = global_importance[i] * sign(feature_value[i] - feature_mean[i])` (proxy). Displayed with "(global)" suffix to indicate approximation.
