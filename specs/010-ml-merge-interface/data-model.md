# Data Model: ML Merge Interface (010)

**Feature**: `010-ml-merge-interface`
**Phase**: 1 — Entities, fields, state transitions

---

## Entities

### MergeDecision (session state — not persisted to disk)

Represents the user's current decision for a single candidate pair.

| Field | Type | Values | Notes |
|---|---|---|---|
| `pair_key` | `Tuple[int, int]` | `(min_cid, max_cid)` | Canonical key: smaller cluster ID first |
| `decision` | `str \| None` | `"approve"` \| `"reject"` \| `None` | None = Undecided |
| `source` | `str` | `"human"` \| `"ml"` | Who set this decision |

**Storage**: Two parallel session state dicts (keeps backward compat):
- `st.session_state.merge_approval_decisions: Dict[Tuple[int,int], str]` — maps pair → "approve"|"reject". Absence = Undecided.
- `st.session_state.merge_decision_sources: Dict[Tuple[int,int], str]` — maps pair → "human"|"ml". Set when decision is written.

**State transitions**:
```
Undecided  --[user clicks Approve]--> Approve (human)
Undecided  --[user clicks Reject]---> Reject  (human)
Undecided  --[ML pre-fill, prob>=thr]--> Approve (ml)
Undecided  --[ML pre-fill, prob<1-thr]--> Reject (ml)
Approve(ml) --[user changes]--> Reject  (human)  ← override: ML badge removed
Reject(ml)  --[user changes]--> Approve (human)  ← override: ML badge removed
Approve(human) --[user clicks again]--> Undecided (cleared)
Reject(human)  --[user clicks again]--> Undecided (cleared)
```

**Training data eligibility**: Only pairs where source == "human" are saved as training samples.

---

### MLPrediction (session state — not persisted to disk)

Stores the ML model's output for a single candidate pair.

| Field | Type | Notes |
|---|---|---|
| `pair_key` | `Tuple[int, int]` | Same canonical key |
| `ml_prob` | `float` | Probability of merge (0.0–1.0) |
| `ml_pred` | `int` | 1=merge, 0=reject (based on threshold at predict time) |
| `top_features` | `List[Tuple[str, float]]` | Top 3 (feature_name, contribution_value) |

**Storage**: `st.session_state.ml_predictions: Dict[Tuple[int,int], MLPrediction]`

Cleared when: mode switches back to heuristic, different model selected, different run loaded.

---

### MergeDecisionRow (extended — in `face_cluster/analysis_views.py`)

Existing dataclass extended with two optional fields:

| New Field | Type | Default | Notes |
|---|---|---|---|
| `ml_prob` | `Optional[float]` | `None` | ML probability for this pair |
| `ml_pred` | `Optional[int]` | `None` | ML class prediction (0/1) |

No other fields change. The `action` field retains its meaning ("merged" / "rejected") in heuristic mode. In ML mode, `action` reflects the ML prediction: "proposed_merge" if ml_pred==1, "proposed_reject" if ml_pred==0.

---

### MergeMode (app-level UI state)

An enum-like string stored in session state.

| Value | Meaning |
|---|---|
| `"heuristic"` | 4-gate ConservativeMerger drives decisions |
| `"ml_model"` | Trained ML model drives decisions |

**Storage**: `st.session_state.merge_mode: str` — default `"heuristic"`.

---

### MergeAnalysisView (extended — in `face_cluster/analysis_views.py`)

Existing dataclass extended with one optional field:

| New Field | Type | Default | Notes |
|---|---|---|---|
| `ml_threshold` | `Optional[float]` | `None` | Threshold used when view was computed in ML mode |

No structural changes. In ML mode, `merge_groups` still contains `CandidateGroup` objects (confidence tiers derived from probability-to-gate mapping). The `merges` list contains `MergeDecisionRow` with `ml_prob` populated.

---

### TrainingDataRecord (merge_training_data DB table — unchanged schema)

Only the write logic changes: records are written only when `source == "human"`. The DB schema (`run_id`, `cluster_a`, `cluster_b`, `label`, `features_json`, ...) is unchanged.

---

## State Diagram: Merge Analysis Tab (ML mode)

```
Tab Opens
    │
    ▼
[Mode Selector: Heuristic / ML Model]
    │
    ├── Heuristic ──────────────────────────────────────────────────┐
    │   Compute MergeAnalysisView (async)                          │
    │   All pairs → Undecided                                       │
    │   [Smart Approve] [Smart Reject] available                   │
    │   [Apply + Remerge] when ≥1 Approved                         │
    │                                                              │
    └── ML Model ──────────────────────────────────────────────────┘
        Select model from dropdown
        Set threshold (default 0.5)
        [Apply threshold] →
            Compute features + predict (async _AsyncState)
            Pre-fill: prob>=thr → Approve(ml), prob<(1-thr) → Reject(ml), else → Undecided
            Overview panel: histogram, counts
        Gallery renders (grouped, same layout as heuristic)
        User reviews undecided pairs, overrides ML suggestions
        [Apply + Remerge]:
            - Approved (human + ml-suggested) → merged
            - Rejected (human + ml-suggested) → not merged
            - Undecided → not merged, carry forward
            - Training data: only human decisions saved
```

---

## Session State Keys (new and modified)

| Key | Type | New/Modified | Purpose |
|---|---|---|---|
| `merge_approval_decisions` | `Dict[pair, str]` | Modified (no pre-fill) | explicit approve/reject decisions |
| `merge_decision_sources` | `Dict[pair, str]` | **New** | "human" or "ml" per decision |
| `ml_predictions` | `Dict[pair, dict]` | **New** | ML probability + top features per pair |
| `merge_mode` | `str` | **New** | "heuristic" or "ml_model" |
| `ml_selected_model` | `str \| None` | **New** | Selected model name |
| `ml_threshold` | `float` | **New** | Current threshold (default 0.5) |
| `ml_predict_worker` | `_AsyncState \| None` | **New** | Background prediction worker |
| `ml_candidate_threshold` | `float` | **New** | Candidate discovery threshold (default 0.45, wider = 0.65) |
