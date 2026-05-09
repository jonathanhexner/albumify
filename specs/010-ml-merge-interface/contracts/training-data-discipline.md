# Contract: Training Data Save Discipline (010)

**Concern**: `_collect_all_merge_labels()` in `app/face_clustering.py`
**Rule**: Only explicit human decisions are saved as training samples.

---

## Current Behavior (broken)

`_collect_all_merge_labels()` collects all entries from `merge_approval_decisions` regardless of whether they were set by the user or pre-filled automatically on tab load. Pairs pre-filled as "reject" by the auto-pre-fill logic (lines 2505-2518) are indistinguishable from user-verified rejections.

## Required Behavior

```python
def _collect_all_merge_labels(
    decisions: Dict[Tuple[int,int], str],
    sources: Dict[Tuple[int,int], str],
    pair_features: Optional[Dict],
    output_dir: Path,
) -> List[Dict]:
    """
    Collect merge labels for training data.

    ONLY saves pairs where sources[pair_key] == "human".
    ML-suggested decisions (sources[pair_key] == "ml") are excluded.
    Pairs absent from decisions (Undecided) are excluded.
    """
```

## Write Conditions Table

| Decision | Source | Written to training DB? | Written to pending_labels.json? |
|---|---|---|---|
| "approve" | "human" | YES — label=1 | YES |
| "reject" | "human" | YES — label=0 | YES |
| "approve" | "ml" | NO | YES (for merge execution) |
| "reject" | "ml" | NO | YES (for merge execution) |
| absent (Undecided) | — | NO | NO |

**Separation of concerns**: `pending_labels.json` (consumed by remerge pipeline) contains all approved pairs (both human and ml-suggested) for merge execution. Training DB only receives human decisions.

---

## Session Step Metadata

When `SessionManager.append_step()` is called for a merge step, the `params` dict MUST include:

```python
{
    "merge_mode": "heuristic" | "ml_model",
    "model_name": str | None,            # None in heuristic mode
    "ml_threshold": float | None,        # None in heuristic mode
    "n_approved_human": int,             # Human-approved pairs
    "n_approved_ml": int,                # ML-suggested approved (not human-verified)
    "n_rejected_human": int,             # Human-rejected pairs
    "n_rejected_ml": int,                # ML-suggested rejected (not human-verified)
    "n_undecided": int,                  # Pairs left undecided
    "n_training_samples_saved": int,     # Samples written to training DB this step
}
```

This makes the audit trail unambiguous: users (and future ML training runs) can see exactly how many of the merge decisions in each step were human-verified vs model-suggested.
