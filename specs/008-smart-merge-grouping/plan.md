# Technical Plan: Smart Merge Grouping

**Created**: 2026-04-20
**Spec**: [spec.md](spec.md)
**Sighting**: SIGHTING-021

---

## Overview

Add transitive grouping, component cohesion scoring, and smart auto-approve to the Merge Analysis tab. The merge candidate pairs are grouped into connected components using union-find. Each group is classified by confidence tier. The UI presents groups instead of individual pairs, with a "Smart Approve" button that auto-resolves obvious decisions.

**Design constraint**: Core grouping + cohesion algorithm lives in `face_cluster/merge.py` (pipeline layer), not the UI. This makes it straightforward to later plug into `ConservativeMerger` as a pipeline-level auto-merge gate. The UI consumes the algorithm output via a thin wrapper.

No changes to file formats or downstream pipeline.

---

## Architecture

```
merge_log.json (existing, unchanged)
       |
       v
_parse_merge_log()          (existing, unchanged)
       |
       v
List[MergeDecisionRow]      (existing)
       |
       v
group_merge_candidates()    (NEW in merge.py - pure algorithm, pipeline-ready)
       |
       v
_build_merge_groups()       (NEW in analysis_views.py - wraps with UI data)
       |
       v
List[MergeGroup]            (NEW dataclass)
       |
       v
MergeAnalysisView           (extended with merge_groups field)
       |
       v
_render_grouped_merge_gallery()  (NEW - group cards with expand)
       |
       v
merge_approval_decisions    (existing pair-based dict, unchanged)
       |
       v
save_manual_merge_snapshot() (existing, unchanged)
```

---

## Algorithm Layer (`face_cluster/merge.py`)

### New Dataclass: `CandidateGroup`

Pure algorithm output — no UI dependencies. Placed alongside existing `MarginDetail`.

```python
@dataclass
class CandidateGroup:
    group_id: int
    cluster_ids: List[int]            # sorted cluster IDs in component
    pair_keys: List[Tuple[int, int]]  # (min_id, max_id) for each pair
    pair_gate_counts: List[int]       # n_gates_passed per pair
    cohesion: float                   # fraction of pairs with 4/4 gates
    min_gates: int
    max_gates: int
    confidence: str                   # "auto_approve" | "review" | "auto_reject"
```

### New Function: `group_merge_candidates()`

```python
def group_merge_candidates(
    candidate_pairs: List[Tuple[int, int]],
    gate_counts: List[int],
    cohesion_threshold: float = 0.8,
    min_gates_for_promotion: int = 3,
) -> List[CandidateGroup]:
```

Reuses the union-find pattern from `apply_manual_merges:720-736`. Takes raw pair/gate data — no FaceRecord or MergeDecisionRow dependency. This is the function `ConservativeMerger` can call later for pipeline-level cohesion-aware merging.

---

## Presentation Layer (`face_cluster/analysis_views.py`)

### New Dataclass: `MergeGroup`

Wraps `CandidateGroup` with UI-facing data.

```python
@dataclass
class MergeGroup:
    core: CandidateGroup               # algorithm output
    pairs: List[MergeDecisionRow]      # full pair data for rendering
    total_faces: int
    n_heuristic_merged: int
    n_heuristic_rejected: int
    representative_pair: MergeDecisionRow  # lowest exemplar_dist

    # Convenience delegates
    @property
    def group_id(self): return self.core.group_id
    @property
    def cluster_ids(self): return self.core.cluster_ids
    @property
    def confidence(self): return self.core.confidence
    @property
    def cohesion(self): return self.core.cohesion
    @property
    def min_gates(self): return self.core.min_gates
    @property
    def max_gates(self): return self.core.max_gates
```

### New Function: `_build_merge_groups()`

```python
def _build_merge_groups(all_rows: List[MergeDecisionRow]) -> List[MergeGroup]:
```

Extracts pairs + gate counts from rows, calls `group_merge_candidates()` from `merge.py`, then maps each `CandidateGroup` → `MergeGroup` by attaching full MergeDecisionRow data.

### Extend `MergeAnalysisView`

Add fields:
```python
merge_groups: List[MergeGroup] = field(default_factory=list)
n_auto_approve: int = 0
n_review: int = 0
n_auto_reject: int = 0
```

Call `_build_merge_groups()` from `compute()` after existing `near_misses` computation.

---

## Layering Diagram

```
face_cluster/merge.py              (ALGORITHM — pipeline-ready)
  CandidateGroup                   pure data: cluster_ids, pair_keys, cohesion, confidence
  group_merge_candidates()         input: pairs + gate_counts → List[CandidateGroup]
                                   no FaceRecord, no UI concerns
       |
       v
face_cluster/analysis_views.py     (PRESENTATION DATA)
  MergeGroup                       wraps CandidateGroup + MergeDecisionRows + face counts
  _build_merge_groups()            calls group_merge_candidates(), attaches UI data
       |
       v
app/face_clustering.py             (UI RENDERING)
  _render_grouped_merge_gallery()  renders MergeGroup cards with crops, buttons
  Smart Approve button             reads confidence, sets pair decisions
```

**Future pipeline integration** (not in this PR, ~5 lines to add):
```python
# ConservativeMerger._find_best_merge_with_decisions():
groups = group_merge_candidates(pairs, gate_counts)
for g in groups:
    if g.confidence == "auto_approve":
        # auto-merge all pairs in this group
```

---

## UI Layer Changes (`app/face_clustering.py`)

### Session State (`_init_state`)

Add:
```python
st.session_state.merge_group_view = True        # default: group view on
st.session_state.merge_group_page = 0           # group gallery page
st.session_state.merge_group_filter = "All"     # filter for group gallery
st.session_state.merge_smart_prefilled = False   # flag: smart pre-fill done
```

Reset in `_invalidate_run_caches()`.

### Smart Approve in `_render_approval_controls()`

After existing bulk actions, add:

1. **Group summary line**:
   ```
   18 groups: 8 auto-approve (22 pairs) | 4 review (8 pairs) | 6 auto-reject (196 pairs)
   ```

2. **"Smart Approve" button**: iterates `view.merge_groups`, sets pair decisions:
   - `auto_approve` groups → all pairs "approve"
   - `auto_reject` groups → all pairs "reject"
   - `review` groups → untouched
   - Switches filter to "Review Only"

### Group Gallery: `_render_grouped_merge_gallery()`

**Toggle**: Checkbox "Group view" at top of gallery. When off, dispatch to existing `_render_flat_merge_gallery()` (renamed from `_render_unified_merge_gallery`).

**Filter options**: "All" | "Review Only" | "Auto-Approve" | "Auto-Reject"

**Group card structure**:
```
+------------------------------------------------------------------+
| Group 3 | REVIEW | 3 clusters, 15 faces | 3 pairs | cohesion 67% |
|------------------------------------------------------------------|
| C5 (8)         C11 (4)         C22 (3)                           |
| [crop][crop]   [crop][crop]    [crop][crop]                      |
|------------------------------------------------------------------|
| 4/4 on 2 pairs, 3/4 on 1 pair                                   |
| [APPROVE GROUP]  [REJECT GROUP]                                  |
| > Show 3 individual pairs                                        |
+------------------------------------------------------------------+
```

**Expanded pair view**: Reuses existing gate badges and `_render_pair_crops()`. Each pair has its own approve/reject buttons that override the group-level decision for that specific pair.

**Group-level actions**:
- "Approve Group" → sets all pairs in group to "approve" in `merge_approval_decisions`
- "Reject Group" → sets all pairs in group to "reject"
- Per-pair override within the expansion takes immediate effect

**Pagination**: By group count, 10 groups/page.

### Smart Pre-fill in `render_merge_analysis_tab()`

After `MergeAnalysisView.compute()` completes (line ~2377), if `merge_smart_prefilled` is False and no saved decisions were loaded:
- Apply smart pre-fill: auto_approve → "approve", auto_reject → "reject", review → unset
- Set `merge_smart_prefilled = True`

This replaces the current heuristic pre-fill (merged→approve, rejected→reject) when the view is available.

### Rename existing gallery

`_render_unified_merge_gallery()` → `_render_flat_merge_gallery()` (no logic changes).

New dispatch:
```python
def _render_merge_gallery(view, result):
    if st.session_state.merge_group_view:
        _render_grouped_merge_gallery(view, result)
    else:
        _render_flat_merge_gallery(view, result)
```

Called from `_render_merge_analysis()` replacing the current `_render_unified_merge_gallery()` call.

---

## Files Modified

| File | Changes | Layer |
|------|---------|-------|
| `face_cluster/merge.py` | `CandidateGroup` dataclass + `group_merge_candidates()` | Algorithm (pipeline-ready) |
| `face_cluster/analysis_views.py` | `MergeGroup` wrapper, `_build_merge_groups()`, extend `MergeAnalysisView` | Presentation data |
| `app/face_clustering.py` | Session state, Smart Approve, group gallery, smart pre-fill | UI |
| `tests/face_clustering/test_merge_analysis.py` | Unit tests for grouping, cohesion, classification | Tests |

**No changes to**: `manual_merge_snapshot.py`, `export.py`, `merge_log.json`, `merge_decisions.json`.

---

## Backward Compatibility

- `merge_approval_decisions` stays `{(min_id, max_id): "approve"|"reject"}` — pair-keyed
- `merge_decisions.json` format unchanged — existing saved decisions load correctly
- Old runs produce groups dynamically from existing merge_log.json
- Flat pair view remains accessible via toggle
- ML training pipeline (`_save_merge_features_if_available`) unaffected — joins by pair key

---

## Union-Find Reference

Existing pattern in `merge.py:720-736`:
```python
parent = {cid: cid for cid in all_cluster_ids}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb
```

The new `group_merge_candidates()` in `merge.py` uses the same inline pattern.

---

## Cohesion Classification Logic

```
cohesion = count(pairs with n_gates_passed == 4) / len(pairs)

if all pairs have n_gates_passed == 4:
    confidence = "auto_approve"
elif cohesion >= 0.8 and min_gates >= 3:
    confidence = "auto_approve"     # cohesion promotion (FR-003)
elif all pairs have n_gates_passed <= 2:
    confidence = "auto_reject"
else:
    confidence = "review"
```

The 80% threshold and min_gates >= 3 guard ensure:
- A group with one 2/4 pair is never auto-approved regardless of cohesion
- A group needs substantial internal agreement before promotion

---

## Test Plan

### Unit Tests (`tests/face_clustering/test_merge_analysis.py`)

| Test | Input | Expected |
|------|-------|----------|
| Transitive grouping | pairs (A,B), (B,C), (D,E) | 2 groups: {A,B,C}, {D,E} |
| Single-pair group | pair (A,B) only | 1 group with 1 pair, 2 clusters |
| Empty input | no pairs | empty groups list |
| All pairs accounted | N pairs | sum of group pairs == N |
| auto_approve: all 4/4 | group where all pairs 4/4 | confidence="auto_approve" |
| auto_reject: all <=2/4 | group where all pairs <=2 | confidence="auto_reject" |
| review: mixed | 4/4 + 3/4 mix, cohesion < 80% | confidence="review" |
| cohesion promotion | 5/6 pairs 4/4, 1 pair 3/4 (83%) | confidence="auto_approve" |
| cohesion no promotion | 1 pair 4/4, 1 pair 2/4 (50%) | confidence="review" (min_gates < 3) |
| Sort order | mixed confidence groups | review first, then auto_approve, then auto_reject |

### Integration / Manual Tests

- Load Austria24_2 run, verify group count << 226
- Smart Approve, then Apply Approved Merges → verify remerge succeeds
- Toggle between group/flat view → verify both render correctly
- Load old run (pre-feature) → verify groups computed, no errors

---

## Implementation Phases

**Phase 1**: Algorithm layer — `CandidateGroup` + `group_merge_candidates()` in `merge.py` + unit tests
**Phase 2**: Presentation layer — `MergeGroup` wrapper + `_build_merge_groups()` in `analysis_views.py`, extend `MergeAnalysisView`
**Phase 3**: Session state + Smart Approve button in `face_clustering.py`
**Phase 4**: Group gallery UI in `face_clustering.py`
**Phase 5**: Smart pre-fill + flat gallery rename + dispatch
**Phase 6**: Manual testing on Austria24_2
