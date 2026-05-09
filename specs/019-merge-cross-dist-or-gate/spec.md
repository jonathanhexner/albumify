# Spec 019 — Merge Gate A: p25 Cross-Distance OR Gate

**Status**: Implemented
**Date**: 2026-04-30
**Author**: Jonathan + Claude

## Problem

Gate A currently requires `p25_exemplar_dist <= merge_exemplar_threshold`.
Exemplar-to-exemplar distance is a strong signal but misses cases where clusters
have many close **non-exemplar** node pairs (e.g., because exemplars sit at cluster
edges or bridge nodes weren't promoted yet — even after SIGHTING-030 fix, exemplar
selection is limited to `N_exemplars_max` nodes).

The user wants an alternative evidence path: if the 25th percentile of **all**
cross-cluster node-pair distances is low enough, that alone should satisfy Gate A
even when exemplars are unlucky.

## Proposed Design

### Gate A becomes an OR

```
passes_gate_a = (p25_exemplar_dist <= T_exemplar) OR (p25_cross_dist <= T_cross)
```

Where:
- `p25_exemplar_dist` — existing metric (25th percentile of exemplar×exemplar dists)
- `p25_cross_dist` — **new** 25th percentile of ALL `nodes_a × nodes_b` distances
- `T_exemplar` — existing `merge_exemplar_threshold` (default 0.35)
- `T_cross` — **new** `merge_cross_threshold` (default TBD — likely ~0.40, since
  all-pairs is noisier than exemplar-pairs)

### Unique-Pair Support Count (Gate B variant)

Current Gate B counts raw cross-pair hits: any node pair below threshold counts.
A single node in cluster A close to many nodes in B can inflate the count
artificially.

Proposed **unique-pair support**: greedy bipartite matching — each node may
participate in at most one counted pair.

Algorithm:
1. Compute all cross-pair distances below threshold
2. Sort ascending
3. Greedily assign pairs; once a node is used, skip it
4. Count = number of assigned pairs

This is more conservative and more meaningful. It answers: "how many independent
face-to-face close correspondences exist?"

This could either **replace** the current raw count or be offered as a second
config option (`merge_support_unique: bool`).

## Configuration Changes (`PipelineConfig`)

| Field | Type | Default | Description |
|---|---|---|---|
| `merge_cross_threshold` | `float` | 0.40 | Threshold for p25 all-pairs cross-dist OR gate |
| `merge_use_cross_gate` | `bool` | `True` | Enable the OR path (if False, Gate A is exemplar-only) |
| `merge_support_unique` | `bool` | `False` | Use unique-pair (greedy bipartite) counting for Gate B |

## Merge Log / Feature Changes

The evidence dict returned by `_evaluate_merge_evidence` gets two new fields:
- `p25_cross_dist: float` — always computed (useful for analysis even if gate is off)
- `unique_support: int` — unique-pair count (always computed)

`merge_log.json` entries carry these fields for post-hoc analysis.

## Acceptance Criteria

1. Gate A fires as OR: existing exemplar path AND new cross-dist path
2. When `merge_use_cross_gate=False`, behavior is identical to today
3. `p25_cross_dist` is logged in `merge_log.json` for every candidate (even when gate disabled)
4. Unique-pair support count is logged alongside raw support count
5. Unit tests cover:
   - OR gate: pair that fails exemplar but passes cross-dist gets `passes_gate_a=True`
   - OR gate: pair that passes exemplar but fails cross-dist still passes
   - Unique-pair support: node reuse is prevented
   - Config flag off: old behavior preserved
6. No performance regression for typical runs (< 500 faces)

## Risks / Edge Cases

- **Performance**: All-pairs distances for large clusters (100+ nodes each) is O(n*m).
  For 500 faces total this is bounded and fast. For larger datasets, numpy
  vectorized slice should keep it sub-millisecond.
- **Threshold tuning**: `T_cross` needs to be calibrated against real data. Start
  conservative (0.40) and let the user tune via the app.
- **Interaction with Gate D (diameter)**: The OR gate makes Gate A more permissive.
  Gate D (post-merge diameter) and Gate C (margin) still act as safety nets.

## Out of Scope

- Replacing adaptive diameter gate with global threshold (separate request)
- ML model retraining with new features (follow-up after feature lands)
