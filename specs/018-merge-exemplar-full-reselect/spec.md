# Feature Specification: Full Exemplar Reselection After Merge

**Created**: 2026-04-30
**Status**: Implemented
**Sighting**: SIGHTING-030

## Problem Statement

After two clusters are merged, `_merge_two_clusters` recomputes exemplars by taking the union
of the two existing exemplar sets and re-ranking by d10 within that pool. Nodes that were never
exemplars in their original cluster are never considered — even if they are well-positioned bridge
nodes in the merged cluster's embedding space.

This causes persistent near-miss rejections: a node in the merged cluster that sits close to a
third cluster is invisible to Gate A if it was never promoted to exemplar in its original small
cluster.

**Confirmed example**: Austria24_4 / recluster_6 — C1 vs C10 fails by 0.005 (0.575 vs 0.570)
across all 5 iterations despite C1 growing 135 → 143. The newly absorbed nodes from C6, C12, C16
were never exemplars, so none enter the candidate pool.

## Objective

Replace the exemplar-union shortcut in `_merge_two_clusters` with a full d10 reselection over
all nodes in the merged cluster, using the same greedy suppression logic already present in
`_ClusterStatHelper.select_exemplars`.

## Constraints

- Must not change the exemplar selection logic for non-merged clusters (initial stage unchanged).
- `_ClusterStatHelper.select_exemplars` already exists in `merge.py` and implements the correct
  algorithm. The fix reuses it — no new algorithm code needed.
- The `exemplars_d10_threshold` gate from `D10ExemplarSelector` is intentionally **not** applied
  post-merge: a large merged cluster may have no node with d10 below the threshold, so we want
  the best available nodes unconditionally.
- Performance: O(|merged_nodes|²) d10 computation per merge step. For typical cluster sizes
  (<200 nodes, <20 merge steps) this is negligible.

## Design

### Current code (`_merge_two_clusters`, lines 751–772)

```python
combined_exemplars = list(set(exemplars_a + exemplars_b))
if len(combined_exemplars) > self.config.N_exemplars_max:
    d10_values = []
    for node in combined_exemplars:
        k = min(self.config.d10_k, len(merged_nodes) - 1)
        dists = [distance_matrix[node, other] for other in merged_nodes if other != node]
        if len(dists) >= k:
            d10 = sorted(dists)[k - 1]
            d10_values.append((d10, node))
    d10_values.sort()
    new_exemplars[cluster_id_a] = [node for _, node in d10_values[:self.config.N_exemplars_max]]
else:
    new_exemplars[cluster_id_a] = combined_exemplars
```

Problems:
1. Candidate pool limited to prior exemplars — bridge nodes excluded.
2. When trimming, does not apply suppression radius — exemplars may cluster together.
3. Skips nodes where `len(dists) < k` without fallback.

### Proposed code

```python
helper = _ClusterStatHelper(distance_matrix)
new_exemplars[cluster_id_a] = helper.select_exemplars(
    merged_nodes,
    d10_k=self.config.d10_k,
    n_max=self.config.N_exemplars_max,
    suppression_radius=self.config.exemplar_suppression_radius,
)
```

`_ClusterStatHelper.select_exemplars` (already in merge.py):
- Ranks ALL nodes in the merged cluster by d10
- Applies greedy suppression (spacing exemplars out)
- Caps at `n_max`
- Handles edge cases (size 0, size 1)

### Expected behaviour change

After merging C1+C6, any node in C6 that sits close to C10 can now become an exemplar of C1,
potentially closing the 0.005 gap in subsequent iterations. More generally, merges that were
previously blocked by the exemplar-union blind spot may now proceed.

## Acceptance Criteria

1. `_merge_two_clusters` uses `_ClusterStatHelper.select_exemplars(merged_nodes, ...)`.
2. A unit test verifies that a bridge node (not in either original exemplar set) becomes an
   exemplar after merge when it has the best d10 score.
3. Existing tests pass.
4. Re-run Austria24_4 recluster and document whether C1 vs C10 behaviour changes.

## Edge Cases

- **Single-node clusters**: `_ClusterStatHelper.select_exemplars` handles `len(nodes) <= 1`.
- **Large cluster exceeds N_exemplars_max**: greedy suppression naturally caps at n_max.
- **All nodes have identical d10**: suppression radius still applies; first node is always
  selected as fallback.
