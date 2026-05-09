# Tasks: Full Exemplar Reselection After Merge

**Spec**: [spec.md](spec.md)
**Sighting**: SIGHTING-030

## Checklist

- [ ] **T1 — Fix `_merge_two_clusters`**
  Replace lines 751–772 in `face_cluster/merge.py` with a call to
  `_ClusterStatHelper.select_exemplars(merged_nodes, d10_k=..., n_max=..., suppression_radius=...)`.
  Delete the old 13-line block entirely.

- [ ] **T2 — Unit test: bridge node discovery**
  In `tests/face_clustering/test_merge.py`, add a test that:
  1. Constructs two clusters A (3 nodes) and B (1 node) with known distances.
  2. Adds a node to A whose exemplar was not selected initially (dense core, not peripheral).
  3. Merges A+B and asserts the bridge node IS now an exemplar of the merged cluster.
  (Use synthetic distance matrix — no real data needed.)

- [ ] **T3 — Verify existing tests pass**
  Run `tests/face_clustering/` suite. No regressions expected since the fix produces
  a strictly better exemplar set.

- [ ] **T4 — Validation on Austria24_4**
  Recluster Austria24_4. Check merge_log.json for C1 vs C10 across iterations.
  Document whether exemplar_dist closes below 0.570.
  Append findings to SIGHTING-030 resolution field.

- [ ] **T5 — Update SIGHTING-030 and CHANGES_LOG.md**
