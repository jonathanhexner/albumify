# spec-031: Absolute max-diameter post-merge step

**Status**: DRAFT — awaiting user approval before implementation
**Triggered by**: SIGHTING-059 Issue 5 (cluster 4 chain merge, internal max distance 0.867)
**Owner**: ML Engineer + Senior SW Engineer
**Approach**: Implement as a **separate pipeline step** (per user direction), not as a 5th gate inside `ConservativeMerger`.

---

## Why a separate step

The existing 4 merge gates are *per-pair* checks (A–B about to be combined: do they pass exemplar / support / margin / diameter-expansion). They cannot see global cluster cohesion after a chain of pair-wise merges has stayed within the relative expansion factor at every step.

A separate post-merge step:
- runs *after* `ConservativeMerger` has produced its final set
- operates on whole clusters, not pairs
- rejects clusters whose diameter exceeds an absolute ceiling
- doesn't need to entangle with the existing merge state machine
- can be enabled/disabled independently

Rollback is one config flag instead of re-tuning four existing gates.

---

## User stories

**US1** (P1): As a user reviewing a face-clustering run, I want any cluster whose internal cosine-distance diameter exceeds a configurable absolute threshold to be flagged and split back into its constituent pre-merge groups, so chain-merged blobs (e.g. cluster 4 with diameter 0.867) don't survive into the final output.

**US2** (P1): As a user tuning merge thresholds, I want a single absolute `max_diameter` slider (default 1.2) that I can lower if I see chain-merged clusters, independently of the existing relative `diameter_expansion_factor`.

**US3** (P2): As a developer auditing a run, I want a per-cluster `cluster_diameter_cap` decision logged the same way merge decisions are (cluster ID, diameter, threshold, action: kept / split), so I can grep the run for "which clusters did the cap reject".

---

## Functional requirements

**FR-001** New pipeline step `cluster_diameter_cap` registered in `face_cluster/pipeline.py` between `merge` and `export`.

**FR-002** Step config:
```yaml
cluster_diameter_cap:
  enabled: true
  max_diameter: 1.2          # absolute cosine-distance ceiling
  action_on_violation: split  # "split" | "reject" | "warn_only"
```
- `split`: revert the cluster back to its pre-merge component clusters (requires tracking provenance)
- `reject`: mark as noise (cluster_id=-1) and re-distribute faces to holdout
- `warn_only`: log + emit decision, but don't change cluster assignments

**FR-003** Diameter definition: maximum pairwise cosine distance among all faces in the cluster (consistent with how the existing diameter gate computes it).

**FR-004** For each merged cluster, emit a `ClusterCapDecision` with:
- `cluster_id`, `n_faces`, `diameter`, `max_diameter_threshold`, `action`, `reason`
- on `split`: the list of pre-merge component cluster IDs the faces were redistributed into
- on `reject`: count moved to noise

**FR-005** Decisions persisted to a new table `cluster_cap_decisions` in `face_clustering.db` (under `_v4/` per spec-030) AND surfaced in a new tab in the FC App ("Cluster Cap").

**FR-006** Disabling the step (`enabled: false`) is a no-op — faces and clusters unchanged, no decisions emitted.

**FR-007** Step does not run if `merge_enabled=false` (no merges to cap).

**FR-008** UI slider in `app/shared/merge_controls.py` added below the existing diameter gate, range 0.5–2.0 step 0.05, default 1.2, with help text linking to spec-031.

---

## Non-functional requirements

**NFR-001** Time complexity O(C × n²) where C is number of merged clusters and n is faces per cluster — same as existing diameter computation. No new dense matrix operations across clusters.

**NFR-002** Step must be skippable without changing run output when `enabled: false` (verified by a regression test on a fixture run).

**NFR-003** Default `max_diameter = 1.2` chosen as: "well above typical same-identity ArcFace cosine distance (~0.4–0.6) but below the 0.867 user-reported chain-merged blob".  Open for retuning during US1 implementation.

---

## Out of scope

- Re-tuning the existing 4 merge gates. They stay as-is.
- Per-cluster sub-clustering (DBSCAN-style) to split into multiple new clusters. The split action reverts to known pre-merge components only.
- Auto-merging the split clusters back together with relaxed thresholds. If the user wants that, it's a separate run with adjusted merge params.

---

## Touch points (estimated lines)

| File | Purpose | Estimated size |
|---|---|---|
| `face_cluster/cluster_diameter_cap.py` (NEW) | Step implementation | ~120 lines |
| `face_cluster/types.py` | Add `ClusterCapDecision` dataclass | +25 lines |
| `face_cluster/pipeline.py` | Register step + wire between merge and export | +30 lines |
| `face_cluster/config.py` (PipelineConfig) | Add `cluster_diameter_cap_*` fields | +6 lines |
| `face_cluster/run_exporter.py` | New table `cluster_cap_decisions`, schema v5 | +40 lines |
| `face_cluster/run_store.py` | New `cap_decisions()` reader | +25 lines |
| `app/shared/merge_controls.py` | New slider | +12 lines |
| `app/face_clustering/tabs/cluster_cap_tab.py` (NEW) | Optional UI tab — could defer | ~80 lines |
| `tests/face_clustering/test_cluster_diameter_cap.py` (NEW) | Unit + regression | ~150 lines |
| `configs/pipeline.yaml` | Add defaults | +6 lines |

Total: ~500 lines net new code across 10 files, of which ~250 is test.

---

## Open questions for the user

1. **Default action on violation**: `split` (revert to pre-merge), `reject` (move to noise), or `warn_only`?
   The "right" answer depends on whether you trust the pre-merge state more than the merged state. For the cluster-4 case, `split` would recover 3 sub-identities; `reject` would lose all 28 faces to noise.
2. **Diameter formula**: max pairwise distance (worst-case), or 95th percentile pairwise (robust)? Worst-case is what the existing gate D uses and is easier to reason about. P95 is more forgiving on outliers.
3. **Should the cap apply to merged clusters only, or to all clusters including never-merged ones?** Strictly the chain problem is about merges, but a never-merged cluster with diameter 0.9 is also suspicious. Default: merged-only (cheaper, narrower blast radius).

---

## Acceptance criteria

- A1: Running on `face_clustering_20260510_231628` with `max_diameter=0.5` and `action=split` produces at least 2 split events; cluster 4 (currently diameter 0.867) is split into at least 2 sub-clusters.
- A2: Running with `enabled=false` produces byte-identical output to a baseline run without the step.
- A3: Every merged cluster produces exactly one `ClusterCapDecision` row.
- A4: The "Cluster Cap" UI tab lists every decision with cluster ID, diameter, threshold, action, and (for split) the post-split cluster sizes.
- A5: Static check: the cap step never runs when `merge_enabled=false`.

---

## References

- SIGHTING-059 (open) — original report, contains the cluster-4 chain forensics
- spec-030 — RunExporter / RunStore (cap decisions persist via the same v4 channel)
- `face_cluster/merge.py:ConservativeMerger` — existing 4-gate logic that the cap sits behind
