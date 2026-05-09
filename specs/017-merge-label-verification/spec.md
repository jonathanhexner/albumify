# Feature Specification: Merge Label Verification Tab

**Created**: 2026-04-25
**Status**: In Progress
**Depends on**: `specs/016-merge-eda-validation/`
**UI Mock**: [mock.html](mock.html) (open in browser)

## Problem Statement

The ML merge classifier needs human-verified labels. Currently labels come from the
pipeline's 4-gate heuristic. The key insight about label quality:

- **Merged pairs (label=1)** — likely correct. Users would have reverted bad merges.
  These just need a quick confirmation pass.
- **Rejected pairs (label=0)** — unreliable. The pipeline is conservative, so many
  rejects are probably wrong (should have been merged). Germany_10 inspection confirms
  incorrect reject labels on visually-same-person pairs.
- **Non-candidate pairs** — very likely correct rejects (exemplar distance too large
  to be the same person).

The verification tab's primary job is to **review rejected pairs** (especially
borderline ones near the merge threshold) and flip the incorrect ones to merge.
This turns the existing heuristic labels into a starting point, not a source of truth.

## Data Landscape

One canonical run per source dataset. **Use the earliest pre-merge run** that has
embeddings + crops — post-merge reclusters collapse the candidate pair space to near
zero (Germany_18_recluster_103838 had 3 pairs; Germany_10 has 748).

| Dataset | Canonical Run | Clusters | Pairs @0.45 | Pairs @0.60 | Faces | Existing DB Labels |
|---------|--------------|----------|------------|------------|-------|--------------------|
| Google_Germany | Germany_12 | 26 | ~15 | ~37 | 1103 | 0 (fresh; Germany_10 excluded — K=5 fragmentation) |
| Austria_24 | Austria24_1 | 39 | ~30 | ~130 | 753 | 29 merges |
| Shira_Album1 | shira_album1/base | 49 | ~13 | ~103 | 2236 | 0 (fresh) |
| Noa_5-7 | Noa5-7 | 20 | ~2 | ~9 | 1206 | 0 |
| Noa_2-5 | Noa2-5 | 9 | ~0 | ~1 | 418 | 0 (low priority) |

Crops live directly in each run's `crops/` directory — no cross-run fallback needed.

## User Stories

### US1 — Review Merge Candidates by Distance (P1)

As an ML engineer, I need to review cluster pairs ordered by distance, see the
heuristic's decision, and confirm or override it.

**Acceptance Criteria:**

1. Table lists all candidate pairs as rows, sorted by `min_exemplar_dist` ascending
   (closest first — most likely to need merging).
2. Each row displays:
   - 5–10 face thumbnails (exemplars first, then remaining faces) from both clusters.
   - `min_exemplar_dist`, `p10_cross_dist`, cluster sizes, heuristic label.
3. Each row has three decision buttons: **Merge**, **Reject**, **Ignore**.
   - **Merge** (label=1): pair is same person, used as positive training sample.
   - **Reject** (label=0): pair is different people, used as negative training sample.
   - **Ignore**: pair is ambiguous — excluded entirely from training data (not saved).
4. Each row has a **Verified / Not Verified** toggle to track review progress.
5. Clicking a row expands a detail panel showing:
   - All face crops from both clusters (exemplars highlighted).
   - Full distance metrics: min_exemplar_dist, p10_cross_dist, p50_cross_dist,
     support_fraction, post_merge_diameter, diameter_expansion.
   - Which of the 4 gates passed/failed (from merge_log), if available.
6. Labels are saved to `training_db` with `source='human'`. Heuristic labels
   (from merge_log) are pre-loaded with `source='heuristic'` and can be overridden.
7. Progress indicator: "Verified X / Y candidate pairs (Z remaining)".
8. User can filter: show only unverified, only heuristic-rejects, or all.

### US2 — Dataset Selection (P1)

As an ML engineer, I need to select which source dataset to label.

**Acceptance Criteria:**

1. Dropdown lists the canonical runs (one per source dataset).
2. Selecting a dataset loads that run's clusters, embeddings, and merge candidates.
3. Crops are loaded from the designated crop source run (may differ from the selected run).
4. Summary shows: n_clusters, n_candidate_pairs, n_verified, n_unverified.

### US3 — Adjustable Distance Threshold for Candidate Generation (P2)

As an ML engineer, I need to control how many candidate pairs to review by adjusting
the distance threshold.

**Acceptance Criteria:**

1. Slider for `candidate_threshold` (range 0.2 — 1.0, default 0.45).
2. Increasing the threshold shows more pairs (further apart = harder cases).
3. Pair count updates live as threshold changes.

### US4 — Label Summary and Export (P2)

As an ML engineer, I need to see labeling progress and export the labeled dataset.

**Acceptance Criteria:**

1. Summary table: per-dataset counts of merge/reject/ignore by source (heuristic vs human).
2. Total counts across all datasets.
3. Export button saves all labeled data as CSV for notebook consumption.
   Ignored pairs are excluded from the export (they carry no label).

## Label Trust Hierarchy

When both heuristic and human labels exist for a pair, human always wins:

1. `source='human'` — highest trust, user explicitly reviewed and labeled
2. `source='heuristic', label=1` (merged) — high trust, user didn't revert
3. `source='heuristic', label=0` (rejected) — low trust, pipeline is conservative
4. `source='auto_negative'` — medium trust, non-candidate pairs far apart

**Ignored pairs** (`source='human', label=NULL` or excluded flag) are omitted from
training entirely — they are neither positive nor negative. Use this for ambiguous
pairs where forcing a label would add noise.

The notebook's `load_training_data()` already applies human overrides on top of
heuristic labels, so this hierarchy is enforced at consumption time.

## Design Constraints

- **Crop fallback**: When the canonical run has no crops, use crops from a sibling run
  on the same dataset. Face IDs are stable across reclusters.
- **No new algorithms**: This tab uses existing `FeatureComputer.compute_all_pairs()` and
  existing `training_db.upsert_training_samples()`.
- **Distance-only features**: The tab shows distance metrics only (min_exemplar_dist,
  p10_cross_dist, support_fraction, diameter_expansion). No t_local/t_global.
- **Async pattern**: Feature computation must run in background thread per CLAUDE.md rules.

## Edge Cases

- Cluster with 1 face: only 1 crop to show. Display it centered.
- Missing crop file: show face_id + "missing" placeholder with resolved path for debugging.
- All pairs already verified: show "All done" message with summary.
- Very large dataset (1000+ pairs at threshold 0.9): all rows render in the table,
  but detail panel only loads crops on click to keep the UI responsive.

## Success Criteria

- **SC-001**: User can verify 50+ pairs per session without UI lag.
- **SC-002**: Labels persist across sessions (stored in training_db).
- **SC-003**: No duplicate labels (UPSERT keyed by run_id, cluster_a, cluster_b).
- **SC-004**: Labeled dataset is consumable by `eda_merge_ml.ipynb` without modification
  (same schema as `load_training_data()`).
- **SC-005**: Heuristic merge labels are pre-loaded so user only needs to confirm, not re-enter.
