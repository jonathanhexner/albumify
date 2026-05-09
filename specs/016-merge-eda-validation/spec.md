# Feature Specification: Merge EDA & Training Data Validation

**Created**: 2026-04-25
**Status**: Implemented
**Notebooks**: `notebooks/face_clustering/eda_merge_explore.ipynb`, `notebooks/face_clustering/eda_merge_ml.ipynb`

## Problem Statement

The ML merge classifier needs clean, correctly-labeled training data to learn when two face clusters should be merged. Current notebooks have several data quality issues that silently corrupt the training signal:

1. **Duplicate runs** — Multiple result directories contain identical merge decisions (same cluster pairs, same outcomes). Including all of them inflates the dataset without adding information, and causes the model to overfit to those repeated patterns.
2. **Transitivity labeling bug** — When cluster A merges with B (iter 1), then the combined A+B merges with C (iter 2), the pair (B, C) in the pre-merge feature matrix is labeled "reject" even though B and C ended up in the same final cluster. This creates false negatives in training.
3. **Candidate threshold mismatch** — Notebooks use `candidate_threshold=0.9` while the pipeline uses `0.45`. Pairs with exemplar distance 0.45–0.9 were never actual merge candidates — labeling them as "reject" adds out-of-distribution noise.
4. **Confirmed-reject labels not used** — When the pipeline produces multiple distinct clusters of clearly different people, those non-candidate pairs are strong negative labels but aren't harvested.
5. **Feature leakage from `t_global`** — `t_global` is constant per run, so the model uses it as a run identifier rather than a per-pair discriminator.
6. **Broken crop display** — Notebook path references are wrong after the move to `notebooks/face_clustering/`, making visual validation impossible.

## User Stories

### US1 — Data Quality Validation (P1: Must-have)

As an ML engineer, I need the EDA notebook to validate data quality before training so I can trust the labels.

**Acceptance Criteria:**

1. **Deduplication**: The notebook identifies runs with identical merge decisions (by hashing `(cluster_a, cluster_b, action)` tuples) and keeps only one representative per unique decision set. A summary table shows: `run_id | hash | kept/dropped | n_merged | n_rejected`.
2. **Zero-merge filter**: Runs with `n_merged == 0` are excluded from positive training data. They may still contribute negative labels if the user explicitly confirms them.
3. **Transitivity closure**: After loading the merge log, the notebook computes the transitive closure of merged pairs using union-find. If (A,B) merged and (B,C) merged, then (A,C) is labeled positive — even though the merge log doesn't explicitly record (A,C). A warning is printed for any transitively-inferred labels.
4. **Threshold alignment**: `candidate_threshold` defaults to the pipeline's `merge_candidate_threshold` (0.45), not an arbitrary wide net. An explicit note explains why this matters.

### US2 — Correct Crop Visualization (P1: Must-have)

As an ML engineer, I need to see exemplar face crops for merged and rejected pairs so I can visually verify label correctness.

**Acceptance Criteria:**

1. `RUN_DIR` path resolves correctly from the notebook's location (`notebooks/face_clustering/`).
2. For each merged/rejected pair, the notebook shows 2–3 exemplar crops from each cluster side-by-side.
3. If a crop file is missing, the notebook displays the face_id and the attempted path for debugging.

### US3 — Feature Interpretability Analysis (P1: Must-have)

As an ML engineer, I need to understand which features drive merge decisions and whether any features leak information.

**Acceptance Criteria:**

1. The notebook flags features that are constant per run (like `t_global`) and notes they act as run identifiers, not per-pair discriminators.
2. Feature distributions are shown for merged vs rejected pairs with clear visual separation assessment.
3. Correlation analysis identifies redundant features (e.g., `t_local` vs `diameter_max`).
4. Derived ratio features are computed and compared to raw features for predictive power.

### US4 — Negative Label Harvesting (P2: Should-have)

As an ML engineer, I need strong negative labels from confirmed-different clusters to balance the training set.

**Acceptance Criteria:**

1. Non-candidate pairs (exemplar distance > `candidate_threshold`) from runs with manual confirmation are harvested as negatives.
2. Different identity clusters within the same run that were never merge candidates are labeled as rejects.
3. Harvested negatives are tagged with `source='auto_negative'` to distinguish from human labels.
4. The notebook validates that harvested negatives don't contradict transitivity — no pair where both clusters later merged into the same final cluster is labeled negative.

### US5 — Feature Engineering Exploration (P2: Should-have)

As an ML engineer, I need to explore derived features that may improve the classifier beyond raw measurements.

**Acceptance Criteria:**

1. The notebook computes and evaluates at least these derived features:
   - `dist_over_t_local` = `min_exemplar_dist / t_local` (distance normalized by cluster spread)
   - `dist_over_t_global` = `min_exemplar_dist / t_global` (run-normalized distance)
   - `gap_to_next` = margin to next-best cluster (from merge evidence, if available)
   - `n_gates_passed` = count of 4-gate checks that pass (0–4)
   - `embedding_dist_percentile` = where this pair ranks among all pairs in the run
2. Each derived feature is evaluated against `min_exemplar_dist` alone (baseline) using AUC lift.

## Edge Cases

- **Single-face clusters**: `t_local = 0`, `diameter = 0`. Ratio features use guards (e.g., `0/0 → 1.0`).
- **Runs with only 1 merge**: Valid for training but extremely limited. Notebook warns when a run contributes fewer than 3 positive labels.
- **Manual label overrides**: Training DB may contain human corrections that flip labels. These override merge-log-derived labels.
- **Cross-run label conflicts**: Same person appears in two albums; clusters are "rejected" in each run but represent the same identity. This is out-of-scope (within-run labels only).

## Success Criteria

- **SC-001**: No duplicate runs in the training DataFrame (verified by hash-based deduplication).
- **SC-002**: Transitive closure produces no label contradictions (pair labeled both 0 and 1).
- **SC-003**: Face crops display correctly for at least 3 merged and 3 rejected pairs per run.
- **SC-004**: At least one derived feature achieves ≥ 0.02 AUC lift over raw `min_exemplar_dist`.
- **SC-005**: `t_global` leakage is documented and mitigated (either dropped or replaced with a per-pair variant).

## Non-Goals

- Training a production-ready classifier (that's `specs/011`).
- Modifying the merge pipeline or 4-gate system.
- Cross-album identity linking.
