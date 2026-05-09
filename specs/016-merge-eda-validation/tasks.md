# Tasks: Merge EDA & Training Data Validation

**Spec**: `specs/016-merge-eda-validation/spec.md`

## Design Notes

### Deduplication Strategy
Hash each run's merge decisions as `sorted((cluster_a, cluster_b, action))` → MD5. Group by hash, keep the run with the most metadata (crops, embeddings). Print dedup table.

### Transitivity Closure
Use union-find on merged pairs from the merge log. For all pairs in the feature DataFrame, check if both clusters belong to the same component. If yes → label=1. This catches the (B,C) case where B and C merged transitively through A.

### t_global Mitigation
Two options:
1. Drop `t_global` entirely and replace with `dist_over_t_global` (per-pair ratio)
2. Keep it but add a "run_id" group-aware cross-validation split (already supported by `split_strategy="by_run"` in TrainConfig)

Recommendation: do both — replace with ratio AND use by-run CV.

### Feature Engineering Priorities
1. Ratio features (normalize raw distances by cluster context)
2. Gate proximity features (how close to each gate's threshold)
3. Rank features (where does this pair sit among all pairs in the run)
4. Deferred groups E+I context features (margin to next-best cluster)

---

## Task Checklist

### Phase 1: Fix eda_merge_explore.ipynb (US1, US2)
- [x] Fix RUN_DIR path from `../results/` to `../../results/` | 2026-04-25
- [x] Add dedup logic: hash merge decisions, filter duplicate runs, print dedup summary | 2026-04-25
- [x] Add zero-merge run filter with warning | 2026-04-25
- [x] Implement transitivity closure for labels using union-find | 2026-04-25
- [x] Align candidate_threshold to pipeline default (0.45) with documented rationale | 2026-04-25
- [x] Add t_global leakage analysis: show it's constant per run, flag in output | 2026-04-25
- [x] Add feature correlation heatmap to identify redundant features | 2026-04-25
- [x] Verify crop display works for merged and rejected pairs | 2026-04-25

### Phase 2: Feature Engineering (US5)
- [x] Add derived ratio features: dist_over_t_local, dist_over_t_global | 2026-04-25
- [ ] Add gate proximity features: n_gates_passed, gate_margin_avg | 2026-04-25
- [x] Add rank features: embedding_dist_percentile (within-run rank) | 2026-04-25
- [x] Evaluate each derived feature vs baseline (min_exemplar_dist AUC) | 2026-04-25

### Phase 3: Fix eda_merge_ml.ipynb (US1, US3)
- [x] Apply same dedup + transitivity + threshold fixes as explore notebook | 2026-04-25
- [x] Add t_global leakage warning and mitigation (drop or use ratio) | 2026-04-25
- [x] Add by-run CV split alongside random stratified | 2026-04-25
- [x] Add correlation analysis section | 2026-04-25

### Phase 4: Negative Harvesting (US4)
- [x] Add section to explore notebook: harvest non-candidate pairs as negatives | 2026-04-25
- [x] Validate harvested negatives don't contradict transitivity closure | 2026-04-25
- [x] Tag harvested negatives with source='auto_negative' | 2026-04-25
