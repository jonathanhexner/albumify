# spec-026 Tasks

## Phase 1: DB Writer (alongside CSVs)
- [x] T1: Create `face_cluster/result_db.py` — raw SQL schema + writer functions
- [x] T2: `write_results_db()` function with all parameters
- [x] T3: Write faces + embeddings + face_scores tables from FaceRecord list
- [x] T4: Write cluster_assignments table — iteration 0 from base cluster_result
- [x] T5: Write clusters table — iteration 0 metrics
- [x] T6: Write merge_decisions table — one row per pair per iteration from merge_log
- [x] T7: Write cluster_assignments for final iteration (when merges occurred)
- [x] T8: Write clusters for final iteration (updated metrics)
- [x] T9: Write run_metadata from config + summary
- [x] T10: Call from `_export_for_analysis()` in cluster_people.py
- [x] T11: Test: 428 faces, 428 embeddings, 105 assignments, 13 merge_decisions, traceability query works

## Phase 2: FC App Loader
- [x] T12: Update `face_cluster/loader.py` — try DB first, fallback to CSVs
- [x] T13: Load faces + embeddings + cluster_assignments(iteration=0) → build ClusterResult
- [x] T14: Load merge_decisions → build merge_log list
- [x] T15: Load cluster_assignments(iteration=max) → build merged_cluster_result
- [x] T16: Fallback to CSV loading when DB doesn't exist
- [ ] T17: Test: FC app loads from DB, all tabs work (needs Playwright verification)

## Phase 3: Remove CSVs (later)
- [ ] T18: Remove CSV writing from export.py
- [ ] T19: Remove CSV loading from loader.py
- [ ] T20: Clean up old files

## Verification
- [ ] V1: Full traceability query works (face → embedding → cluster at each iteration)
- [ ] V2: Merge history queryable (which faces moved between clusters)
- [ ] V3: FC app Merge Analysis shows face crops from DB-loaded data
- [ ] V4: FC app Clusters (Base) and Clusters (Merged) both work from DB
