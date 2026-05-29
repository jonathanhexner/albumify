# Tasks — Merged Clusters + Quality tabs (065)

## Phase 1 — Repository extension (~30 min)
- [ ] T001 Add `FilterDecisionCriteria` dataclass + `list_filter_decisions(criteria) -> list[FilterDecisionRow]` method to `cluster_analysis_repo.py`.
- [ ] T002 Extend `test_cluster_analysis_repo_synthetic.py`: +3 cases for criteria filters.

## Phase 2 — Two services (~1.5 h)
- [ ] T010 NEW `face_cluster/views/merged_clusters.py` with `MergedClustersService`.
- [ ] T011 NEW `face_cluster/views/quality.py` with `QualityService` + `QualitySummary` dataclass.
- [ ] T012 NEW `test_merged_clusters_service_synthetic.py` (4 cases).
- [ ] T013 NEW `test_quality_service_synthetic.py` (5 cases).
- [ ] T014 + 1 real-fixture case each (opt-in slow) against Budapest reference run.

## Phase 3 — Two tabs (~2 h)
- [ ] T020 NEW `app/face_clustering_v2/tabs/merged_clusters_tab.py` (≤ 80 LOC).
- [ ] T021 NEW `app/face_clustering_v2/tabs/quality_tab.py` (≤ 80 LOC).
- [ ] T022 NEW component `app/face_clustering_v2/components/quality_bar_chart.py` (~40 LOC; Plotly stacked bar).
- [ ] T023 Wire both into `main.py`.
- [ ] T024 Add arch test entries.

## Phase 4 — Baseline e2e Scenarios E + F (~1 h)
- [ ] T030 Add Scenario E (Merged Clusters) + Scenario F (Quality) to `test_v2_e2e_budapest_baseline.py`.

## Phase 5 — Close-out (~30 min)
- [ ] T040 Run `pytest -m budapest` → all 6 scenarios green.
- [ ] T041 `/code-review` → REVIEW.md. CHANGES_LOG. Status → Implemented. Commit + push.

**Total: ~4-5 h.**
