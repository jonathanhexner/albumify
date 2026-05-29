# Tasks — Recluster tab (063)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

## Phase 1 — Runner mode (~1 h)
- [ ] T001 Add `FCAppRunner.recluster(prior_run_dir, step_configs) -> FCAppRunResult` in `face_cluster/fc_app_runner.py`. ~10 LOC. Loads `face_records` from `RunStore(prior_run_dir).faces()`; pre-populates context; calls `self.run(...)`.
- [ ] T002 Unit test in `tests/face_clustering/test_fc_app_runner.py`: `test_recluster_skips_producer_chain` + `test_recluster_uses_prior_face_records`.

## Phase 2 — Service (~1.5 h)
- [ ] T010 NEW `face_cluster/views/recluster.py` with `ReclusterService` + `ReclusterResult` dataclass. Methods: `list_recent_runs(limit=20)`, `recluster(prior_run_dir, params) -> ReclusterResult`. Sync only.
- [ ] T011 NEW `tests/face_clustering/views/test_recluster_service_synthetic.py` — 6 cases per spec §"Tests".
- [ ] T012 + 1 real-fixture case (opt-in `slow`) reclustering the Budapest reference run; assert n_clusters in [12, 18].

## Phase 3 — Tab + components (~1.5 h)
- [ ] T020 NEW `app/face_clustering_v2/tabs/recluster_tab.py` (≤ 80 LOC). Three regions: prior-run picker, params editor (reuse `UI_SPEC`), "Run recluster" button + spinner.
- [ ] T021 Wire into `app/face_clustering_v2/main.py` — add 4th tab.
- [ ] T022 Add arch test entries: tab in the no-DB-no-FS-no-cfg.get scan list + LOC budget.

## Phase 4 — Baseline e2e Scenario C (~1 h)
- [ ] T030 Add `test_v2_scenario_c_recluster_reference_run` to `tests/face_clustering/test_v2_e2e_budapest_baseline.py`: load reference run via History, switch to Recluster tab, click Run, assert n_clusters in [12, 18].

## Phase 5 — Close-out (~30 min)
- [ ] T040 Run `pytest -m budapest tests/face_clustering/test_v2_e2e_budapest_baseline.py` → all 3 scenarios green.
- [ ] T041 `/code-review` → REVIEW.md.
- [ ] T042 CHANGES_LOG entry.
- [ ] T043 Spec status → Implemented. Commit + push.

**Total: ~4-6 h.**
