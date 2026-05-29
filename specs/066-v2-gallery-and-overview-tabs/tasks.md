# Tasks — Gallery + Overview tabs (066)

## Phase 1 — OverviewService (~1 h)
- [ ] T001 NEW `face_cluster/views/overview.py` with `OverviewService` + `DashboardMetrics` dataclass. Constructor: `(history_repo, runs_dir)`. Method: `compute_dashboard() -> DashboardMetrics`.
- [ ] T002 NEW `test_overview_service_synthetic.py` (5 cases).
- [ ] T003 + 1 real-fixture case (opt-in slow) against user's real action_log.

## Phase 2 — Two tabs (~1.5 h)
- [ ] T010 NEW `app/face_clustering_v2/tabs/gallery_tab.py` (≤ 80 LOC). Iterates clusters; for each shows row of exemplar thumbnails + Open button.
- [ ] T011 NEW `app/face_clustering_v2/tabs/overview_tab.py` (≤ 80 LOC). 4-metric strip + 3 Plotly charts.
- [ ] T012 NEW component `app/face_clustering_v2/components/dashboard_charts.py` (~50 LOC; reusable bar + time-series helpers).
- [ ] T013 Wire both into `main.py`. Add arch test entries.

## Phase 3 — Baseline e2e Scenarios G + H (~1 h)
- [ ] T020 Add Scenario G + H to `test_v2_e2e_budapest_baseline.py`.

## Phase 4 — Parity close-out (~30 min)
- [ ] T030 Run `pytest -m budapest` → all 8 scenarios green.
- [ ] T031 Update `specs/042-fc-app-v2-tab-parity/spec.md` status → Implemented (this is the umbrella's last child).
- [ ] T032 `/code-review` → REVIEW.md. CHANGES_LOG. Spec-066 status → Implemented. Commit + push.

**Total: ~3-4 h.**
