# spec-069 — Tasks (re-scoped 2026-06-05)

**Spec**: [spec.md](spec.md) · **Investigation**: [INVESTIGATION.md](INVESTIGATION.md)
· **Design**: [DESIGN.html](DESIGN.html) · **Status**: Draft (awaiting go-ahead)

Primary deliverable: a **sortable per-face metrics table tab** (blur / area /
det_score + pose-when-available + assigned/unassigned status + thumbnail).
Buildable now — no pipeline change. Prototyped by `verify_run.py` /
`VERIFY_v2_budapest_20260605.html`.

## Build — Face Metrics tab (no pipeline dependency)

- [x] T01 — `face_cluster/views/face_metrics.py`: `FaceMetricsService` +
      `FaceMetricRow`. status derived (`all_faces − assigned`), no DB change.
      **DONE** (4 synthetic tests green).

- [x] T02 — `app/face_clustering_v2/tabs/face_metrics_tab.py`: sortable
      `st.dataframe` + `ImageColumn` thumbnail + metric cols + status filter +
      summary metrics. spec-068 telemetry. Wired into `main.py` (after Face
      Analysis). **DONE.**

- [x] T03 — `test_face_metrics_service_synthetic.py`: row-per-face, status
      partition (assigned+unassigned==total), metrics surfaced. **4 green.**

- [x] T04 — AppTest: telemetry test asserts `tab.done name=face_metrics`;
      real-data check vs `v2_budapest_20260605` → 0 exceptions,
      `n_faces=340 n_assigned=107 n_unassigned=233`. **DONE.**

- [ ] T05 — Close-out: CHANGES_LOG (added); docs/architecture HTML for the new
      service + tab; `/code-review` -> REVIEW.md; flip Status -> Implemented.
      **User visual sign-off pending** (open the tab, sort, confirm).

## Deferred — gate-reason breakdown (BLOCKED on SIGHTING-093)

- [ ] D01 — After SIGHTING-093 (persist `filter_decisions` + wire pose),
      add the per-gate reason column + funnel breakdown to the same tab.

## Open question for kickoff

Native `st.dataframe` sort (idiomatic, supports image column, but canvas — so
tested via service + AppTest, not browser cell-poking) vs. the custom HTML
table. **Recommend native `st.dataframe`.** Confirm at T02.
