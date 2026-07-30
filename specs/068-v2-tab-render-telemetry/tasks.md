# spec-068 — Tasks

**Spec**: [spec.md](spec.md) · **Status**: Draft

Add driver-agnostic render telemetry to the v2 tabs so a single log line tells
us which tab ran, with what run dir, and what counts — the signal we lacked
during spec-067.

## Tasks

- [x] T01 — `app/face_clustering_v2/_telemetry.py`: `tab_start(name, run_dir)`,
      `tab_done(name, **counts)`, `tab_skipped(name, reason)`. Logger name
      `fc_app_v2.tabs`. ASCII, one line per event. (AC1, AC5)

- [x] T02 — Instrument the 7 tabs (one start + one done/skipped per return
      path). Counts per the spec table. (AC2)
      - run_tab (submit only), cluster_analysis_tab, face_analysis_tab,
        merged_clusters_tab, quality_tab, recluster_tab (start + submit),
        history_tab

- [x] T03 — `tests/face_clustering/test_v2_tab_telemetry.py` (AppTest +
      custom log handler, not caplog — app logging setup may alter
      propagation): (a) seeded run emits `tab.done name=quality` +
      `tab.done name=history`; (b) no-seed emits
      `tab.skipped ... reason=no_run_loaded`. **GREEN.** (AC3, AC4)

- [x] T04 — Confirm AC6: existing `test_v2_app_smoke.py` 8/8 green (no
      render behaviour change).

- [ ] T05 — Close-out: CHANGES_LOG entry (done); `/code-review` -> REVIEW.md;
      flip Status -> Implemented only after no High findings.

## Acceptance criteria -> task map

| AC | Task(s) |
|----|---------|
| AC1 helper exists | T01 |
| AC2 all 7 tabs instrumented | T02 |
| AC3 AppTest caplog sees tab.done | T03 |
| AC4 no-run path logs tab.skipped | T03 |
| AC5 ASCII single-line | T01, T02 |
| AC6 no render behaviour change | T04 |
