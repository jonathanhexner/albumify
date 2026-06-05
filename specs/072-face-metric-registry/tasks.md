# Tasks — Face-metric display registry (072)

## Phase 1 — Spec type + registry
- [ ] T001 `ColumnSpec` gains optional `help: Optional[str] = None` (`_specs.py`).
- [ ] T002 NEW `render_metric_strip(obj, columns, *, n_cols=5)` (`components/metric_strip.py`).
- [ ] T003 NEW `FACE_METRIC_COLUMNS: List[ColumnSpec]` in `face_metrics.py` (7 metrics incl. area %).

## Phase 2 — Canonical attrs on both row types
- [ ] T010 `FaceMetricRow` gains `area_ratio`; `list_faces` populates from `f.area_ratio`.
- [ ] T011 `FaceView` gains `area_ratio` field (compute from FaceRecord) + props `blur`/`yaw`/`pitch`/`roll`.

## Phase 3 — Refactor consumers onto the registry
- [ ] T020 `face_metrics_tab.py`: metric columns from `FACE_METRIC_COLUMNS` (keep thumbnail/id/status/cluster structural).
- [ ] T021 `face_detail_panel.py`: replace 5 `st.metric` with `render_metric_strip(view, FACE_METRIC_COLUMNS)`.

## Phase 4 — Tests + review
- [ ] T030 `test_metric_strip.py` (render + None→"—" + area % formatter); `FACE_METRIC_COLUMNS` covers area_ratio.
- [ ] T031 AppTest Face Analysis + Face Metrics → 0 exceptions; budapest Scenario D green.
- [ ] T032 `/code-review` → REVIEW.md; CHANGES_LOG; status → Implemented.
