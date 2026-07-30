# spec-078 — Metric-strip conformance sweep + arch guard

**Created**: 2026-06-05 · **Status**: Implemented · **Priority**: P2
**Predecessors**: spec-072 (`render_metric_strip` rail), registry audit (gap ④)
**Source**: user — migrate the 8 v2 modules still hand-writing `st.metric` onto the rail, then lock it.

## Problem
The metric-strip registry exists (`render_metric_strip` + `ColumnSpec`), but only
`face_detail_panel` uses it. 8 modules hand-write ~31 `st.metric` calls.

## What we build
- `face_cluster/views/_specs.py`: `ColumnSpec` gains optional `getter` (compute value from the
  object — for `len(...)`, PASS/FAIL, inline median) and `delta` (st.metric delta string).
- `face_cluster/views/metric_specs.py`: NEW — one `List[ColumnSpec]` per strip
  (CLUSTER / DEBUG / RUN_SUMMARY / FORCE_MERGE / QUALITY / OVERVIEW / FACE_COUNT / IMAGE_COUNT) +
  an `_age` display helper.
- Migrate 8 files: replace each `cN.metric(...)` block with `render_metric_strip(obj, COLS)`
  (count strips pass a `SimpleNamespace`).
- `app/face_clustering_v2/components/metric_strip.py`: pass `delta` through to `st.metric`.
- NEW `tests/architecture/test_metric_strip_conformance.py`: no bare `.metric(` in v2
  components/tabs except `metric_strip.py`.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | 0 bare `st.metric` in v2 outside `metric_strip.py` | arch test |
| 2 | each migrated strip renders the same labels/values | AppTest + screenshot |
| 3 | force_merge keeps PASS/FAIL + delta detail | render check |
| 4 | budapest B/D/E + F unaffected | pytest -m budapest |
