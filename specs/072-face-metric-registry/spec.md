# spec-072 — Face-metric display registry (single source of truth)

**Created**: 2026-06-05
**Status**: Implemented
**Priority**: P2
**Predecessors**: spec-064 (Face Analysis), spec-069 (Face Metrics tab)
**Audit**: `docs/architecture/registry_audit_20260605.html` — gap ④ "no metric-strip registry" + gap ③ "face_metrics table inline".

---

## Problem

Face metrics are declared in 3+ places, all differently: the DB `faces` table, the
Face Analysis strip (`face_detail_panel.py` — 5 hardcoded `st.metric`), and the Face
Metrics table (`face_metrics_tab.py` — inline 9-col dataframe). Change/add a metric →
edit several files. (User also wants **Area %** added — currently only Area px shows.)

## Decision: reuse `ColumnSpec`, don't add `MetricSpec`

`ColumnSpec` (`face_cluster/views/_specs.py`) already has `field` / `label` /
`fallback_fields` / `formatter` / `display(row)` — everything a metric needs except a
tooltip. So we add one optional `help` field to `ColumnSpec` and declare the metrics
**once** as `FACE_METRIC_COLUMNS: List[ColumnSpec]`. A new `render_metric_strip`
renders that same list as `st.metric` widgets; the table keeps using `rows_to_records`.
(Same "one declaration, two renderers" rail as the existing table/param registries —
audit ①②③.)

## What we build

```
FACE_METRIC_COLUMNS  (one List[ColumnSpec], incl. area_pct)
   ├─ render_metric_strip(obj, cols)   → Face Analysis  (st.metric row)
   └─ rows_to_records(rows, cols)      → Face Metrics    (sortable table)
```

| Change | File |
|---|---|
| `ColumnSpec` gains optional `help: Optional[str]` | `face_cluster/views/_specs.py` |
| NEW `render_metric_strip(obj, columns, *, n_cols)` | `app/face_clustering_v2/components/metric_strip.py` |
| NEW `FACE_METRIC_COLUMNS` (blur, area px, **area %**, det_score, yaw, pitch, roll) | `face_cluster/views/face_metrics.py` |
| `FaceMetricRow` gains `area_ratio`; service populates it | `face_cluster/views/face_metrics.py` |
| `FaceView` gains `area_ratio` field + canonical-name props (`blur`/`yaw`/`pitch`/`roll`) so one list reads both row types | `face_cluster/views/face_view.py` |
| Face Metrics table builds metric cols from the registry (keeps thumbnail/id/status/cluster as structural cols) | `app/face_clustering_v2/tabs/face_metrics_tab.py` |
| Face Analysis strip → `render_metric_strip(view, FACE_METRIC_COLUMNS)` | `app/face_clustering_v2/components/face_detail_panel.py` |

Canonical attribute names (both row types expose them): `blur, area, area_ratio,
det_score, yaw, pitch, roll`. A column whose attribute is absent/None renders "—" /
is skipped — so the strip (FaceView has no det_score) just omits it.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | One `FACE_METRIC_COLUMNS` list; both table + strip consume it | grep + test |
| AC2 | **Area %** appears in Face Metrics table AND Face Analysis strip | render + test |
| AC3 | No hardcoded `st.metric(...)` left in `face_detail_panel.py`; no inline metric dict in `face_metrics_tab.py` | grep |
| AC4 | `render_metric_strip` unit test (incl. None → "—", area_ratio %) | pytest |
| AC5 | AppTest: Face Analysis + Face Metrics render, 0 exceptions | AppTest |
| AC6 | budapest Scenario D (Face Analysis) + the Face Metrics path stay green | pytest -m budapest -k "d or face" |

## Non-goals
- Albumify's metric strips (separate, larger; tracked in the audit).
- A `MetricSpec` type (explicitly rejected — reuse `ColumnSpec`).
- Arch-test that bans all `st.metric` (could follow once other strips migrate).
