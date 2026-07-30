# REVIEW — spec-072 Face-metric display registry

**Reviewed**: 2026-06-05 · **Scope**: this spec's diff only (working tree has heavy concurrent work; reviewed spec-072 files in isolation). **Verdict**: ✅ no High-severity findings.

## Files changed
- `face_cluster/views/_specs.py` — `ColumnSpec.help` (optional, default None; non-breaking)
- `app/face_clustering_v2/components/metric_strip.py` — NEW `render_metric_strip`
- `face_cluster/views/face_metrics.py` — `FACE_METRIC_COLUMNS`, `FaceMetricRow.area_ratio` + `.area_pct`
- `face_cluster/views/face_view.py` — `area_ratio`/`det_score` fields + canonical-name props (`blur`/`yaw`/`pitch`/`roll`/`area_pct`)
- `app/face_clustering_v2/components/face_detail_panel.py` — strip → registry
- `app/face_clustering_v2/tabs/face_metrics_tab.py` — table metric cols → registry
- `tests/face_clustering/views/test_face_metric_registry.py` — NEW (6 cases)

## Checklist walk

| § | Area | Finding |
|---|---|---|
| 1 | **Correctness** | Table uses `spec.read` (raw numeric) → sorting preserved; strip uses `spec.display` (string). `0.0` not dropped (verified by test). None → "—". ✅ |
| 2 | **Reuse / simplification** | Reused `ColumnSpec` instead of a near-duplicate `MetricSpec` (audit's own conclusion). Net **−~25 LOC** of hardcoded metric rendering. ✅ |
| 3 | **Tests** | 6 unit (labels, area_pct derive, raw-numeric, None→empty, zero-kept, help) + AppTest 0 exc + budapest Scenario D green. ✅ |
| 4 | **Layering** | Registry + service stay Streamlit-free; renderer is the only Streamlit piece. `test_v2_layering` unaffected (strip is a component, not a tab). ✅ |
| 5 | **Schema/contracts** | No DB change. `FaceView`/`FaceMetricRow` are view dataclasses; `area_ratio` already on the faces table + `FaceRecord`. ✅ |
| 6 | **Naming/idiom** | Canonical attr names (`blur`/`yaw`/…) shared via properties; matches `FaceMetricRow`. ✅ |
| 7 | **Docs** | spec.md + tasks.md + this REVIEW; audit HTML references gap ③④. No architecture HTML touched (no schema change). ✅ |
| 8 | **Risk/edge cases** | Legacy run with NaN pose → props return the NaN/None; strip shows the value or "—". Table mixes `ColumnSpec` (structural) implicitly — only metric cols come from registry, thumbnail/id/status/cluster stay explicit. ✅ |

## Notes / follow-ups (Low)
- Table shows raw floats (e.g. blur `1204.0`); a later polish could add `st.column_config.NumberColumn` formats driven from the same specs. Not blocking.
- Other metric strips (cluster_metrics, overview, quality, Albumify) still hardcoded — migrating them onto `render_metric_strip` is the natural follow-up (tracked in the audit, gap ④).

**No High findings → cleared for Implemented.**
