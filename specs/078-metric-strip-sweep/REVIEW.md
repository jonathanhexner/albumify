# REVIEW — spec-078 Metric-strip conformance sweep

**2026-06-05** · scope: spec-078 diff · ✅ no High findings.

## Files
- `face_cluster/views/_specs.py` — `ColumnSpec.getter` + `.delta` (compute value / metric delta).
- NEW `face_cluster/views/metric_specs.py` — 8 strip registries + `_age`/`_avg_clusters` helpers.
- `app/face_clustering_v2/components/metric_strip.py` — pass `delta` to `st.metric`.
- Migrated 8: components/{cluster_metrics, cluster_debug, force_merge, run_detail} + tabs/{quality, overview, face_metrics, images}.
- NEW `tests/architecture/test_metric_strip_conformance.py`.

## Checklist
| § | Finding |
|---|---|
| Conformance | arch test green — 0 bare `st.metric` in v2 outside `metric_strip.py` (was 31 across 8 files). ✅ |
| Faithfulness | values verified vs originals: `len()` getters (Exemplars/Outliers/Bridge), PASS/FAIL + delta detail (force_merge), inline median (`18.6 (med 15)`), age (`_age` moved to metric_specs). ✅ |
| Reuse | one `render_metric_strip` rail; getters duck-type the source object → no view-type imports in metric_specs; module Streamlit-free. ✅ |
| Behavior deltas (Low) | run_detail now always shows 5 (None→"—") instead of conditionally hiding empty groups — clearer, harmless. overview `_age` relocated from the tab. |
| Tests | strip-value checks + AppTest 11 tabs 0 exc + 41-test arch/unit batch + budapest B/D/E. ✅ |

**No High → Implemented.** v2 is now fully on the registry rail (tables + metric strips + DB + config), locked by arch tests.
