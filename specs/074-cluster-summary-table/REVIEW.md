# REVIEW — spec-074 All-clusters summary table

**Reviewed**: 2026-06-05 · **Scope**: spec-074 diff. **Verdict**: ✅ no High findings.

## Files
- `face_cluster/views/cluster_analysis.py` — `ClusterSummaryRow`, `CLUSTER_SUMMARY_COLUMNS`, `ClusterAnalysisService.cluster_summary()`
- `app/face_clustering_v2/components/cluster_summary_table.py` — NEW `render_cluster_summary`
- `app/face_clustering_v2/tabs/cluster_analysis_tab.py` — summary above the picker (cached, loop-safe click → drill-in)
- `tests/face_clustering/views/test_cluster_summary.py` — NEW (4)

## Checklist
| § | Finding |
|---|---|
| Correctness | nearest = min exemplar-to-exemplar cosine dist on normalized embeddings; one-pass O(n²) over clusters (≪ run sizes). Verified on real run (C1↔C7 0.418, sizes correct). Self excluded; `-1` when single cluster. ✅ |
| Reuse | `_embeddings_matrix` + result proxy reused; columns via `ColumnSpec` (audit ③ advanced). Raw `read()` keeps the table sortable. ✅ |
| Tests | 4 synthetic (1-row-per-cluster / size+diameter / nearest-matches-size / registry-read) + AppTest 0 exc + budapest Scenario B green. ✅ |
| Layering | `cluster_summary` Streamlit-free (numpy only); tab does no SQL/FS — calls service + component. ✅ |
| Perf | cached per run dir in session_state (no recompute per rerun); expander body still executes but reads the cache. ✅ |
| Edge cases | click loop avoided via `_summary_last_pick` sentinel; click → `_goto_cluster` reuses the spec-066 one-shot picker nav. ✅ |

## Follow-ups (Low)
- Floats show full precision in the table; a `column_config` number format (driven from the same specs) is optional polish.
- The Merge? flag already surfaces under-merge candidates — partial overlap with issue #1's "nearest pairs" ask.

**No High findings → Implemented.**
