# REVIEW — spec-075 Nearest cluster-pairs view

**2026-06-05** · scope: spec-075 diff · ✅ no High findings.

- `cluster_analysis.py`: `NearestPairRow` + `NEAREST_PAIR_COLUMNS` + `nearest_cluster_pairs()`; shared `_exemplar_matrices()` (refactored out of `cluster_summary`).
- `merged_clusters_tab.py`: `_render_nearest_pairs` expander (cached, ColumnSpec-driven), shown above the filter so it appears even when 0 merge_decisions match.

| § | Finding |
|---|---|
| Correctness | pairs sorted by min exemplar cosine dist; merge verdict joined by `frozenset((a,b))`; verified on real run (3 evaluated pairs show full why-not-merged; closest non-evaluated pairs follow). ✅ |
| Reuse | `_exemplar_matrices` shared with spec-074; `cached_cluster_service` reused. ✅ |
| Tests | 1 synthetic (sorted+structured) + AppTest 0 exc + e2e Scenario E. ✅ |
| Layering | service Streamlit-free (numpy); tab uses service + pandas render. ✅ |

No High → Implemented.
