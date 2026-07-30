# REVIEW — spec-080 Navigation rework

**2026-06-05** · scope: spec-080 diff · ✅ no High findings.

## Files
- NEW `app/face_clustering_v2/_nav.py` — `render_nav` + `navigate_to` (pending-key pattern).
- `app/face_clustering_v2/main.py` — `st.tabs` → `render_nav([...])`.
- `components/face_grid.py`, `components/cluster_strip.py`, `tabs/face_metrics_tab.py` — "Open" → `navigate_to`.
- `tests/.../e2e_budapest/conftest.py` — `goto_page` helper; all 9 scenarios migrated off `get_by_role('tab')`.

## Checklist
| § | Finding |
|---|---|
| Correctness | `navigate_to` uses a PENDING flag applied before the radio is created (Streamlit forbids mutating a live widget key) — the cause of the first "no switch". Verified: Gallery "Open" → Cluster Analysis, cluster 1 selected. Face-Metrics row-select loop-safe via `_fm_last_pick`. ✅ |
| Perf | only the active page's render fn runs per rerun (was all 11 tab bodies) — lighter; removes the hidden-tab `img`/`stMetric` selector hazards. ✅ |
| Tests | budapest **B, D, E, G, H, I green** with the new nav. Two e2e timing races the single-page render EXPOSED (not app bugs) fixed: H waited for `.js-plotly-plot` mount; I waited for the 2nd pair-crop caption. ✅ |
| Pre-existing (not nav) | **C** — recluster picks the concurrent `v2_budapest_20260605b` as parent (SIGHTING). **F** — SIGHTING-092, the legacy reference run has 0 `filter_decisions` → no quality chart. Both failed before this change. |
| Deferred | st.navigation/URL multipage could replace the radio for nicer URLs; the radio is sufficient + controllable. |

**No High → Implemented.** Click-to-open works app-wide; the nav is the single source of the active view.
