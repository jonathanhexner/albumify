# spec-065 — v2 Merged Clusters tab + v2 Quality tab (grouped)

**Created**: 2026-05-30
**Status**: Draft
**Priority**: P2 (both)
**Predecessors**: spec-042, spec-045, spec-059 (SQLAlchemy)
**Reference baseline**: must keep `tests/face_clustering/e2e_budapest/` green; adds Scenarios E + F.

Two tabs in one spec because both are **pure read-only viewers** over existing tables (`merge_decisions` / `filter_decisions`), both compose the existing `ClusterAnalysisRepository`, both share the same component shapes (filter bar + typed table + drill-down). Shipping them together avoids two near-identical PRDs.

---

## Problem

### Merged Clusters tab

The merger writes one row to `merge_decisions` for every candidate pair it considered (passed or rejected; `actually_merged` boolean). Operators today have no way to see why a merge happened (or didn't) — the data is in the DB but no UI. Legacy `merged_clusters_tab.py` was a 67-LOC viewer; v2 needs a typed equivalent.

### Quality tab

`filter_decisions` records every gate verdict per face (which gate fired, what value was measured, why rejected). Operators need this to tune profiles — "with profile_4, how many faces got rejected on blur?" "How many on pose?" Legacy app didn't have a dedicated Quality tab; we're adding it now.

## What we build

### Backend (shared)

`ClusterAnalysisRepository` already exposes `get_merge_log()`. Add **one new method**:
- `list_filter_decisions(criteria: FilterDecisionCriteria) -> list[FilterDecisionRow]` — composes `RunStore.filter_decisions()`; filters by `filter_name` (e.g. "blur") / `rejected` (bool) / `item_type` (face/image).

Two thin Services:
- `MergedClustersService.list_merge_decisions(criteria) -> list[MergeDecisionRow]` — passthrough over `repo.get_merge_log()` + filtering
- `QualityService.list_filter_decisions(criteria) -> list[FilterDecisionRow]` + `summary() -> QualitySummary` (per-gate pass/fail counts)

### Frontend

**`merged_clusters_tab.py`** (~70 LOC):
- Filter bar: `actually_merged` toggle + cluster_a/b dropdowns + iteration filter
- Typed dataframe (using spec-042's `ColumnSpec` pattern + `render_run_table` equivalent)
- Detail panel on row select: full 28-column row + "View clusters A / B" buttons that switch to Cluster Analysis with that cluster selected

**`quality_tab.py`** (~80 LOC):
- Summary strip (5 metrics: total faces, total decisions, n_rejected, top rejection gate, gate pass rate)
- Per-gate bar chart (Plotly) — x axis = gate names, y = pass / fail counts stacked
- Typed table of all rejections; filter by gate name + reason

### Tests

| Layer | File | Cases |
|---|---|---|
| Repository | extend `test_cluster_analysis_repo_synthetic.py` | +3: list_filter_decisions criteria filters work |
| Service synthetic | NEW `test_merged_clusters_service_synthetic.py` (4 cases) + NEW `test_quality_service_synthetic.py` (5 cases) | per-method coverage |
| Service real | + 1 each opt-in `slow` against Budapest reference run |
| Architecture | both tabs in the LOC + no-DB-FS-cfg.get arch scan |
| Baseline e2e | **Scenario E** (Merged Clusters): load reference run → tab → assert ≥ 1 row in table. **Scenario F** (Quality): same → assert per-gate chart has ≥ 1 bar |

## Locked decisions

1. **Grouping rationale: minimum duplication.** Both tabs are pure tables-with-filters. Splitting them = two near-identical PRDs / specs / commits. Grouping ships both in one logical unit; reviewer compares them side-by-side.
2. **No new compute.** Both tabs are read-only views of existing tables. No clustering math.
3. **Sync only.** spec-079 lesson.
4. **Reuse `render_run_table` pattern.** spec-042 H1 pilot's typed dataframe component handles selection — extend it, don't fork it.

## E2E contract (binding — part of AC5)

This spec OWNS **Scenarios E and F** in `tests/face_clustering/e2e_budapest/`.

### Scenario E — Merged Clusters

| Field | Value |
|---|---|
| Test file | `tests/face_clustering/e2e_budapest/test_scenario_e_merged_clusters.py` (NEW) |
| Click sequence | History → row containing `6437d335` → "Load into analysis tabs" → Merged Clusters tab |
| Concrete assertions | (1) tab visible; (2) ≥ 1 row in the merge_decisions dataframe (reference run has merge_decisions populated); (3) clicking the first row shows a detail panel; (4) detail panel exposes all expected fields: `cluster_a`, `cluster_b`, `actually_merged`, `exemplar_dist`, `support` (assert text presence in panel) |
| New constants in `conftest.py` | `EXPECTED_MERGE_DECISIONS_MIN_ROWS = 1` |
| README.md update | move Scenario E row from "Planned" to active |

### Scenario F — Quality

| Field | Value |
|---|---|
| Test file | `tests/face_clustering/e2e_budapest/test_scenario_f_quality.py` (NEW) |
| Click sequence | History → row containing `6437d335` → "Load into analysis tabs" → Quality tab |
| Concrete assertions | (1) tab visible; (2) summary strip ≥ 4 metric widgets; (3) per-gate bar chart Plotly element present (≥ 1 trace / 1 bar); (4) total rejected count visible AND ∈ `[220, 240]` (band around `340 - 107 = 233` — gate decisions on the reference run) |
| New constants in `conftest.py` | `EXPECTED_REJECTED_BAND = (220, 240)` (derived from `EXPECTED_N_FACES_TOTAL - EXPECTED_N_FACES_ASSIGNED ± 7`) |
| README.md update | move Scenario F row from "Planned" to active |

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `MergedClustersService` + `QualityService` typed sync APIs | grep + tests |
| AC2 | Both tab files ≤ 80 LOC; no SQL/FS/`cfg.get` | LOC + arch test |
| AC3 | 9 new synthetic service tests pass | pytest |
| AC4 | 2 real-fixture smoke tests pass on Budapest reference run | pytest -m slow |
| AC5 | Baseline e2e Scenarios E + F green | pytest -m budapest |
| AC6 | "View clusters A/B" buttons in Merged Clusters navigate to Cluster Analysis | covered by Scenario E |

## Effort estimate

**~4-5 hours total** (both tabs): Repository extension ~30 min, two services ~1.5 h, two tabs ~2 h, tests + baseline scenarios ~1 h.
