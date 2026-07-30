# spec-084 — Results metrics transparency

**Created**: 2026-06-25 · **Status**: In Progress · **Priority**: P2
**Predecessors**: spec-081 (Image Analysis), SIGHTING-103 (Results perf)
**Source**: user — "store the [filter] reason; are there other metrics we're not storing; give a simple explanation for each metric (what it means, what range)." Tooltip presentation chosen by user.

## Problem
The Results metrics table shows a binary `Status` (Selected/Filtered) with **no reason**, and
several computed signals are dropped before they reach the UI. The user cannot tell *why* an
image was filtered, nor what any score means / its range.

Findings (read-only audit):
- The per-image **reason already exists** end-to-end: `select_best` emits a `StepDecision`
  (`select_best.py:313-331`) and the result payload already carries `step_decisions` with
  `reason` (`pipeline_service.py:260-265`). It is simply never **joined per image** into the table.
- **Computed but not stored**: `quality_score` + `person_penalty` — the two halves of
  `composite_score = quality_score + person_penalty` (`select_best.py:349-353`). These are the
  direct "why is the composite this value" breakdown and are discarded.
- **No metric glossary**: columns have no meaning/range help.

## What we build
1. **Reason column** — `pipeline_service._build_image_metrics` attaches `filter_reason` by
   looking up the `select_best` decision for that path (built once into a `{item_id: reason}`
   map). Flows through `result_service` → `ImageInfo.filter_reason` → a `Reason` column in
   `metrics.py` (shown only for filtered rows; selected rows show the positive reason too).
2. **Composite breakdown** — `_compute_composite_scores` stores `quality_score` and
   `person_penalty` into new `PipelineContext` dicts; `_build_image_metrics` exposes them;
   `ImageInfo` gains `quality_score` / `person_penalty`; table adds two columns.
3. **Column tooltips** — a single `METRIC_HELP: dict[str, str]` (label → "meaning; range") in
   `metrics.py`, wired via `st.column_config.*(help=...)` for every column. One source of truth.

Out of scope (logged as follow-ups, not built here): pagination / API-result caching
(SIGHTING-103 deferred items); surfacing the long tail of per-face sub-metrics (asymmetry, EAR,
smile components) — can be added later using the same `METRIC_HELP` mechanism.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | Filtered rows show a non-empty reason matching the `select_best` decision | unit test (service join) |
| 2 | `quality_score + person_penalty == composite_score` (within float tol) and both are exposed per image | unit test |
| 3 | Every metrics-table column has a `help=` tooltip from `METRIC_HELP` | unit test (no column lacks help) |
| 4 | Results page renders with the new columns, 0 exceptions, on real data | AppTest / manual |
| 5 | No regression: existing metrics still present; perf cache (SIGHTING-103) intact | manual |

## Notes
- Reason data is already in the payload — phase 1 is a join, not new computation.
- `quality_score`/`person_penalty` require new `PipelineContext` fields → update
  `docs/architecture` classes HTML per the documentation mandate.
- Tooltips use `st.column_config(..., help=)`; the `ImageColumn` thumbnail and text columns all
  accept `help`.
