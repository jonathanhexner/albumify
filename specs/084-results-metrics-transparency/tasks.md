# spec-084 tasks

## Phase 1 — Filter reason column
- [x] T1.1 `pipeline_service`: build `{item_id: reason}` from select_best decisions,
      pass into `_build_image_metrics`; add `"filter_reason"`.
- [x] T1.2 `result_service._build_image_dict`: pass `filter_reason` through.
- [x] T1.3 `api_client`: add `filter_reason` to `ImageInfo` + parse it.
- [x] T1.4 `metrics.py`: add a `Reason` column (after `Status`).

## Phase 2 — Composite breakdown
- [x] T2.1 `PipelineContext`: add `quality_scores` and `person_penalties`.
- [x] T2.2 `select_best._compute_composite_scores`: store quality + penalty per path.
- [x] T2.3 `pipeline_service._build_image_metrics`: expose `quality_score`, `person_penalty`.
- [x] T2.4 `result_service` + `api_client.ImageInfo`: pass through 2 fields.
- [x] T2.5 `metrics.py`: add `Quality` and `Penalty` columns.

## Phase 3 — Column tooltips
- [x] T3.1 `metrics.py`: add `METRIC_HELP` dict.
- [x] T3.2 generate column config from `METRIC_HELP` so every column has `help=`.

## Tests
- [x] T4.1 `tests/api/test_results_metrics.py` — 6 tests (AC1/AC2/AC3), green.
- [ ] T4.2 AppTest / manual: Results renders with new columns, 0 exceptions (AC4/5).
      **← user is testing this.**

## Close-out
- [ ] T5.1 Update `docs/architecture` classes HTML (new context fields + ImageInfo fields).
- [x] T5.2 `CHANGES_LOG.md` entry.
- [ ] T5.3 Run `/code-review` → `REVIEW.md`; resolve High findings.
- [ ] T5.4 Flip spec status In Progress → Implemented.
