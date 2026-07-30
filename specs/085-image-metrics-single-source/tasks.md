# spec-085 tasks

- [x] T1 `schemas/result.py`: added the missing fields to `ImageMetrics`.
- [x] T2 `result_service._build_image_dict`: rebuilt from `ImageMetrics` (SSOT).
- [x] T2b (C-lite) `pipeline_service._build_image_metrics`: returns `ImageMetrics(...).model_dump()` — producer uses the one shape too. Parity test strengthened to `==`.
- [x] T3 `tests/api/test_image_metrics_contract.py` — 3 tests (AC1/AC2/AC3), green.
- [x] T4 `tests/api/` → 21 passed (spec-084 + bbox suites still green).
- [x] T5 Updated `docs/architecture/results_image_data_flow.html` (RESOLVED banner).
- [x] T6 `CHANGES_LOG.md` entry.
- [x] T7 `/code-review` → REVIEW.md. 1 blocker (spec-034 context rows) + §5 follow-up FIXED;
      classes.html finding withdrawn (over-flagged). 27 tests pass. Status → Implemented.
      Remaining: manual AC4 (table on real data) is the user's check; §6 `extra="forbid"`
      deferred to a future C-full spec.
- [x] T8 Blocker fix: `specs/034-pipeline-context-contract/spec.md` rows for quality_scores +
      person_penalties. T9 §5: `tests/api/test_results_endpoint_roundtrip.py`.
