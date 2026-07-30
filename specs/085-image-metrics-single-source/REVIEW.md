# spec-085 — Code Review

**Reviewed**: 2026-06-25 · **Reviewer**: Claude (code-review) · **Base**: working tree
**Scope**: spec-085 (ImageMetrics single source of truth, C-lite) + the spec-084 changes it builds on.

## Part 1 — How it works

**Goal:** kill the drift between the per-image record's three hand-maintained field lists so
no computed metric is silently dropped before the UI (the spec-084 blank-columns bug).

**Module inventory / data flow:**
```
PipelineContext (scattered per-path dicts)
  → pipeline_service._build_image_metrics()  → ImageMetrics(...).model_dump()   [producer, C-lite]
  → PipelineResult.image_metrics  (JSON column, {path: dict})                    [storage]
  → result_service._build_image_dict()  → ImageMetrics(**data).model_dump()      [projection]
  → routers/results.py  response_model=list[ImageMetrics]                        [contract gate]
  → api_client → ImageInfo → metrics table
```
`ImageMetrics` (`schemas/result.py`) is now the one shape; producer + projection both derive
from it. Bespoke extraction (context → values) stays as code in the producer.

**Dependency direction:** `services` → `schemas` (new import in result_service + pipeline_service).
Correct direction (service depends on schema, not vice-versa).

## Part 2 — Findings

### §1 Structure — pass
Focused change; `_build_image_dict` shrank from a 14-field hand list to 3 lines. `ImageMetrics`
grew to ~24 fields but has a single responsibility (the per-image contract) with a docstring.

### §2 Code quality — pass
No new bare `except`. `_build_image_dict` uses `ImageMetrics(**data)` (pydantic `extra="ignore"`
default) → unknown keys in old stored dicts are dropped on purpose (normalisation); acceptable.

### §3 Naming — pass
`schemas/result.py` / `ImageMetrics` consistent with siblings.

### §4 Layering & coupling — pass (improved)
**Single writer / no duplicated logic** is the whole point: the field set is defined once.
This *removes* a duplication smell rather than adding one.

### §5 Testability — pass (follow-up RESOLVED)
- Test inventory (spec-085): 3 unit/contract (`tests/api/test_image_metrics_contract.py`) +
  1 HTTP round-trip (`tests/api/test_results_endpoint_roundtrip.py`). Plus spec-084's
  `tests/api/test_results_metrics.py` (10) green. `tests/api/` = **27 passed**.
- **RESOLVED:** added `test_results_endpoint_roundtrip.py` — a FastAPI `TestClient` hits
  `GET /api/v1/results/{job}/images` and asserts `person_detected`/`best_frontal_score`/
  `roll_angles`/`filter_reason`/`quality_score`/`person_penalty` survive the `response_model`.
  Exercises the real route + coercion, not just the service functions.

### §6 Boundary contracts — pass-with-followup
- `ImageMetrics` is invoked at three points (producer, projection, response_model) — not dead.
- **Finding (low):** `ImageMetrics` has **no `extra="forbid"`**. With it, a producer typo /
  drift would fail loudly instead of silently ignoring. Deferred because it needs verification
  that no stored/coerced dict carries unexpected keys (old runs appear clean). This is the
  "forgot a field → silent None" gap the spec explicitly defers to a possible C-full registry.

### §7 Documentation — pass (blocker RESOLVED)
- **BLOCKER RESOLVED:** added `quality_scores` + `person_penalties` rows to
  `specs/034-pipeline-context-contract/spec.md` (next to `composite_scores`).
  `test_pipeline_context_fields_covered` now passes. spec-034 IS the canonical per-field
  contract doc for `PipelineContext`.
- **classes.html finding WITHDRAWN (over-flagged):** `docs/architecture/classes.html` does not
  track per-field score dicts (it never listed `composite_scores`) nor the API DTOs
  (`ImageMetrics`/`ImageInfo` absent). There is no field-level drift to fix there. The canonical
  docs for this change are spec-034 (PipelineContext, updated) and
  `results_image_data_flow.html` (API shape, updated with the RESOLVED banner).
- Present: spec.md ✓, tasks.md ✓, REVIEW.md ✓, CHANGES_LOG ✓, LEARNINGS (spec-104) ✓.

### §8 Risk — pass
- Backwards-compat: old runs lack the new keys → `Optional` defaults to None → "N/A" in UI.
  Documented; Reason/Quality/Penalty populate on new runs only.
- Hot-path: producer now constructs one Pydantic model per image, **once at run completion**
  (not per request). Negligible; no per-request cost added (projection already built a dict).

## Part 3 — Verdict

| Area | Verdict |
|---|---|
| §1 Structure | accept |
| §2 Code quality | accept |
| §3 Naming | accept |
| §4 Layering | accept |
| §5 Testability | accept (HTTP round-trip test added) |
| §6 Boundary contracts | accept w/ follow-up (`extra="forbid"` consideration) |
| §7 Documentation | accept (blocker fixed; classes.html finding withdrawn) |
| §8 Risk | accept |

**Handoff UNBLOCKED** (2026-06-25). Blocker resolved, §5 follow-up done; `tests/api/` +
context-contract = 27 passed. spec-085 may flip to Implemented.

### Follow-up tickets
1. ✅ Blocker — spec-034 rows added; context-contract test green.
2. ✅ §5 — `tests/api/test_results_endpoint_roundtrip.py` added.
3. ~~classes.html~~ — withdrawn (doc doesn't track these; canonical docs updated).
4. (Deferred, future spec) — C-full metric registry + `extra="forbid"` to make drift loud.
