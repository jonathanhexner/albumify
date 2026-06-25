# spec-086 — Code Review

**Reviewed**: 2026-06-25 · **Reviewer**: Claude (code-review) · **Verdict: ACCEPT** (0 blockers, 4 follow-ups)

## Part 1 — How it works

**Goal**: stop the People & Faces image list from being starved of per-image metrics
(empty "Selected" filter, no face boxes). Replace the JSON-blob stitch with normalized
SQL tables + a repository, behind the unchanged `ImageMetrics` contract.

**Module inventory (spec-086 surface only):**

| File | Change | Responsibility |
|---|---|---|
| `api/database/models.py` | +`ImageMetricRow`, +`FaceMetricRow` | normalized per-image / per-face tables (central DB) |
| `api/services/pipeline_service.py` | +`_write_metric_tables` | write path (dual-write from the blob's dict) |
| `api/repositories/image_repository.py` | NEW | read path — `get_images_for_person` SQL JOIN → `list[ImageMetrics]` |
| `api/services/people_service.py` | `get_person_images` rewrite | delegate to repository, merge onto each image |
| `api/schemas/people.py` | `PersonImageResponse(ImageMetrics)` | stop the response_model stripping fields |
| `app/streamlit/components/people_browser.py` | +"Not selected" filter | UI |

**Data flow**: pipeline run → `_build_image_metrics` dict → (blob `image_metrics`) **and**
`_write_metric_tables` → `image_metric_rows`/`face_metric_rows` (faces linked to `person_id`).
Read: `get_person_images` → `ImageRepository.get_images_for_person` → JOIN on
`face_metric_rows.person_id` → `ImageMetrics` → `PersonImageResponse` → UI.

**Dependency direction**: schema/models (low) ← repository ← service ← UI. No reverse imports.

## Part 2 — Findings

**§1 Structure** — `pass`. New modules single-purpose; `image_repository.py` ~130 LOC one job.
No dead code. Layering respected.

**§2 Code quality** — `pass-with-followup`. `_write_metric_tables` is wrapped in
`try/except → logger.warning` (pipeline_service.py). Defensible (dual-write must not fail a
run; mirrors the adjacent people-creation guard) but it means a silent feature failure.
→ TODO: surface metric-write failures (counter/sighting) rather than warn-only.

**§3 Naming / package** — `pass`. New `repositories/` subpackage; `*MetricRow` consistent with
existing `*Row` naming. `__all__` not used elsewhere in this layer — consistent.

**§4 Layering & coupling** — `pass-with-followup`.
- *Single writer per state*: the blob `image_metrics` and the two tables are a **documented
  dual-write from the same dict** (the contract that prevents drift), enforced by
  `test_equivalence_with_blob`. Acceptable; named in code + db_global.html.
- Local import of `ImageRepository` inside `get_person_images` (defensive) → TODO: lift to
  module scope (no real cycle).

**§5 Testability** — `pass` (load-bearing section). Test inventory:
- Unit: 5 (`tests/api/test_image_repository.py`) — write-links-person, repo-returns-metrics,
  selected/not-selected partition, endpoint-carries-metrics+validates, **blob-equivalence**.
- E2E/real-data: live HTTP against the running API (54 imgs / 7 selected / 47 not, bboxes
  present) + Playwright on the real Albumify app (Selected badges; populated popup).
- *Failure-mode walk-through*: bug class = "endpoint silently drops per-image metrics" →
  `test_endpoint_carries_metrics_and_validates` fails if `PersonImageResponse` narrows again;
  `test_equivalence_with_blob` fails if the two writers diverge.

**§6 Boundary contracts** — `pass-with-followup`. `PersonImageResponse` now inherits
`ImageMetrics`; `extra="forbid"` is **not** set — consistent with spec-085's explicitly
deferred C-full item (same model family). The new tables are SQLAlchemy ORM (not DataFrame
I/O), like the other central-DB tables, so no Pandera. → tracked under spec-085's open item.

**§7 Documentation** — `pass-with-followup`. Present: spec.md, tasks.md, REVIEW.md,
CHANGES_LOG, `db_global.html` (both new tables + writer row + TOC — the DB-column mandate
item), `people_images_missing_metrics.html` (RESOLVED banner). → TODO: add `ImageRepository`
+ `PersonImageResponse` to `classes.html` and a read-side note to `data_flow.html`.

**§8 Risk register** — `pass`.
- *Migration story* (the high-severity criterion): new tables auto-created by
  `create_all`; existing runs handled by `scripts/backfill_metric_tables.py`; and
  `get_person_images` **falls back gracefully** (image row without a metric match still
  returns, fields default) so un-backfilled runs are no worse than before. Not a blocker.
- *Backwards-compat*: old runs not backfilled show pre-fix behavior until backfilled — no
  regression. Documented.
- *Hot-path perf*: write adds ~1 insert/image + 1/face (~460/run) — negligible vs minutes of
  pipeline; read is 2–3 indexed queries.

## Part 3 — Verdict & tickets

**ACCEPT — no §1–§7 `fail`.** The one high-severity *criterion* in scope (DB schema change →
migration story) is satisfied (auto-create + backfill + graceful fallback).

Follow-up tickets (TODO.md):
1. Lift `ImageRepository` import to module scope in `people_service`.
2. Surface `_write_metric_tables` failures (don't warn-only).
3. Add `ImageRepository`/`PersonImageResponse` to `classes.html`; read-side note in `data_flow.html`.
4. (Tracked w/ spec-085) `extra="forbid"` on the `ImageMetrics` family.

None block handoff. Spec may move to **Implemented**.
