# spec-086 tasks

Slice 1 of the SQL-tables + repository program. **Built with Approach B** (central
API DB, not per-run run_db — see spec.md DECISION). Order: write path before read path.

## Investigate / decide
- [x] T0 Confirmed the API does NOT write run_db (`execute_spec` does no persistence;
      `RunExporter` only in `run_pipeline()`/`face_cluster_export`, neither in the API
      `default_pipeline`). `fc_export_dir` is None on API runs. → chose Approach B.
- [x] T1 Confirmed the central API DB (`sim_bench.db`) is the lower-risk home: shared
      session, `create_all` auto-creates new tables, no golden-hash contract, no run-dir.

## Write path (API writes normalized tables)
- [x] T2 NEW tables `image_metric_rows` + `face_metric_rows` in
      `sim_bench/api/database/models.py` (FKs to pipeline_runs / people; auto-created
      by `init_db` → `create_all`). No manual migration needed for new tables.
- [x] T3 `pipeline_service._write_metric_tables(run_id, image_metrics, people)` — writes
      from the SAME `image_metrics` dict as the blob (dual-write), links each face to its
      Person.id. Called after people creation; wrapped in try/except so it can't fail a run.
- [x] T4 Verified on real data: backfill of 19 existing runs → 122 images / 340 faces /
      ~93–119 faces linked per run (`scripts/backfill_metric_tables.py`).

## Repository (read via SQL JOIN)
- [x] T5 `sim_bench/api/repositories/image_repository.py` —
      `get_images_for_person(run_id, person_id)`: JOIN on `face_metric_rows.person_id`,
      returns `list[ImageMetrics]` (all faces per image, for the popup).
- [x] T6 Unit tests (in T9 file): repository returns metrics, partitions selected, JOIN
      excludes images without the person.

## Endpoint swap (behind unchanged contract)
- [x] T7 `schemas/people.py::PersonImageResponse` now **inherits `ImageMetrics`** (+ keeps
      face_count/faces), so FastAPI no longer strips the fields.
- [x] T8 `people_service.get_person_images()` delegates to `ImageRepository`; merges metrics
      onto each image (path-normalized match); `faces` (person instances) preserved.
- [x] T9 `tests/api/test_image_repository.py` — 5 tests incl. equivalence with the blob
      (AC4) and response-model validation (AC2). Green.

## Frontend
- [x] T10 `people_browser.py` — added "Not selected" filter option + elif.

## Verify
- [x] T11 Playwright on the real Albumify app (API:8077 / Streamlit:8511):
      person detail shows "54 photos" + green Selected badges; Image Detail popup shows
      Selected + populated Quality scores (were blank). Screenshots: `PROBE_person_detail.png`,
      `PROBE_popup_boxes.png`.
- [x] T12 `tests/api/` → 27 passed (incl. new suite). Budapest v2 gate N/A: this is an
      Albumify + API-write-path change; the v2 face-clustering app path (`execute_spec`
      / FCAppRunner) is untouched and `_write_metric_tables` runs only in the API pipeline.
- [x] T13 Docs: RESOLVED banner on `docs/architecture/people_images_missing_metrics.html`;
      new tables + repository added to data-flow doc.
- [x] T14 `CHANGES_LOG.md` entry.
- [x] T15 `/code-review` → `REVIEW.md`: ACCEPT, 0 blockers, 4 follow-ups (TODO spec-086-1..4).
      Spec → Implemented.
