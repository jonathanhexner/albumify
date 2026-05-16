# Tasks: Albumify E2E Acceptance Test (035)

## Design Notes

- **D1**: Use the *production* pipeline registry (`default_pipeline`) so the test will catch breakages from any step being added/reordered in `configs/pipeline.yaml`. Don't hand-build a minimal pipeline — that would defeat the purpose.
- **D2**: 5-image fixture with 2 distinct identities (3 + 2 photos) is the minimum that produces a meaningful cluster outcome. Going smaller risks "trivially correct" passes.
- **D3**: Use `tempfile.TemporaryDirectory` for `output_dir`; do not litter the workspace.
- **D4**: Allow the test to share `~/.sim_bench/sim_bench.db` (the universal feature cache) — first run is slow, subsequent runs are fast. Document this.
- **D5**: Assertions are on **structural** outcomes (non-empty clusters, non-NULL DB columns), not on **clustering quality** (which is non-deterministic enough to make flaky tests).

## Phase 1: Fixture

- [ ] T001 Pick 5 CC0 images of 2 distinct individuals (3 + 2). Document source in `tests/fixtures/face_clustering_e2e/README.md`.
- [ ] T002 Commit images to `tests/fixtures/face_clustering_e2e/*.jpg`.
- [ ] T003 Resize each image to ≤1024px on the long edge to keep test fast.

**Checkpoint**: `ls tests/fixtures/face_clustering_e2e/*.jpg` returns 5 files.

## Phase 2: Test

- [ ] T010 Create `tests/face_clustering/test_albumify_e2e.py` with `@pytest.mark.face_e2e` marker.
- [ ] T011 Test invokes the production pipeline via `sim_bench.api.services.pipeline_service.PipelineService` (or equivalent direct invocation), pointing at the fixture dir.
- [ ] T012 Assert `context.people_clusters` is non-empty after run.
- [ ] T013 Assert `RunStore(output_dir).image_detail(path)` returns ≥1 face with `is_core=True` for at least one fixture image.
- [ ] T014 Assert `RunStore.filter_decisions()` is non-empty (closes spec-032 acceptance on Albumify path).
- [ ] T015 Test cleans up `TemporaryDirectory` on exit.

**Checkpoint**: `.venv/Scripts/python -m pytest tests/face_clustering/test_albumify_e2e.py -v` passes in <60s.

## Phase 3: CI integration

- [ ] T020 Register the `face_e2e` marker in `pyproject.toml` so pytest doesn't warn.
- [ ] T021 Add a CI step that runs the e2e test on every PR (currently no CI configured; document the manual gate in WORKFLOW.md until CI lands).
- [ ] T022 Update `docs/guides/CODE_REVIEW_CHECKLIST.md` §5 — strike "at least one real E2E test" off the open list for this area.

**Checkpoint**: A regression that re-introduces the SIGHTING-061 class of bug (e.g. flip `blur_min` to honor yaml again without adding a blur step) fails this test loudly.
