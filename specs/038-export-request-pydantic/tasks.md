# Tasks: ExportRequest Pydantic Bundle (038)

## Design Notes

- **D1**: Two-step migration. Phase 1 lands the new request type and dual API. Phase 2 (next PR) removes the legacy kwargs.
- **D2**: `arbitrary_types_allowed=True` is non-negotiable — `ClusterResult`, `FaceRecord`, numpy arrays all flow through.
- **D3**: Don't validate the contents of `faces` (already validated by `FaceRecord`); don't validate the `ClusterResult` dataclasses (cheap to construct, slow to deep-validate). Just hold them.

## Phase 1: Land ExportRequest, dual API

- [ ] T001 Add `ExportRequest(BaseModel)` to `face_cluster/run_exporter.py` with the 14 fields.
- [ ] T002 Refactor `RunExporter.export(self, request: Optional[ExportRequest] = None, **legacy_kwargs)`:
  - If `request` is None, build one from legacy_kwargs (emit `DeprecationWarning`).
  - Validate that the caller did not pass both.
  - All downstream code reads from `request.*`, not from kwargs.
- [ ] T003 Migrate `face_cluster/pipeline.py` — construct ExportRequest, pass it.
- [ ] T004 Migrate `sim_bench/pipeline/steps/face_cluster_export.py` — same.
- [ ] T005 Test `tests/face_clustering/test_export_request.py`:
  - constructs an ExportRequest with valid args → exports succeeds
  - typo'd field raises ValidationError
  - legacy-kwarg path still works, emits DeprecationWarning
  - mixed (both request + kwargs) raises TypeError

**Checkpoint**: `pytest tests/face_clustering/test_run_exporter.py tests/face_clustering/test_export_request.py` passes; spec-035 E2E test still green.

## Phase 2: Schedule legacy removal

- [ ] T010 File a TODO entry for "Remove legacy kwargs path from RunExporter.export (after one release)".
- [ ] T011 Add the kwarg path to the deprecation watch in `docs/project/SIGHTINGS.md` or similar tracker.

**Checkpoint**: No new callers using the kwarg path; once the watch period passes, removal is one commit.
