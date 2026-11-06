# spec-116 — Code review

**Status:** Blocked, 2 High findings · **Date:** 2026-10-31 · **Base:** `main` (changes uncommitted on `spec/116-one-run-path`) · **Checklist:** `docs/guides/CODE_REVIEW_CHECKLIST.md`

Self-review by the implementing agent, not an independent one.

## Part 1: How it works

```
Configure & Run -> POST /pipeline/run {runner}
  PipelineService.start_pipeline   RunSpec.build, validate, JobService.create (spend cap), run row
  PipelineService.execute_pipeline -> JobService.run
     stage    InputStore copies, manifest key -> original
     run      LocalRunner child process | ModalRunner
              JobCurator -> album.json, bundle.json, run_output.json, progress.json
     load     ResultsLoader -> control DB, displays, crops
     on_loaded  RunOutput.remap(PathMap) -> RunPersister -> commit with run status
Albums -> Results / People & Faces / Explore (regular tables; RunEmbeddings, then cache)
```

Layering: `app` -> `api` -> `serving` / `control` -> `pipeline`. Exception, recorded: `serving/run_output.py` imports `api.schemas.result` and, lazily, `api.services.people_service`. `pipeline_service` imports `routers.jobs` lazily for its default `JobService`.

## Part 2: Findings

### High

| # | Finding | Evidence | Status |
|---|---|---|---|
| H1 | **The L4 path has no real-data proof.** Checklist: "a new public API with no E2E test exercising it on real data". Both attempts stayed in Modal's queue, so `spec` reaching the worker, `run_output.json` copy-back and the remap of real `/inputs/` paths ran only against a fake | updates.md 2026-10-31; R5 R6 R9 | Open: needs an L4 run |
| H2 | **R13 fails its check.** 768 px is not close enough at production settings, and with the sharpness gate off people still split 11 vs 8. Every run now uses 768 px, so this changes results for every album | `reports/2026-10-30_spec116_r13_fullsize_vs_768/`; SIGHTING-144 | Open: owner decision |

### Medium

| # | Finding | Ticket |
|---|---|---|
| M1 | The PC must stay online for a whole L4 run (`runners.py`, `run_job`). A drop stops the app and raises past the `except`; a drop during a paid run would lose it | SIGHTING-145 |
| M2 | A queued L4 run reads "Running" | SIGHTING-143 |
| M3 | Behaviour changes to confirm: straightened picks show as the original; photos in subfolders are not run; Re-run left the Jobs page; a failed people or metric write now fails the run (it was a warning) | spec §6 notes, updates.md |
| M4 | `SpecBuilder.calc` now applies `people_clustering`, so regression runs may shift from their baselines. Not measured | TODO |

### Low

| # | Finding | Ticket |
|---|---|---|
| L1 | `PeopleService.create_from_clusters` is now used by tests only; `JobProgress` and the websocket queue are never fed | TODO |
| L2 | `run_output.py` 320 LOC (justified in its docstring); `job_service.py` 327 and `job_store.py` 309 passed 300 in this spec with no such note | TODO |
| L3 | `start_pipeline` has 6 parameters (was 5) | Accepted: mirrors the request |
| L4 | Albums shows 7 old runs stuck at `running`, made before spec-116, with no job to end them | TODO |
| L5 | `reports/` `report.html` + `summary.md` for R13 not written: the agent harness refused report files | updates.md |

## Part 3: Verdict per area

| § | Area | Verdict | Note |
|---|---|---|---|
| 1 | Structure | pass-with-follow-up | L1, L2, L3 |
| 2 | Code quality | pass | The two swallowed writes became failures (R8) |
| 3 | Naming and packages | pass | `__all__` on every new module |
| 4 | Layering and coupling | pass-with-follow-up | One writer for result rows; `serving` -> `api` schema import recorded in Part 1 |
| 5 | Testability | **fail** | H1. Local path covered by real E2E and real Budapest runs; L4 only faked. Fake Modal justified: no spend in tests |
| 6 | Boundary contracts | pass | `RunRow` forbids extras; `RunSpec` validated before anything is created; `RunOutput` is a dataclass read with `cls(**json)`, so unknown keys raise |
| 7 | Documentation | pass-with-follow-up | `classes.html`, `data_flow.html` updated; no DB column changed; L5 |
| 8 | Risk register | **fail** | H2, M1 |

**Test inventory:** 1,739 passed + 2 skipped full suite (0 failed); 11 Playwright on the real app with Modal faked; 11 Budapest V2 gate; 3 real local Budapest pipelines plus diagnostic. The 2 skips were not investigated individually.

**Failure walk-through:** staged path stored: `test_every_path_is_the_original`, `test_rows_name_the_originals_on_both_machines`. Boxes on the wrong scale: `test_pixel_boxes_scale_and_ratio_boxes_stay`. Job path ignores clustering: `test_people_clustering_reaches_the_steps`. Partial rows: `test_a_failed_write_leaves_no_rows`. Run stuck forever: `test_a_process_that_died_ends_failed`. Spend cap bypass: `test_l4_over_cap_is_refused`.

**Blockers:** H1 and H2 block `Implemented`. Decision needed on each: fix, waive (recorded here), or keep at In Progress. Spec stays In Progress.
