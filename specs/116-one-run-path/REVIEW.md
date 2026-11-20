# spec-116 — Code review

**Status:** Accepted, no High open (2026-11-15), on the merge plan below: spec-116 reaches `main` only together with spec-117, in one PR from `spec/117-neon-r2` (`b95e485`). Not `Implemented` yet: the full suite on `b95e485` (see updates.md 2026-11-15, pending) and the H2 decision copied into spec.md / updates.md · **Date:** 2026-11-15 (refresh of 2026-11-14, 2026-10-31) · **Base:** `main...spec/116-one-run-path` (`4cb2adf`..`edc7678`), as merged into `spec/117-neon-r2` `b95e485` · **Checklist:** `docs/guides/CODE_REVIEW_CHECKLIST.md`

Self-review by an agent, not an independent one. The three-dot diff also carries spec-115 commits (`7c9d9f2`..`01192fe`, reviewed under spec-115); this review covers `4cb2adf`, `7a29c4a`, `edc7678`.

## Changes since 2026-11-14

- **Merge plan (decides H1):** spec-116 is not merged alone. It goes to `main` in the same PR as spec-117 (`spec/117-neon-r2`, now `b95e485` after merging `spec/117-m3-fixes`). The spec-116-only transport (ephemeral Modal app + copy-back) never reaches `main`: spec-117 replaced it, and spec-117's path is proven on real data. H1 is **superseded**.
- **H2 decided by owner** (`f11f595`, SIGHTING-144 now DECIDED: keep 768 px for every run; any `min_sharpness` retune is a separate follow-up).
- **M5 decided by owner** (`f11f595`, SIGHTING-154: a follow-up, not a merge blocker).
- **M1, M2 fixed by spec-117 code** in the joint merge (deployed app + `spawn`; "waiting for gpu" stage). Their sightings (145, 143) are still marked OPEN and should be closed.
- The spec-117 fix commits (`10358db`, `7ad54ac`, `151b79d`) touch no spec-116 code; nothing new found in them for this spec (spec-117 REVIEW.md has their findings).
- **Evidence added:** Budapest V2 gate 11/11 on `spec/117-m3-fixes` (contains all spec-116 code), which closes the "pending" gate from 2026-11-14. Neon parity 315/316 (one test timeout, fixed in `151b79d`). Full suite on `b95e485`: see updates.md 2026-11-15 (pending).

## Part 1: How it works

```
Configure & Run -> POST /pipeline/run {runner}
  PipelineService.start_pipeline   RunSpec.build, validate, JobService.create (spend cap), run row
  PipelineService.execute_pipeline -> JobService.run
     stage    InputStore copies, manifest key -> original
     run      LocalRunner child process | ModalRunner (ephemeral app, run_app.run() held open)
              JobCurator -> album.json, bundle.json, run_output.json, progress.json
     load     ResultsLoader -> control DB, displays, crops   (L4: run_output.json copied back)
     on_loaded  RunOutput.remap(PathMap) -> RunPersister -> commit with run status
Albums -> Results / People & Faces / Explore (regular tables; RunEmbeddings, then cache)

select_best (7a29c4a):  scene -1 -> _select_unclustered: each photo scored alone, kept iff >= min_score_threshold
classify    (7a29c4a):  a -1 photo is never near_dup
```

spec-117 replaces the transport row: deployed app + `spawn`, and `WorkerPersister` on the worker calls the same `RunOutput.from_json(...).remap(PathMap...)` -> `RunPersister` (`sim_bench/serving/worker_persister.py:71-74`). No copy-back. In the joint merge this is the only GPU path that reaches `main`.

Layering: `app` -> `api` -> `serving` / `control` -> `pipeline`. Exception, recorded: `serving/run_output.py` imports `api.schemas.result` and, lazily, `api.services.people_service`. `pipeline_service` imports `routers.jobs` lazily for its default `JobService`. `7a29c4a` adds `album/normalizer.py` -> `pipeline.clustering_labels` (data -> lower layer, fine).

## Part 2: Findings

### High

| # | Finding | Evidence | Status |
|---|---|---|---|
| H1 | **The spec-116 L4 transport has no real-data proof.** Proven via spec-117 on real Slovakia26: `RunSpec` reaches the worker (`f7d8ba85`: 659 photos, 329 picks, 7 people); `RunOutput` round trip + `PathMap.remap` give original paths, 0 staged (R5); `RunPersister` rows open in Albums, Results, People, Explore (R6); cached re-run `fd52013c` (R9). Not proven: this branch's `ModalVolumeRemote.run_job` with the ephemeral `run_app.run()` (`sim_bench/serving/runners.py:149-161` on `spec/116-one-run-path`), copy-back of `run_output.json`, the "copying back" stage (R7), the Volume cache (R9 as written), cancel of a queued L4 run | `specs/117-neon-r2-one-database/updates.md` 2026-11-14 M2; SIGHTINGS 153, 154 | **Superseded** (2026-11-15): merge plan above. The unproven code is replaced by spec-117 (`runners.py` `ModalRunner` / `DetachedModalRemote`) before it reaches `main`. Re-opens if 116 is ever merged alone. Cancel of a queued L4 run stays a spec-117 gate (R9) |
| H2 | **R13 fails its check.** 768 px is not close enough at production settings (8 vs 96 photos reach the album, 11 vs 8 people); every run uses 768 px | `reports/2026-10-30_spec116_r13_fullsize_vs_768/`; SIGHTING-144 | **Decided by owner** 2026-11-14, `f11f595`: keep 768 px for every run; the threshold now means "sharp at 768 px"; a `filter_quality.min_sharpness` retune is a separate follow-up (`docs/project/SIGHTINGS.md` SIGHTING-144 status). Recorded in SIGHTINGS.md only: copy into spec.md R13 and updates.md before `Implemented` |

### Medium

| # | Finding | Ticket | Status |
|---|---|---|---|
| M1 | The PC must stay online for a whole L4 run: `runners.py:155` holds `run_app.run()` open | SIGHTING-145 (still OPEN) | **Fixed by spec-117** in the joint merge: deployed app + `spawn`, the job row is followed (`sim_bench/serving/runners.py:207-214` at `b95e485`); `f7d8ba85` finished with the PC's API/UI stopped; a lost link no longer fails the run (spec-117 H1, `10358db`, `tests/serving/test_detached_modal_status.py`). Close SIGHTING-145 |
| M2 | A queued L4 run reads "Running": `runners.py:229` wrote `running` before the worker starts | SIGHTING-143 (still OPEN) | **Fixed by spec-117** in the joint merge: stage "waiting for gpu" until the worker's first write (`sim_bench/serving/runners.py:205-206` at `b95e485`). No test names this; close SIGHTING-143 with one, or on the R9 queued-cancel run |
| M3 | Behaviour changes to confirm: straightened picks show as the original; photos in subfolders are not run; Re-run left the Jobs page; a failed people or metric write fails the run; no-scene photos have no per-scene cap (Slovakia26 no-scene picks 24 -> 85) | spec §6 notes, updates.md, SIGHTING-146 | Still open (owner to confirm; not a merge blocker) |
| M4 | Regression baselines not re-measured: `people_clustering` K=51 / blur_min 73.3 (`configs/pipeline.yaml:205,211`) and the `7a29c4a` selection change shift picks and people on every album | TODO | Partly: Budapest gate 11/11 on the fix branch holds its baselines; album-level (picks/people) baselines still not re-measured. Follow-up |
| M5 | Re-running the same album on the GPU gives different picks (329 vs 331, 3 photos differ); spec-110 AC4 not met | SIGHTING-154 | **Decided by owner** 2026-11-14, `f11f595`: follow-up, does not block the merge (SIGHTING-154 status line) |
| M6 | Nothing ties `pipeline.yaml` `people_clustering` to the validated `profile_4.json`; the K=5 drift can recur silently | SIGHTING-147 "Prevention (open)" | Still open (follow-up) |

### Low

| # | Finding | Ticket |
|---|---|---|
| L1 | `PeopleService.create_from_clusters` is now used by tests only; `JobProgress` and the websocket queue are never fed | TODO |
| L2 | `run_output.py` 320 LOC (justified in its docstring); `job_service.py` 327 and `job_store.py` 309 passed 300 in this spec with no note; `select_best.py` 653 LOC grew by 37 in `7a29c4a` | TODO |
| L3 | `start_pipeline` has 6 parameters (was 5) | Accepted: mirrors the request |
| L4 | Albums shows 7 old runs stuck at `running`, made before spec-116, with no job to end them | TODO |
| L5 | `reports/` `report.html` + `summary.md` for R13 not written | updates.md |
| L6 | `min_score_threshold` default `0.4` is repeated at `select_best.py:346` and `:408` | TODO |
| L7 | Wrong per-photo reasons: `build_reason_by_path` (`run_output.py:26-43`) keeps `detect_persons`' reason over `filter_quality`'s; "Best in cluster"/"Rank N" use the unsorted list (`select_best.py:373`) | SIGHTING-149, SIGHTING-150 (OPEN, planned in spec-118) |
| L8 | `_merge_decisions_panel.py:702+` loop body keeps a 16-space indent under an 8-space `for`; `tests/face_clustering/test_streamlit_e2e.py` is stale | TODO |
| L9 | **New.** SIGHTING-143 and SIGHTING-145 are still OPEN although the merged code fixes them (M1, M2) | Close in SIGHTINGS.md |

## Part 3: Verdict per area

| § | Area | Verdict | Note |
|---|---|---|---|
| 1 | Structure | pass-with-follow-up | L1, L2, L3, L8 |
| 2 | Code quality | pass-with-follow-up | Swallowed writes became failures (R8); L6 |
| 3 | Naming and packages | pass | `__all__` on every new module |
| 4 | Layering and coupling | pass-with-follow-up | One writer for result rows (`RunPersister`, also the worker's writer in 117); `serving` -> `api` import recorded |
| 5 | Testability | pass-with-follow-up | was fail: H1 superseded by the joint merge; the path that reaches `main` is the spec-117 one, proven on real data and now tested against the real Modal status (`tests/serving/test_detached_modal_status.py`). Still no test for K=51 (M6) or re-run stability (M5, owner follow-up) |
| 6 | Boundary contracts | pass | `RunRow` forbids extras; `RunSpec` validated before anything is created; `RunOutput` read with `cls(**json)`, unknown keys raise |
| 7 | Documentation | pass-with-follow-up | `classes.html`, `data_flow.html` updated in `4cb2adf` (and for 117 in `7ad54ac`); L5; H2 decision and merge plan not yet in spec.md / updates.md; L9 |
| 8 | Risk register | pass-with-follow-up | was fail: H2 and M5 decided by owner, M1 fixed by spec-117. Open follow-ups: M3, M4, M6 |

**Test inventory**

| Kind | Count | Where |
|---|---|---|
| Full suite (spec-117 HEAD `9cd0e67`, all 116 code) | 1,805 passed, 0 failed, 71 skipped | 2026-11-14 run. On `b95e485`: see updates.md 2026-11-15 (pending) |
| Unit, new since 2026-10-31 | 5 | `tests/test_step_decisions.py`, `tests/album/test_normalizer.py`, `tests/architecture/test_logging_aligned.py`, `tests/face_clustering/test_streamlit_app.py` |
| Playwright on the real app | 11, Modal faked | `tests/test_configure_run_on_e2e.py`, `test_albums_table_e2e.py`, `test_jobs_page_run_e2e.py` |
| Budapest V2 gate | **11/11 passed** on `spec/117-m3-fixes` (all 116 code) | `tests/face_clustering/e2e_budapest/` |
| Neon parity | 315/316 on the fix branch; the one failure (cancel test 30 s wait) fixed in `151b79d`, 8 cancel tests then passed on Neon | `tests/storage/test_neon_parity.py` |
| Real runs, this branch | 3 local Budapest + diagnostic (2026-10-31); Slovakia26 re-runs `277fb705` (SIGHTING-146) and `52776cc5` (K=51, 8 people, 302 picks) | SIGHTINGS 146, 147 |
| Real runs, GPU (spec-117 path) | `f7d8ba85` succeeded, `fd52013c` cached re-run | spec-117 updates.md M2 |

**Failure walk-through:** staged path stored: `test_every_path_is_the_original`, `test_rows_name_the_originals_on_both_machines`. Boxes on the wrong scale: `test_pixel_boxes_scale_and_ratio_boxes_stay`. Job path ignores clustering: `test_people_clustering_reaches_the_steps`. Partial rows: `test_a_failed_write_leaves_no_rows`. Run stuck forever: `test_a_process_that_died_ends_failed`. Spend cap bypass: `test_l4_over_cap_is_refused`. No-scene photos pooled / called near-dups: the three SIGHTING-146 tests. App hang from rendering all tabs: `test_only_the_chosen_tab_renders`. Lost link to Modal fails the run: `tests/serving/test_detached_modal_status.py` (spec-117). Album K drifting from the profile: **no test** (M6). Re-run picks changing: **no test** (M5, owner follow-up).

**Verdict: Accepted, no High open**, conditional on the merge plan (116 merges only with 117, one PR). Before `Implemented`: full suite on `b95e485` recorded (see updates.md 2026-11-15, pending); the H2 owner decision and the merge plan recorded in spec.md / updates.md; SIGHTINGS 143 and 145 closed (L9). M3, M4, M6 are follow-ups.
