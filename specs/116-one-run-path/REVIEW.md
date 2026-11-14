# spec-116 — Code review

**Status:** Blocked, 2 High findings (H1 narrowed, H2 unchanged) · **Date:** 2026-11-14 (refresh of 2026-10-31) · **Base:** `main...spec/116-one-run-path` (`4cb2adf`..`edc7678`) · **Checklist:** `docs/guides/CODE_REVIEW_CHECKLIST.md`

Self-review by an agent, not an independent one. The three-dot diff also carries spec-115 commits (`7c9d9f2`..`01192fe`, reviewed under spec-115); this review covers `4cb2adf`, `7a29c4a`, `edc7678`.

## Changes since the 2026-10-31 review

- **Committed:** the reviewed W1-W3 work landed as `4cb2adf` (same code as the old review; no further 116 code after it apart from the two fixes below).
- **New code reviewed:** `7a29c4a` (SIGHTING-146: photos in no scene are judged alone) and `edc7678` (SIGHTING-147 people clustering K 5 -> 51, blur_min 50 -> 73.3; SIGHTING-148 face clustering app renders one tab, one log dir per process).
- **H1 narrowed:** spec-117 (stacked on this branch) ran the GPU path for real. The shared spec-116 classes are now proven on real data; the spec-116-only transport (ephemeral Modal app + copy-back) is not. See H1.
- **H2 unchanged:** SIGHTING-144 is still OPEN, owner decision.
- **New:** M5 (re-run picks differ, SIGHTING-154), M6 (K=51 has no parity guard), L6-L8. M4 grew: `edc7678` and `7a29c4a` change picks/people on every run.
- **Test inventory refreshed** (spec-117 HEAD `9cd0e67`, which contains all 116 code).

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

spec-117 replaces the transport row: deployed app + `spawn`, and `WorkerPersister` on the worker calls the same `RunOutput.from_json(...).remap(PathMap...)` -> `RunPersister` (`sim_bench/serving/worker_persister.py:71-74` at `9cd0e67`). No copy-back.

Layering: `app` -> `api` -> `serving` / `control` -> `pipeline`. Exception, recorded: `serving/run_output.py` imports `api.schemas.result` and, lazily, `api.services.people_service`. `pipeline_service` imports `routers.jobs` lazily for its default `JobService`. `7a29c4a` adds `album/normalizer.py` -> `pipeline.clustering_labels` (data -> lower layer, fine).

## Part 2: Findings

### High

| # | Finding | Evidence | Status |
|---|---|---|---|
| H1 | **The spec-116 L4 transport has no real-data proof.** Checklist: "a new public API with no E2E test exercising it on real data". **Now proven (spec-117, real Slovakia26, Modal NVIDIA A10 fallback for an L4 request):** `RunSpec` reaches the worker and runs (`f7d8ba85`: 659 photos, 329 picks, 7 people); `RunOutput` JSON round trip + `PathMap.remap` on real `/inputs/` keys give original `D:\Slovakia26\...` paths, 0 staged (R5 paths); `RunPersister` rows open in Albums, Results (329 cards, 0 missing), People (7, crops), Explore (R6); a cached GPU re-run `fd52013c` with cached steps 11-99 ms, pipeline 33 s vs 16.5 min of steps (R9, cache is Neon there, not the Volume); runs survive the PC's API and UI stopping. **Not proven:** the code on THIS branch: `ModalVolumeRemote.run_job` with the ephemeral `run_app.run()` (`sim_bench/serving/runners.py:149-161`), copy-back of `run_output.json` from the Volume, remap+persist on the PC, the "copying back" stage (R7), the Volume cache (R9 as written), cancel of a queued L4 run. spec-117 rewrote `runners.py` (+/-192 lines) and `curate_job.py`, so its runs say nothing about them. Faces: crops shown, "cut the right face" not checked | `specs/117-neon-r2-one-database/updates.md` 2026-11-14 M2; SIGHTINGS 153, 154 | **Open if 116 merges alone. Resolved by supersession if 116 only reaches `main` together with 117** (owner choice; record it here) |
| H2 | **R13 fails its check.** 768 px is not close enough at production settings (8 vs 96 photos reach the album, 11 vs 8 people), and with the sharpness gate off people still split 11 vs 8. Every run uses 768 px, so this changes results for every album | `reports/2026-10-30_spec116_r13_fullsize_vs_768/`; SIGHTING-144 **OPEN** (2026-10-31, "owner decision"), no decision recorded since | Open: owner decision |

### Medium

| # | Finding | Ticket |
|---|---|---|
| M1 | The PC must stay online for a whole L4 run: `runners.py:155` holds `run_app.run()` open; a drop raises on leaving the `with`, past the `except` at `:161`. Fixed in spec-117 by a deployed app (`f7d8ba85` finished with the PC's API/UI stopped), still present on this branch | SIGHTING-145 (OPEN) |
| M2 | A queued L4 run reads "Running": `runners.py:229` writes `running` before the worker starts | SIGHTING-143 (OPEN) |
| M3 | Behaviour changes to confirm: straightened picks show as the original; photos in subfolders are not run; Re-run left the Jobs page; a failed people or metric write now fails the run. **New (`7a29c4a`):** no-scene photos have no per-scene cap any more, every one >= `min_score_threshold` is kept (Slovakia26 no-scene picks 24 -> 85) | spec §6 notes, updates.md, SIGHTING-146 |
| M4 | Regression baselines not re-measured. `SpecBuilder.calc` applies `people_clustering`, and since `edc7678` that block is K=51 / blur_min 73.3 (`configs/pipeline.yaml:205,211`), plus the `7a29c4a` selection change: picks and people shift on every album | TODO |
| M5 | **New.** Re-running the same album on the GPU gives different picks (329 vs 331, 3 photos differ, near-duplicate shots); spec-110 AC4 not met. Seen on the spec-117 path; the cause (fresh vs cached features, Siamese) is in code this spec routes every run through | SIGHTING-154 (OPEN) |
| M6 | **New.** Nothing ties `pipeline.yaml` `people_clustering` to the validated `profile_4.json`; the K=5 drift that split 4 people into 76 can recur silently. Checklist §8 "workarounds named and tested" | SIGHTING-147 "Prevention (open)" |

### Low

| # | Finding | Ticket |
|---|---|---|
| L1 | `PeopleService.create_from_clusters` is now used by tests only; `JobProgress` and the websocket queue are never fed | TODO |
| L2 | `run_output.py` 320 LOC (justified in its docstring); `job_service.py` 327 and `job_store.py` 309 passed 300 in this spec with no note; `select_best.py` 653 LOC grew by 37 in `7a29c4a` (`_select_unclustered`, `select_best.py:395`) | TODO |
| L3 | `start_pipeline` has 6 parameters (was 5) | Accepted: mirrors the request |
| L4 | Albums shows 7 old runs stuck at `running`, made before spec-116, with no job to end them | TODO |
| L5 | `reports/` `report.html` + `summary.md` for R13 not written | updates.md |
| L6 | **New.** `min_score_threshold` default `0.4` is repeated at `select_best.py:346` and `:408` (same pattern as before; the schema default exists at `:73`) | TODO |
| L7 | **New.** Wrong per-photo reasons: `build_reason_by_path` (`run_output.py:26-43`, moved unchanged from `pipeline_service`) keeps `detect_persons`' reason over `filter_quality`'s; and "Best in cluster"/"Rank N" use the unsorted list (`select_best.py:373`). Pre-existing, but R6 promises "per-photo reasons" | SIGHTING-149, SIGHTING-150 (OPEN, planned in spec-118) |
| L8 | **New.** `_merge_decisions_panel.py:702+` loop body keeps its old 16-space indent under an 8-space `for` (valid, misleading); `tests/face_clustering/test_streamlit_e2e.py` is stale (clicks tabs that no longer exist), noted in SIGHTING-148, not fixed | TODO |

## Part 3: Verdict per area

| § | Area | Verdict | Note |
|---|---|---|---|
| 1 | Structure | pass-with-follow-up | L1, L2, L3, L8 |
| 2 | Code quality | pass-with-follow-up | Swallowed writes became failures (R8). `7a29c4a` adds a decision row per no-scene photo with the real threshold in the reason; L6 |
| 3 | Naming and packages | pass | `__all__` on every new module; `TABS` dict in `app/face_clustering/main.py:70-84` is clear |
| 4 | Layering and coupling | pass-with-follow-up | One writer for result rows (`RunPersister`, also the worker's writer in 117); `serving` -> `api` import recorded |
| 5 | Testability | **fail** | H1 (this branch's L4 transport only faked). New fixes each have a test that fails without them: `tests/test_step_decisions.py::test_noise_photos_are_judged_alone_not_pooled`, `::test_noise_photos_judged_alone_with_face_subclusters`, `tests/album/test_normalizer.py::test_noise_photo_is_never_a_near_dup`, `tests/face_clustering/test_streamlit_app.py::test_only_the_chosen_tab_renders`, `tests/architecture/test_logging_aligned.py::test_setup_logging_is_one_dir_per_process`. K=51 verified on a real Slovakia26 re-run (8 people, was 76) but has no test (M6) |
| 6 | Boundary contracts | pass | `RunRow` forbids extras; `RunSpec` validated before anything is created; `RunOutput` read with `cls(**json)`, unknown keys raise; the 117 worker uses the same `from_json` on real output |
| 7 | Documentation | pass-with-follow-up | `classes.html`, `data_flow.html` updated in `4cb2adf`; no DB column changed; SIGHTINGS 146-148 + LEARNINGS + CHANGES_LOG for the fixes; L5. `data_flow.html` does not describe the -1 rule in `select_best` (minor) |
| 8 | Risk register | **fail** | H2, M1, M5, M6 |

**Test inventory**

| Kind | Count | Where |
|---|---|---|
| Full suite (spec-117 HEAD `9cd0e67`, all 116 code) | 1,805 passed, 0 failed, 71 skipped | this session's run. Skips were 2 on 2026-10-31; the 69 extra were not investigated individually (L-level follow-up) |
| Unit, new since old review | 5 (above) | `tests/test_step_decisions.py`, `tests/album/test_normalizer.py`, `tests/architecture/test_logging_aligned.py`, `tests/face_clustering/test_streamlit_app.py` |
| Playwright on the real app | 11, Modal faked | `tests/test_configure_run_on_e2e.py`, `test_albums_table_e2e.py`, `test_jobs_page_run_e2e.py` |
| Budapest V2 gate | **pending** (being run separately; needed after `edc7678` changed the face clustering app's nav and `7a29c4a` the picks) | `tests/face_clustering/e2e_budapest/` |
| Real runs, this branch | 3 local Budapest + diagnostic (2026-10-31); Slovakia26 re-runs `277fb705` (SIGHTING-146) and `52776cc5` (K=51, 8 people, 302 picks) | SIGHTINGS 146, 147 |
| Real runs, GPU (spec-117 path) | `f7d8ba85` succeeded, `fd52013c` cached re-run; 3 earlier failures fixed under SIGHTING-153 | spec-117 updates.md M2 |

**Failure walk-through:** staged path stored: `test_every_path_is_the_original`, `test_rows_name_the_originals_on_both_machines`. Boxes on the wrong scale: `test_pixel_boxes_scale_and_ratio_boxes_stay`. Job path ignores clustering: `test_people_clustering_reaches_the_steps`. Partial rows: `test_a_failed_write_leaves_no_rows`. Run stuck forever: `test_a_process_that_died_ends_failed`. Spend cap bypass: `test_l4_over_cap_is_refused`. No-scene photos pooled / called near-dups: the three SIGHTING-146 tests above. App hang from rendering all tabs: `test_only_the_chosen_tab_renders`. Album K drifting from the profile: **no test** (M6). Re-run picks changing: **no test** (M5).

**Blockers:** H1 and H2 block `Implemented`; the Budapest gate is pending. Decisions needed: (H1) merge 116 only with 117 and record H1 as superseded, or run the spec-116 transport on real data, or waive; (H2) accept 768 px with SIGHTING-144 as follow-up, fix the gate, or keep In Progress. Spec stays In Progress.
