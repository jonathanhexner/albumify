# spec-115 — Tasks (Album workbench)

**Status:** In Progress — W1, W2 done 2026-10-23; W3 next.
**Spec:** [spec.md](spec.html) · [updates](updates.html) · roadmap stage **C** · [ROADMAP.md](../../ROADMAP.html)

Design lives in the spec (§3 plan, §4 class diagram, §5-§6 file tables). This file orders the work.

## W1 — Stable inputs and a safe cache (done 2026-10-23, commit de83348)
- [x] T1.1 Cache version check tested: `tests/pipeline/test_cache_model_version.py` (4). (R6)
- [x] T1.2 `InputStore` + job manifest + `JobStore.resolve`; API file route uses it. `tests/serving/test_input_store.py` (13). (R6, R14)
- [x] T1.3 Pipeline gets the job's paths explicitly; `run_pipeline` attaches the feature cache. Tests +5; real 12-photo pair. (R6)

## W2 — Results-only; display images and crops made on the PC (done 2026-10-23)
- [x] T2.1 Orientation gate first (R10): on the real `slovakia26` job, crop 20 faces from the PC's input copy with the
      bundle boxes; pass = IoU ≥ 0.5 with the worker's crop on ≥ 19 of 20 (report on disk, not in git). On a fail the
      worker keeps sending crops and W2 renders only the display images. | 2026-10-23: 18/20 by ArcFace, all 20 right face on
      inspection, rotation proven by a no-rotate control; owner accepted. Found SIGHTING-141 (worker crops colour-swapped; fixed).
- [x] T2.2 `LocalRenders.calc`: `display/` for picks, `faces/<id>.jpg` for every face with a box; skips files that exist.
- [x] T2.3 `ResultsLoader` runs `LocalRenders` before loading; `JobCurator` stops writing `display/`
      (`CurateJobInputs.write_displays` default False) and crops (`_write_bundle` passes no crops; the bundle keeps boxes).
- [x] T2.4 `JobStore.results_files(job_id)`: the files a worker returns (`album.json`, `run.json`, `results/*`).
- [x] T2.5 Tests: `tests/control/test_local_renders.py`; job folder identical for local and fake-Modal; old jobs keep their files. (R10, R14)

## W2b — File serving per job (R7)
- [x] T2.6 `JobStore.resolve` takes the job's manifest into account: an `inputs/` key is served only if that job lists it.
      Test `tests/api/test_jobs_router.py::test_other_jobs_photo_is_404`.

## W3 — Modal: inputs and cache Volumes, results-only download, retention
- [ ] T3.1 `modal_app`: Volumes `albumify-inputs` (`/inputs`) and `albumify-cache`; `curate_job(job_id, keys, pipeline)` reads
      `/inputs/<key>`; `ALBUMIFY_FEATURE_CACHE` on the cache Volume, `reload()` before and `commit()` after.
- [ ] T3.2 `ModalVolumeRemote.missing/upload`: list the hashes once, `batch_upload` only the missing photos. (R9)
- [ ] T3.3 `download_results`: only `JobStore.results_files`. (R8)
- [ ] T3.4 Mark photos used per job; `purge_inputs` (daily) deletes photos unused 7 days and the old per-job `input/` copies
      on `albumify-jobs`; `delete_inputs(keys)` for the button, keeping photos another job used in the last 7 days. (R13)
- [ ] T3.5 `tests/serving/test_modal_staging.py` with a fake Volume.

## W4 — The app
- [ ] T4.1 `RunSettings.collect` into `run.json`; `JobDetail.settings`; Compare runs "Settings that differ". (R12)
- [ ] T4.2 `JobService.estimate` + `GET /jobs/estimate`; `configs/serving.yaml` `max_photos_l4` (1,000), `max_gpu_minutes` (30),
      `l4_seconds_per_photo`, `l4_dollars_per_hour`; over a limit → 422. Tests `ut_Estimate`. (R2)
- [ ] T4.3 `Runner.cancel` (local: kill the child; Modal: cancel the spawned call) + `POST /jobs/{id}/cancel`; L4 jobs one at a time;
      end state per R4 (partial `results/` deleted, queue slot freed). Tests `ut_Cancel`. (R4)
- [ ] T4.4 Jobs page: run form, estimate + confirm for L4, live status fragment, Cancel, Re-run → Compare. (R1-R5)
- [ ] T4.5 `DELETE /jobs/{id}/remote-inputs` + button. (R13)
- [ ] T4.6 `tests/test_jobs_page_run_e2e.py` (Playwright, real API + app, local `minimal`); screenshots stay on disk.

## W5 — Prove on real data (owner approval before each L4 run)
- [ ] T5.1 Slovakia26 from the form on the L4: R1, R8 (file count, copy-back time), album opens with crops (R10).
- [ ] T5.2 Same photos again on the L4: R9 (0 uploads, cache hits, GPU time).
- [ ] T5.3 48-photo slice local vs L4 → Compare runs: R11.
- [ ] T5.4 Cancel on both runners (R4); delete-photos button + Volume listing (R13).
- [ ] T5.5 Closes spec-112 TA6.1/TA6.3/TA6.4 (post-SIGHTING-132 Slovakia run, edit round-trip, viewer screenshots).

## Close out
- [ ] TC.1 `docs/architecture/{classes,data_flow}.html`; CHANGES_LOG; `updates.md` with evidence.
- [ ] TC.2 Full suite; `/code-review` → REVIEW.md; Budapest gate not affected (no v2 tab change), stated in REVIEW.
