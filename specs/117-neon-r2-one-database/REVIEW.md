# spec-117 — Code review

**Status:** Blocked, 2 High findings · **Date:** 2026-11-14 · **Base:** `spec/116-one-run-path` (16 commits, `99b2cc3`..`9cd0e67`, branch `spec/117-neon-r2`, local only) · **Checklist:** `docs/guides/CODE_REVIEW_CHECKLIST.md`

Review by a separate agent; not human. Scope: code under `sim_bench/`, `app/`, `scripts/`, `tests/` (79 files, +4,256 / -808); docs checked for §7 only.

## Part 1: How it works

```
.env (DATABASE_URL, R2_*) --load_env()--> API startup / local worker child / Modal secret "albumify-cloud"
                                   |
            storage/database.py DatabaseConfig  -- one engine factory (SQLite default | Neon via psycopg 3)
            storage/object_store.py  LocalObjectStore(jobs_root) | R2ObjectStore (jobs/<job_id>/...)

Configure & Run -> PipelineService -> JobService.run -> stage photos (manifest: keys, originals, sizes)
   LocalRunner : child process, DB URL by env var (never argv)      \
   ModalRunner : upload missing photos, DetachedModalRemote.spawn,  -> worker_curator():
                 store call_id, follow the JOB ROW                  /    JobCurator.calc
                                                                         DbProgressWriter -> jobs.progress, heartbeat 5 s
                                                                         run_pipeline (UniversalCacheHandler on DATABASE_URL,
                                                                                       staged:<sha16>/<name> content keys, 1 query/step)
                                                                         WorkerPersister: result files -> object store,
                                                                           ResultsLoader (job rows, LocalRenders 16-thread uploads),
                                                                           write_run_rows (API rows, original paths) -> succeeded
   RunWatchdog (API thread, 30 s; and on each status GET): heartbeat > 90 s or call ended -> failed
Albums / Results / People / Explore: GET -> autocommit read session -> ResultCache (finished rows) + 768 px thumbnails
```

Layering intended: `app` -> `api` -> `serving` / `control` -> `pipeline`; `storage` at the bottom. Exceptions (existing + new):
`serving/run_output.py` -> `api.schemas`; new: `serving/runners.py:274` -> `api.database.session`,
`serving/worker_persister.py:60-61,87` -> `api.database.models`, `api.services.run_persister` (lazy). This is why the GPU
image needed `fastapi` guarded in `api/database/session.py:8-11` (SIGHTING-153 #1).

## Part 2: Findings

### High

| # | Finding | Evidence | Status |
|---|---|---|---|
| H1 | **"Modal not reachable" is reported as "the call failed", so a healthy detached run can be failed and its paid results lost.** `DetachedModalRemote.status` maps *every* exception except `TimeoutError`/`OutputExpiredError` to `"failed: ..."`; `modal.exception.ConnectionError` (subclass of `modal.exception.Error`, verified in the venv) and gRPC/auth errors land there too. `RunWatchdog._reason` then fails the job ("ended without saving") even with a fresh heartbeat; its `except` for an unreachable Modal (`run_watchdog.py:245`) never fires with the real remote. `ModalRunner._follow` does the same and `JobService.run` then calls `repo.fail`. When the worker finishes, `WorkerPersister.calc` does `transition(LOADING)` from `failed`, which is invalid, so no rows are written. That is the PC-network-blip case R8 exists for. Tests do not catch it: `FakeCalls.status` *raises* (`tests/serving/test_run_watchdog.py:20-22, 91-94`), so it never behaves like the real remote | `sim_bench/serving/detached_modal.py:297-305`; `run_watchdog.py:235-240`; `runners.py:229-234`; `pipeline_service.py:198` (runs on every status GET); `worker_persister.py:110`; `control/jobs.py:35-42` | Open. Fix: only Modal's remote-execution errors (cancelled, function timeout, remote exception) count as `failed`; client/connection errors raise or return "unknown"; add a test of `DetachedModalRemote.status` with a fake `modal` module |
| H2 | **§7 docs mandate: the architecture HTMLs stop at W6.** Last touched in `f6e7274`. Not documented since then: `ImageMetrics.display_path` (Pydantic contract), `GET /results/{id}/decisions`, `DELETE /jobs/{id}`, `ResultCache`, `RunDisplayCopies`, autocommit read sessions (`get_session(request)`, `ControlDB.read_session`), `JobRepo.set_progress` / `runners()`, parallel uploads (`UPLOAD_WORKERS`). `grep` of `docs/architecture/*.html` finds none of `display_path`, `ResultCache`, `RunDisplayCopies`, `read_session` | `api/schemas/result.py:18`; `api/routers/results.py:43`; `api/routers/jobs.py:110`; `api/services/result_service.py:20-62`; `api/database/session.py:24-29,85-86`; `control/db.py:50-52,70-77` | Open (docs only: `classes.html`, `data_flow.html`) |

### Medium

| # | Finding | Evidence | Status |
|---|---|---|---|
| M1 | The worker never checks that it has the shared database. If the Modal secret lacks `DATABASE_URL`, `ControlDB(None)` falls back to a SQLite file inside the container: `set_progress` updates no row, nothing errors, and the results go to a file that is thrown away. The PC finds out only when the call ends ("ended without saving"), after the GPU time is spent. This is a silent default at a boundary (§2). The PC side checks this (`runners.py:197-199`); the worker side does not | `serving/runners.py:281`; `serving/modal_app.py:240-276`; `control/jobs.py:130-145` | Open: fail fast in `_curate_job` unless `DATABASE_URL` is Postgres and `R2_*` are set |
| M2 | Two runs over the same photos at once (this PC + the L4, which this spec makes possible) can hit `uq_universal_cache_key`. `store_many` reads first, then inserts and commits once per step, so the second writer gets `IntegrityError`, the whole step fails, and the session is left unrolled-back | `pipeline/cache_handler.py:125-135`; `api/database/models.py:157` | Open: upsert (`ON CONFLICT DO UPDATE`, both dialects) or rollback + retry |
| M3 | New `serving -> api` imports (Part 1). The `try: from fastapi import Request` in `session.py` works around this, and the architecture test guards it. The real fix is the session/engine factory living in `storage/` | `serving/runners.py:274`; `serving/worker_persister.py:60-61,87`; `api/database/session.py:8-11`; `tests/architecture/test_worker_imports_without_fastapi.py` | Accept with follow-up (refactor TODO) |
| M4 | No CHANGES_LOG entries for 4 code commits: `dd33596` (SIGHTING-152 perf: autocommit reads, ResultCache, one-query lists, thumbnails; SIGHTING-151), `c0a2606`, `1b7d747`, `c56d6db`. The 9 entries added all date from 2026-11-06 (W1-W7) | `git diff spec/116...HEAD -- CHANGES_LOG.md` | Open (§7) |
| M5 | The definition of done is not met. R11 (< 2 s on Neon) is met on People only (Results first open 11.6 s, Explore 7.7 s). SIGHTING-154 (re-run picks differ by 3 photos; spec-110 AC4) is open | spec.md R11; updates.md 2026-11-13/14 | Owner decision (accept the gap, or follow-up spec) |
| M6 | The spend cap under-counts on the fallback GPU. The estimate prices the dearest GPU but times the run with the L4's `l4_seconds_per_photo`: it said 9.7 GPU min / $0.18, and the uncached A10G run took 16.5 min of steps. Any unknown GPU is also priced at the L4 rate. The cap is a safety mechanism (§8) | `serving/job_service.py:142-143`; `tests/serving/test_modal_gpu_choice.py:38`; updates.md 2026-11-14 | Open |

### Low

| # | Finding | Evidence | Status |
|---|---|---|---|
| L1 | `a or b if cond else c` with no parentheses (§2 names this exact pattern). Both happen to be correct | `api/services/pipeline_service.py:216`; `serving/job_service.py:178` | TODO |
| L2 | Dead code: `LocalRenders.render_display` / `render_crop` have no callers; `_redact` alias kept "for its callers" | `control/local_renders.py:123-128`; `control/db.py:37` | TODO |
| L3 | More than 4 parameters: `_render_all` has 7; `run_pipeline` grew to 7; `DetachedModalRemote.start` has 5 | `control/local_renders.py:77`; `serving/curate_job.py:73` | TODO |
| L4 | Files over 300 LOC grew with no size note: `result_service` 460 -> 515, `job_service` 327 -> 387, `job_store` 309 -> 323, `modal_app` 360 -> 366 | `wc -l` | TODO (carried from spec-116 L2) |
| L5 | `ResultCache` eviction now runs in the wrong process. `RunPersister` evicts in the worker, so a re-loaded run (`POST /jobs/{id}/load`) is served stale by the API until restart or LRU. The docstring ("the run's writer evicts its own entry") is no longer true | `api/services/run_persister.py:28`; `result_service.py:29-31` | TODO |
| L6 | Each status GET of a detached run calls Modal (`reconcile([job_id])`). The watchdog thread logs only the exception type, which leaves nothing to debug with | `pipeline_service.py:198`; `api/main.py:79-81` | TODO |
| L7 | `redact_url` hides user:password but not a `?password=` query parameter. `main.py` uses the literal `"DATABASE_URL"` instead of `ENV_VAR` | `storage/database.py:46-53`; `api/main.py:71` | TODO |
| L8 | Leftovers on delete (R14 cost): deleting an album leaves its job rows and R2 objects. `delete_job` leaves the API run rows pointing at deleted files | `api/services/album_service.py:65-77`; `serving/job_service.py:218-235` | TODO |
| L9 | Untyped boundaries: `jobs.progress` JSON (worker -> DB -> API) and the `manifest` dict sent to Modal have no schema. `ImageMetrics` has no `extra="forbid"` (existed before) | `serving/db_progress.py:51-61`; `serving/modal_app.py:268-270` | TODO |
| L10 | The cache key ignores device and `model_version` (existed before; spec079 memory). Now that the PC and the GPU share rows, CPU- and GPU-computed features mix. Not the cause of SIGHTING-154 (GPU after GPU differs too), but it makes runs harder to compare | `pipeline/cache_handler.py:22-50, 81-88` | Note on SIGHTING-154 |

**Checked and fine:** passwords never logged. `DatabaseConfig.__repr__` is redacted, both engines log `redacted()`, the local worker gets the URL by env var not argv, the migration script prints the target redacted, R2 errors name missing variables only, the image adds `configs/` + Python sources only (no `.env`), and `tests/conftest.py:15-16` clears all cloud variables. Object keys are checked (`..`, absolute, backslash, NUL), and `_file` serves only the job's own keys. `job_people` rename only touches a `people` table that has `job_id` (the API's has none). Column migrations compile per dialect. Postgres FK order was fixed (`bb3ab4b`). The cache read ends its transaction (`cache_handler.py:196-198`). Parallel uploads: the first failure fails the load and the executor waits for the rest. `np.load` from R2 bytes keeps `allow_pickle=False`.

## Part 3: Verdict per area

**Test inventory (new/changed):** about 97 new test functions in 17 files. Static/architecture: 7 (`test_no_secrets_leak` 6, `test_worker_imports_without_fastapi` 1). Unit/synthetic: ~88 (storage 23 incl. moto, a fake S3; serving 29; pipeline cache 9; api 13; control 8; streamlit 3; scripts 6, also run on Neon). Parity: `test_neon_parity` (2) re-runs tests/api, control, serving on a throwaway Neon schema. Real E2E: T8.1 local run `5a6cfec8` on Neon + R2, T8.2 L4 run `f7d8ba85` (329 picks, 7 people, PC API/UI stopped), T8.3 cached re-run `fd52013c` (cached steps 11-99 ms). Mocks: moto for R2 and fake Modal calls (no spend in tests) are justified, but the Modal fake does not match the real `status` contract (H1).

**Evidence:** full suite on SQLite at `9cd0e67`: 1805 passed, 0 failed, 71 skipped, 147 deselected (`m3_full_suite.log`). **Pending, see updates.md:** `needs_neon` / `needs_r2` sets, V2 Budapest gate (binding), T8.4 secret scan + Neon/R2 sizes (R13, R14), cancel of a queued L4 run (R9). Each must land in updates.md before `Implemented`.

| § | Area | Verdict | Note |
|---|---|---|---|
| 1 | Structure | pass-with-follow-up | L2, L3, L4; new modules are small and have one job each (`database` 112, `object_store` 202, `db_progress` 165, `worker_persister` 163, `run_watchdog` 82) |
| 2 | Code quality | pass-with-follow-up | M1 (silent SQLite fallback in the worker), L1. Broad excepts are deliberate and logged by type (progress, watchdog, crop) |
| 3 | Naming and packages | pass | New `sim_bench/storage/` package; `__all__` on every new module; `job_people` rename |
| 4 | Layering and coupling | pass-with-follow-up | M3. One writer per state: the worker writes results; `JobService` writes only when a worker did not (documented `job_service.py:9-11`) |
| 5 | Testability | **fail** | H1: the failure mode R8/R9 is meant to prevent (lost link to Modal) has no test against the real status mapping. Real E2E on both machines otherwise good |
| 6 | Boundary contracts | pass-with-follow-up | Object keys enforced; `PipelineStatus` fields typed; L9 (progress JSON, manifest untyped). Config knob -> producer: `serving.yaml gpu` -> `modal_app.GPU_TYPES` and the spend cap (M6) |
| 7 | Documentation | **fail** | H2 (architecture HTMLs), M4 (CHANGES_LOG). spec/tasks/updates present; LEARNINGS updated (+12 lines) |
| 8 | Risk register | pass-with-follow-up | M2, M5, M6, L5, L8, L10; workaround `session.py:8-11` guarded by an architecture test; legacy path-keyed cache rows stay PC-only (documented) |

**Verdict: Blocked.** Do not flip to `Implemented` until H1 is fixed with a test, H2 + M4 docs land, and the pending gates are recorded in updates.md. Waiving either High needs the owner's explicit approval, recorded here.

**Proposed follow-ups (not filed by this review):**

| Kind | Item |
|---|---|
| Sighting | H1: Modal client errors read as a failed run (watchdog + `_follow` fail a live L4 run; results refused) |
| Sighting | M1: the worker runs without `DATABASE_URL` and loses its results silently |
| Sighting | M2: concurrent cache writers -> `IntegrityError` fails the step |
| Sighting | M6: spend estimate uses L4 timing for the A10G fallback (~1.7x low) |
| Docs (this spec) | H2: `classes.html`, `data_flow.html` for the post-W6 changes; M4: CHANGES_LOG for 4 commits |
| Spec / owner | M5: R11 page times (Results/Explore render cost) and SIGHTING-154 reproducibility: accept or new spec |
| TODO | M3 engine/session factory into `storage/`; L1-L9 housekeeping |
