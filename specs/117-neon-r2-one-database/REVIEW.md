# spec-117 — Code review

**Status:** Accepted, no High open (2026-11-15). Not `Implemented` yet: the gates under Verdict must land in updates.md first (full suite on `b95e485`, T8.4 secret scan + sizes, cancel of a queued L4 run, `modal deploy` of the fix image) · **Date:** 2026-11-15 (refresh of 2026-11-14) · **Base:** `spec/116-one-run-path` .. `b95e485` (branch `spec/117-neon-r2`, merge of `spec/117-m3-fixes`) · **Checklist:** `docs/guides/CODE_REVIEW_CHECKLIST.md`

Review by a separate agent; not human. Scope: code under `sim_bench/`, `app/`, `scripts/`, `tests/` (79 files, +4,256 / -808 at `9cd0e67`; the fix commits add 14 files, +334 / -25); docs checked for §7 only.

## Changes since 2026-11-14

- **Fix branch merged** (`b95e485`, `git log 92f7882..b95e485`): `10358db` (H1 / SIGHTING-156, M1, M6, SIGHTING-155), `7ad54ac` (H2 docs), `f11f595` (owner decisions, recorded in `docs/project/SIGHTINGS.md` only), `151b79d` (cancel tests' fake job waits 300 s, not 30 s). CHANGES_LOG gained 7 entries dated 2026-11-14, M4's among them.
- **Fixed:** H1, H2, M1 (database half), M4, M6. **Decided by owner:** M5. **Unchanged:** M2, M3, L1-L10.
- **New:** M7 (other errors that mean "this PC has a problem with Modal" still read as "failed"), M8 (the worker still runs without `R2_*`), L11 (the M1 guard is not in the deployed GPU image yet), L12 (the H1 test uses raw `GRPCError` where Modal 1.6.1 raises `ServiceError`), L13 (a cancel while Modal is unreachable fails the job but does not stop the GPU).
- **Evidence added:** Budapest gate 11/11 on the fix branch. Neon parity 315/316 on the fix branch; the one failure (`tests/control/test_job_service.py` `BlockingRunner` 30 s wait) was raised to 300 s in `151b79d`, after which the 8 cancel tests passed on Neon. The SIGHTING-155 heartbeat test passes on Neon. Full suite on `b95e485`: see updates.md 2026-11-15 (pending).

## Part 1: How it works

```
.env (DATABASE_URL, R2_*) --load_env()--> API startup / local worker child / Modal secret "albumify-cloud"
                                   |
            storage/database.py DatabaseConfig  -- one engine factory (SQLite default | Neon via psycopg 3)
            storage/object_store.py  LocalObjectStore(jobs_root) | R2ObjectStore (jobs/<job_id>/...)

Configure & Run -> PipelineService -> JobService.run -> stage photos (manifest: keys, originals, sizes)
   LocalRunner : child process, DB URL by env var (never argv)      \
   ModalRunner : upload missing photos, DetachedModalRemote.spawn,  -> worker_curator():
                 store call_id, follow the JOB ROW                  /    require_shared_db(runner)   (M1, 10358db)
                                                                         JobCurator.calc
                                                                         DbProgressWriter -> jobs.progress, heartbeat 5 s
                                                                         run_pipeline (UniversalCacheHandler on DATABASE_URL,
                                                                                       staged:<sha16>/<name> content keys, 1 query/step)
                                                                         WorkerPersister: result files -> object store,
                                                                           ResultsLoader (job rows, LocalRenders 16-thread uploads),
                                                                           write_run_rows (API rows, original paths) -> succeeded
   RunWatchdog (API thread, 30 s; and on each status GET): heartbeat > 90 s, or call ended -> failed
   DetachedModalRemote.status: running | done | failed: <why> | gone | unknown (PC cannot reach Modal; H1, 10358db)
       call_ended("unknown") is False -> the heartbeat decides
Albums / Results / People / Explore: GET -> autocommit read session -> ResultCache (finished rows) + 768 px thumbnails
```

Layering intended: `app` -> `api` -> `serving` / `control` -> `pipeline`; `storage` at the bottom. Exceptions (existing + new):
`serving/run_output.py` -> `api.schemas`; new: `serving/runners.py:274` -> `api.database.session`,
`serving/worker_persister.py:60-61,87` -> `api.database.models`, `api.services.run_persister` (lazy). This is why the GPU
image needed `fastapi` guarded in `api/database/session.py:8-11` (SIGHTING-153 #1).

**H1 consumers checked (2026-11-15).** Every caller of `status()` goes through `call_ended`; nothing compares the
string or tests `startswith("failed")`:

| Consumer | Code | "unknown" handled |
|---|---|---|
| `RunWatchdog._reason` | `run_watchdog.py:70-71` (`status is None or not call_ended(status)` -> no verdict) | yes: heartbeat age decides (`:61-69`) |
| `ModalRunner._follow` | `runners.py:229-230` | yes: keeps polling; a dead worker is failed by the watchdog, `_follow` then sees the terminal row (`:222-226`) |
| `PipelineService.get_status` | `pipeline_service.py:198` -> `reconcile([job_id])` | yes, via the watchdog |
| UI pages | `app/streamlit/pages/jobs.py:78`, `configure.py:123` compare the *job* status, not the Modal status | n/a |

## Part 2: Findings

### High

| # | Finding | Evidence | Status |
|---|---|---|---|
| H1 | **"Modal not reachable" is reported as "the call failed", so a healthy detached run can be failed and its paid results lost.** `DetachedModalRemote.status` mapped *every* exception except `TimeoutError`/`OutputExpiredError` to `"failed: ..."`; `modal.exception.ConnectionError` and gRPC errors landed there too. `RunWatchdog._reason` then failed the job; `ModalRunner._follow` did the same; `WorkerPersister.calc` then refused the results. Tests did not catch it: `FakeCalls.status` *raises* (`tests/serving/test_run_watchdog.py:20-22, 91-94`) | `sim_bench/serving/detached_modal.py:297-305` (old); `run_watchdog.py:235-240`; `runners.py:229-234`; `pipeline_service.py:198`; `worker_persister.py:110` | **Fixed** `10358db`: `status()` returns `"unknown"` when `unreachable(e)` (`detached_modal.py:57-60, 69-80`: `OSError`, Modal `ConnectionError` / `ClientClosed` / `ServiceError`, grpclib stream/protocol errors, gRPC UNAVAILABLE / DEADLINE_EXCEEDED); `call_ended` excludes it (`:83-86`). All three consumers checked (table above). Test: `tests/serving/test_detached_modal_status.py` (12 tests) drives the real `status()` with real Modal / grpclib exception classes and runs the watchdog on top (`ut_WatchdogWithRealRemote`). Remaining edges: M7, L12. SIGHTING-156 RESOLVED |
| H2 | **§7 docs mandate: the architecture HTMLs stop at W6.** Not documented: `ImageMetrics.display_path`, `GET /results/{id}/decisions`, `DELETE /jobs/{id}`, `ResultCache`, `RunDisplayCopies`, autocommit read sessions, `JobRepo.set_progress` / `runners()`, parallel uploads (`UPLOAD_WORKERS`) | `api/schemas/result.py:18`; `api/routers/results.py:43`; `api/routers/jobs.py:110`; `api/services/result_service.py:20-62`; `api/database/session.py:24-29,85-86`; `control/db.py:50-52,70-77` | **Fixed** `7ad54ac`: every listed name is in `docs/architecture/classes.html:368-372` (rows "spec-117 after W6", `ResultCache`, `RunDisplayCopies`, `ImageMetrics.display_path`, autocommit read sessions) and `docs/architecture/data_flow.html:425-456` (new "spec-117 read path" block: `require_shared_db`, `UPLOAD_WORKERS = 16`, `status` "unknown", `DELETE /jobs/{id}`, `slowest_slowdown`, `get_session(request)`, `ControlDB.read_session`, `/decisions`, `JobRepo.runners()`). The `10358db` additions are documented too |

### Medium

| # | Finding | Evidence | Status |
|---|---|---|---|
| M1 | The worker never checked that it has the shared database: without `DATABASE_URL` the Modal worker wrote to a SQLite file inside the container and lost the paid results | `serving/runners.py:281` (old); `serving/modal_app.py:240-276`; `control/jobs.py:130-145` | **Fixed (database half)** `10358db`: `require_shared_db(runner)` (`runners.py:271-282`) runs first in `worker_curator` (`:288`) and stops a `modal_*` worker with no or a SQLite `DATABASE_URL`. Test: `tests/serving/test_worker_needs_shared_db.py` (6, incl. "nothing created before the check"). The `R2_*` half the 2026-11-14 fix asked for is not checked: M8. The guard reaches the GPU only after a `modal deploy` (L11) |
| M2 | Two runs over the same photos at once can hit `uq_universal_cache_key`; the second writer's step fails and the session is left unrolled-back | `pipeline/cache_handler.py:125-135`; `api/database/models.py:157` | Still open (follow-up: upsert or rollback + retry; no sighting filed yet) |
| M3 | New `serving -> api` imports; the `fastapi` guard in `session.py` works around them | `serving/runners.py:274`; `serving/worker_persister.py:60-61,87`; `api/database/session.py:8-11`; `tests/architecture/test_worker_imports_without_fastapi.py` | Accept with follow-up (refactor TODO) |
| M4 | No CHANGES_LOG entries for `dd33596`, `c0a2606`, `1b7d747`, `c56d6db` | `git diff spec/116...HEAD -- CHANGES_LOG.md` | **Fixed** `10358db`: `CHANGES_LOG.md:41-44` ("[DOCS] spec-117 review M4 — entries for four earlier commits") names all four with what each changed |
| M5 | The definition of done is not met: R11 (< 2 s on Neon) holds on People only (Results first open 11.6 s, Explore 7.7 s); SIGHTING-154 (re-run picks differ by 3 photos; spec-110 AC4) open | spec.md R11; updates.md 2026-11-13/14 | **Decided by owner** 2026-11-14, `f11f595`: both are follow-ups and do not block the merge (`docs/project/SIGHTINGS.md` SIGHTING-152 "Still open" line and SIGHTING-154 status). Not yet copied into spec.md R11 or updates.md: record it there before `Implemented` |
| M6 | The spend cap under-counted on the fallback GPU (L4 timing, dearest price; 9.7 min estimated vs 16.5 min on the A10G) | `serving/job_service.py:142-143` (old); `tests/serving/test_modal_gpu_choice.py:38` | **Fixed** `10358db`: minutes = L4 time x `slowest_slowdown` (`job_service.py:142`; `serving_config.py:37-39, 52-54`; `configs/serving.yaml:35-37`, A10G 1.7), dollars from those minutes at the dearest rate (`:143`); the cap (`max_gpu_minutes`, `:147`) now counts A10 minutes. Tests: `tests/serving/test_modal_gpu_choice.py::test_the_spend_cap_is_checked_at_the_dearest_gpu`, `::test_the_estimate_assumes_the_slowest_gpu`; `tests/control/test_job_service.py::ut_Estimate::test_l4_estimate_has_minutes_and_dollars`. An unknown GPU still counts at L4 speed and price (deliberate, tested) |
| M7 | **New.** `unreachable(e)` misses other errors that come from this PC's link to Modal, not from the call: `AuthError` (expired token), `PermissionDeniedError`, `InternalError` (gRPC INTERNAL during a Modal incident). Modal 1.6.1 raises these as `_GRPCErrorWrapper` subclasses (`modal/_grpc_client.py:27-44`), so they fall through to `"failed: ..."` and a healthy run is failed exactly as in H1. Less likely than a network blip | `sim_bench/serving/detached_modal.py:61, 77-80` | Still open (follow-up): treat every `_GRPCErrorWrapper` except `NotFoundError` as unknown, or invert the rule (only `FunctionTimeoutError`, `RemoteError`, `InternalFailure`, `ExecutionError`, `NotFoundError` mean failed) |
| M8 | **New.** The worker still runs without the shared object store. `make_object_store` falls back to `LocalObjectStore` when any `R2_*` is missing (`storage/object_store.py:196-202`); on Modal that is the `/jobs` Volume, so the rows land in Neon but their photos/crops are not in R2 and the PC shows broken images. Same silent-default class as M1 | `sim_bench/serving/runners.py:271-282, 295` | Still open (follow-up): extend `require_shared_db` to require all `R2_*` for `modal_*` runners, with a test |

### Low

| # | Finding | Evidence | Status |
|---|---|---|---|
| L1 | `a or b if cond else c` with no parentheses (§2 names this exact pattern). Both happen to be correct | `api/services/pipeline_service.py:216`; `serving/job_service.py:178` | TODO |
| L2 | Dead code: `LocalRenders.render_display` / `render_crop` have no callers; `_redact` alias kept "for its callers" | `control/local_renders.py:123-128`; `control/db.py:37` | TODO |
| L3 | More than 4 parameters: `_render_all` has 7; `run_pipeline` grew to 7; `DetachedModalRemote.start` has 5 | `control/local_renders.py:77`; `serving/curate_job.py:73` | TODO |
| L4 | Files over 300 LOC grew with no size note: `result_service` 460 -> 515, `job_service` 327 -> 387, `job_store` 309 -> 323, `modal_app` 360 -> 366 | `wc -l` | TODO (carried from spec-116 L2) |
| L5 | `ResultCache` eviction runs in the worker process, so a re-loaded run is served stale by the API until restart or LRU | `api/services/run_persister.py:28`; `result_service.py:29-31` | TODO |
| L6 | Each status GET of a detached run calls Modal; the watchdog thread logs only the exception type | `pipeline_service.py:198`; `api/main.py:79-81` | TODO |
| L7 | `redact_url` hides user:password but not a `?password=` query parameter; `main.py` uses the literal `"DATABASE_URL"` | `storage/database.py:46-53`; `api/main.py:71` | TODO |
| L8 | Leftovers on delete: album delete leaves job rows and R2 objects; `delete_job` leaves API run rows pointing at deleted files | `api/services/album_service.py:65-77`; `serving/job_service.py:218-235` | TODO |
| L9 | Untyped boundaries: `jobs.progress` JSON and the Modal `manifest` dict; `ImageMetrics` has no `extra="forbid"` | `serving/db_progress.py:51-61`; `serving/modal_app.py:268-270` | TODO |
| L10 | The cache key ignores device and `model_version`; CPU- and GPU-computed features now mix | `pipeline/cache_handler.py:22-50, 81-88` | Note on SIGHTING-154 |
| L11 | **New.** The M1 guard runs inside the Modal worker, but the deployed `albumify-curator` image dates from `c56d6db` (updates.md 2026-11-14 M2). Until `modal deploy` from `b95e485`, the GPU runs without it. H1 and M6 are PC-side and live on the next API restart | `sim_bench/serving/modal_app.py:274-275` | Open: redeploy, record in updates.md |
| L12 | **New.** `test_detached_modal_status.py` raises raw `grpclib.GRPCError(UNAVAILABLE / DEADLINE_EXCEEDED / NOT_FOUND)`; Modal 1.6.1 converts these to `ServiceError` / `NotFoundError` before they reach `get()` (`modal/_grpc_client.py:27-52`). `ServiceError` is in `unreachable` but has no test case; the code is right, the "real exception types" claim is only partly true | `tests/serving/test_detached_modal_status.py:55-62, 73-74` | TODO: add `ServiceError` and `NotFoundError` cases |
| L13 | **New** (pre-existing path). A cancel while Modal is unreachable: `ModalRunner._follow` calls `self._remote.cancel` (`runners.py:226-227`), which raises; `JobService.run` (`job_service.py:351`) ends the job failed, but the call keeps running and billing until it ends, and its results are refused | `sim_bench/serving/runners.py:226-227` | TODO |

**Checked and fine:** passwords never logged. `DatabaseConfig.__repr__` is redacted, both engines log `redacted()`, the local worker gets the URL by env var not argv, the migration script prints the target redacted, R2 errors name missing variables only, the image adds `configs/` + Python sources only (no `.env`), and `tests/conftest.py:15-16` clears all cloud variables. Object keys are checked (`..`, absolute, backslash, NUL), and `_file` serves only the job's own keys. `job_people` rename only touches a `people` table that has `job_id`. Column migrations compile per dialect. Postgres FK order was fixed (`bb3ab4b`). The cache read ends its transaction (`cache_handler.py:196-198`). Parallel uploads: the first failure fails the load and the executor waits for the rest. `np.load` from R2 bytes keeps `allow_pickle=False`. **Fix commits:** `require_shared_db` prints no URL; `test_detached_modal_status.py` restores the asyncio event loop policy after importing `modal` (keeps the later Playwright tests on the Proactor loop); the `151b79d` change only raises an upper bound on a fake job that is always cancelled.

## Part 3: Verdict per area

**Test inventory (new/changed):** about 97 new test functions in 17 files at `9cd0e67`, plus 19 from the fix branch: `tests/serving/test_detached_modal_status.py` (12), `tests/serving/test_worker_needs_shared_db.py` (6), `tests/serving/test_modal_gpu_choice.py::test_the_estimate_assumes_the_slowest_gpu` (1); changed: `test_db_progress.py` (heartbeat polls to a deadline), `test_job_service.py` (A10 estimate, 300 s fake job). Real E2E: T8.1 local run `5a6cfec8` on Neon + R2, T8.2 L4 run `f7d8ba85` (PC API/UI stopped), T8.3 cached re-run `fd52013c`. Mocks: moto for R2 and fake Modal calls are justified; the Modal status contract is now tested against the real `status()` (H1).

**Evidence:** full suite on SQLite at `9cd0e67`: 1805 passed, 0 failed, 71 skipped. Fix branch: Budapest V2 gate 11/11; Neon parity 315/316, the one failure fixed in `151b79d` and its 8 cancel tests then passed on Neon; SIGHTING-155 heartbeat test passes on Neon. Full suite on `b95e485`: see updates.md 2026-11-15 (pending).

| § | Area | Verdict | Note |
|---|---|---|---|
| 1 | Structure | pass-with-follow-up | L2, L3, L4; new modules small (`database` 112, `object_store` 202, `db_progress` 165, `worker_persister` 163, `run_watchdog` 82) |
| 2 | Code quality | pass-with-follow-up | M1 fixed for the database; M8 is the same silent default for R2. L1 |
| 3 | Naming and packages | pass | `unreachable`, `UNKNOWN` added to `__all__` |
| 4 | Layering and coupling | pass-with-follow-up | M3 |
| 5 | Testability | pass-with-follow-up | was fail: H1 now has a test of the real `status()` + watchdog. L12 (missing `ServiceError` case) |
| 6 | Boundary contracts | pass-with-follow-up | L9. Config knob -> producer: `serving.yaml gpu_slowdown` -> `ServingConfig.slowest_slowdown` -> `JobService.estimate` -> spend cap (M6 fixed, tested) |
| 7 | Documentation | pass-with-follow-up | was fail: H2 and M4 fixed. Follow-ups: owner decisions (M5, SIGHTING-144) are in SIGHTINGS.md only, copy them into spec.md / updates.md; SIGHTING-155 "Left" line still asks for the parity re-run that has now happened |
| 8 | Risk register | pass-with-follow-up | M2, M7, M8, L5, L8, L10, L11, L13; M5 decided by owner |

**Verdict: Accepted, no High open.** Medium findings M2, M7, M8 are follow-ups and do not block the merge. Before `Implemented`, updates.md must record: the full suite on `b95e485` (see updates.md 2026-11-15, pending); T8.4 secret scan + Neon/R2 sizes (R13, R14); cancel of a queued L4 run (R9); `modal deploy` of the fix image (L11); the M5 owner decision.

**Proposed follow-ups (not filed by this review):**

| Kind | Item |
|---|---|
| Sighting | M2: concurrent cache writers -> `IntegrityError` fails the step |
| Sighting | M7: Modal auth / permission / internal errors on the status poll read as a failed run |
| Sighting | M8: the Modal worker runs without `R2_*` and its files stay on the Volume |
| Docs (this spec) | owner decisions into spec.md / updates.md; SIGHTING-155 "Left" line; L11 redeploy |
| TODO | M3 engine/session factory into `storage/`; L1-L10, L12, L13 housekeeping |
