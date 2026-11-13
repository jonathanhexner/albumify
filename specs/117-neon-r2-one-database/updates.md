# spec-117 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress 2026-11-06 (owner approved; W1-W6 being built in worktree `D:\sim-bench-117`)

## 2026-11-13 — SIGHTING-152 part 3: read requests in autocommit (R11 met on People only)

Owner: "go ahead with autocommit for reads".

| Claim | Evidence |
|---|---|
| GET/HEAD get the autocommit session on Postgres, writes keep a transaction | `tests/api/test_read_sessions.py` (8: method -> factory, startup call, `init_db` on a Postgres URL, `ControlDB.read_session`) |
| No GET writes more than one row | audit of every `commit/add/delete/flush` in `sim_bench/api`: only the default-profile sync (one row) |
| Writes still work on Neon | live: create, update, read back, delete a config profile (201/200/200/404 after delete) |
| Nothing else broke | tests/api, serving, control, streamlit, storage, architecture: 486 passed |
| Live on Neon | per call albums/people 0.48 -> 0.34 s, runs 1.9 -> 1.37 s; Albums again 2.4 s, Results again 3.2 s, People 1.5 s, Explore again 5.9 s |
| R11 (< 2 s) | **met on People only**; Results/Explore are Streamlit drawing 302 photos (~3-4 s even on SQLite); Albums first ~8 s (schema read after API start) |

## 2026-11-13 — SIGHTING-152 part 2: thumbnails from the 768 px copies, runs-table and People queries (R11 not met yet)

Owner: "go ahead with the thumbnails and queries".

| Claim | Evidence |
|---|---|
| Thumbnails come from the run's 768 px copy | `tests/api/test_result_queries.py::test_images_and_clusters_carry_the_runs_768px_copy`, `tests/streamlit/test_thumb_sources.py` (all five loaders); 6 ms vs 90 ms per photo on Slovakia26 |
| Runs table is one query (+ jobs id/runner/status) | `::test_runs_table_query_count_does_not_grow_with_runs` (now asserts 1 query and the counts) |
| People list is one query without face instances | `::test_people_list_is_one_query_without_face_instances` |
| Nothing else broke | tests/api, serving, control, streamlit, storage, architecture: 478 passed |
| Live on Neon (both servers restarted, same browser walk) | Results first open 86 -> 12.3 s, Explore first 48.4 -> 8.0 s, People 4.5 -> 1.8 s, Albums again 3.0 s, Results again 3.9 s; 302 thumbnails, 0 missing, 0 errors |
| R11 (< 2 s) | **not met**: each request pays 3-5 extra round trips (pre-ping, separate BEGIN, ROLLBACK/COMMIT; 0.80 s vs 0.48 s autocommit for one query); runs list 1.8-2.2 s; first request after API start 8 s (schema reflection); Streamlit renders a 302-photo page in ~3-4 s even on SQLite |

## 2026-11-13 — SIGHTING-152 part fix: queries + result cache (R11 not met yet)

Owner: "go ahead with M1 and the cache" (chosen over copying results back to local SQLite).

| Claim | Evidence |
|---|---|
| Lists read summary columns in fixed queries | `tests/api/test_result_queries.py::test_result_list_is_summary_only_in_fixed_queries`, `::test_runs_table_query_count_does_not_grow_with_runs` (fails without the join: 5 vs 4) |
| A finished result is read from the database once | `::test_a_finished_result_is_read_from_the_database_once`, `::test_an_unfinished_run_is_not_cached`, `::test_deleting_the_album_evicts_its_results`, `::test_decisions_route` |
| Nothing else broke | tests/api, serving, control, streamlit, storage, architecture: 474 passed |
| Live on Neon (same profiler + browser script) | API 75 s -> 29 s, SQL queries 188 -> 45. Results again 18.4 -> 4.6 s, Explore again 54.5 -> 5.9 s, Albums 7.7 -> 3.7 s; Explore shows step decisions; Results 302 cards, 0 errors |
| R11 (< 2 s) | **not met**: first visits 48-86 s (thumbnails from originals, C5), runs list ~4 s per call, People 4.5 s |

## 2026-11-13 — SIGHTING-151 fixed; SIGHTING-152 measured (no fix yet)

| Claim | Evidence |
|---|---|
| Results shows every pick | `tests/streamlit/test_gallery_keys.py` (colliding keys): fails before, passes after; live app on Neon: 302 Detail buttons, 0 errors |
| Where Neon page time goes | profiling API (per-request time + SQL count) under the real UI: 75 s API of 211 s page time, 61 s in 188 SQL queries; list endpoints load whole 2.8 MB result rows (Slovakia26: 2.0 s vs 0.18 s for summary columns); runs list N+1 (26 album queries per call); 7 repeat fetches per Results visit; cold thumbnails 105 s vs 18 s warm; round trip 160-320 ms. Detail in SIGHTING-152 |
| Neon vs local | Albums 7.7-15 s vs 1.0 s; Results 18 s warm vs 2.9 s; Explore 54 s vs 4.5 s; People 4.4 s vs 0.8 s |

## 2026-11-13 — W7 real migration; T8.1 run from the app: works, too slow, one gallery bug

Owner OK for the real migration (copy everything incl. cache). Merged spec/116 into spec/117 first (`b163e19`): this
branch was missing the K=51 fix, so the first Neon run (`e35ec67e`, started by API) clustered 76 people.

| Claim | Evidence |
|---|---|
| R4 real migration | `migrate_to_neon.py --copy --yes`: 17 tables, all counts verified (25,636 cache rows), 2 min 19 s; SQLite kept |
| T8.1 run started from the app on Neon + R2 | Playwright on the real app (8511): Configure & Run -> Slovakia26 -> Full -> This PC -> Run; run `5a6cfec8`, K=51, completed in 23.7 min: 659 photos, 302 picks, 7 people (SQLite run `52776cc5`: 302 / 8) |
| Albums shows the run | top row: This PC, 659 / 302 / 7, completed |
| Results opens it with images | 231 of 231 drawn images load; **but** the gallery stops at a duplicate button key: SIGHTING-151 (old bug, not Neon) |
| People & Faces, Explore open | no errors; 7 people |
| **Page timings (R11): FAIL** | Results 54 s cold / 28.5 s warm (SQLite app 2.9 s), Explore 54 s, Albums 8-12 s, People 2.2 s: SIGHTING-152 |

Open in W8: T8.1 timings (SIGHTING-152), T8.2/T8.3 need `modal deploy` + L4 spend (owner OK), T8.4 secret scan, T8.5.

## 2026-11-06 — W7 migration script written and tested (NOT run on the real data)

`scripts/migrate_to_neon.py` `NeonMigrator`: `plan` (row counts per table in `sim_bench.db` and `control.db`),
`copy` (refuses a target holding any Albumify rows; creates the tables; copies in FK order, in batches, through the
ORM column types; control.db's old `people` lands in `job_people`; moves Postgres id sequences past the copied ids),
`verify` (per-table counts, raises on a difference). The sources are only read. `--skip universal_cache` leaves the
cache out (it is regenerable and may be large against the Neon free size).

| Claim | Evidence |
|---|---|
| R4 copy + verify, counts match, refuses a non-empty target, sources unchanged (hash), JSON/blob/datetime round-trip, new ids after the copy | `tests/scripts/test_migrate_to_neon.py`: 6 passed on SQLite, and 6 passed with the target on a throwaway Neon schema (183 s, schema dropped) |
| Real migration | **not run** (owner instruction: W7 real run and W8 wait); per-table counts go here when it is |

**Risks for the real run**: the API's `sim_bench.db` never had foreign keys switched on, so orphan rows (e.g. people of
a deleted run) would be refused by Postgres; run `--plan` first and fix or skip what fails. Feature-cache rows keyed
by full staged paths still hit on this PC (legacy lookup) but not on the L4 until a local run rewrites them as
content keys.

## 2026-11-06 — W6 One feature cache landed

`UniversalCacheHandler.load_many` / `store_many`: one IN query per (feature, model), i.e. one per step, and one
commit per step's writes (`BaseStep` uses `store_many`; `load_from_cache` delegates to `load_many`). Photos from the
shared input store are keyed `staged:<sha16>/<name>.jpg` (content, the same on this PC and the L4) and skip the
mtime check; rows an older run stored under the full staged path are still found by the same query. `run_pipeline`
opens the cache through `DATABASE_URL` (`feature_cache_config`; `ALBUMIFY_FEATURE_CACHE` still names an isolated
SQLite file; unset -> `sim_bench.db`). The `albumify-cache` Volume and its copy in/out left `modal_app.py` in W5.
Also landed: `tests/architecture/test_no_secrets_leak.py` (R13 unit part) and the architecture docs
(`data_flow.html` spec-117 section, `classes.html` new classes, `db_schemas.html` new `jobs` columns + `job_people`).

| Claim | Evidence |
|---|---|
| R12 one query per step (715 photos), one commit per step's writes | `tests/pipeline/test_cache_handler_batch.py::test_a_step_reads_with_one_query_not_one_per_photo`, `test_a_step_writes_in_one_commit` passed |
| R12 the L4 reuses this PC's cache (other folder, other mtimes) | `test_the_l4_reuses_what_this_pc_cached` passed; older path-keyed rows still hit; mtime check kept outside the store |
| R12 the worker's cache follows `DATABASE_URL` | `test_the_worker_cache_follows_database_url` passed |
| R13 no secret in logs, argv, run records, the worker image or status responses | `tests/architecture/test_no_secrets_leak.py`: 6 passed |
| R15 the full default suite on SQLite | `pytest tests` (worktree, pydeps): **1,772 passed, 71 skipped, 0 failed** (281 s). The skips are the private test photos, which are not in the worktree (`test_data/face_clustering`, `samples/ukbench`), plus 2 standing skips |
| R3 the whole parity set on Neon | see the parity entry below |

**Decisions / deviations**
- `load_many` no longer touches `last_accessed` on reads (nothing reads it; on Neon it was one UPDATE per photo at the next commit).
- IN lists are chunked at 5,000 paths (SQLite 3.45 here allows 32,766 parameters, Postgres 65,535), so a step over 715 photos plus their legacy keys is one query.
- `tests/storage`, `tests/scripts` have no `__init__.py`, like `tests/control`.

## 2026-11-06 — W5 Detached L4 runs landed (code + fakes; deploy and real L4 are W8)

`serving/detached_modal.py` `DetachedModalRemote` spawns `curate_job` on the DEPLOYED app by name and returns the
call id; `status` / `cancel` by call id. `ModalRunner` now uploads missing photos (`ModalVolumeRemote`, which lost
`run_job` and `download_results`), sets the stage "waiting for gpu", spawns, stores the call id on the job and
follows the job row until the worker ends it; nothing is copied back. It refuses a SQLite job DB when the remote is
the real one (the L4 cannot reach a file here). `serving/run_watchdog.py` `RunWatchdog.reconcile` fails a detached
job whose heartbeat is older than 90 s (and cancels the call), or whose call ended before it started / without
results; it re-reads the row first so a worker that just finished is never overruled. The API starts it every 30 s
when `DATABASE_URL` is set, and `PipelineService.get_status` runs it for the run asked about. `reap_orphans` skips
jobs with a call id (the process that started them may restart). `JobService.cancel` reaches Modal by the stored
call id when no runner in this process knows the job. `modal_app.py`: job functions ask for `GPU_TYPES` (from
`configs/serving.yaml`, `[L4, A10G]`), mount secret `albumify-cloud`, run `runners.worker_curator` (Neon + R2), record
the GPU on the job row and in run.json; the `albumify-cache` Volume and its copy in/out are gone. The spend cap is
checked at the dearest GPU (`ServingConfig.dearest_dollars_per_hour`, A10G $1.10/h).

| Claim | Evidence |
|---|---|
| R8 detached: call id stored, manifest travels, nothing copied back, the job row is the outcome | `tests/serving/test_modal_staging.py`: 15 passed (10 runner tests incl. `test_nothing_comes_back_and_the_job_row_is_the_outcome`, `test_the_call_id_is_stored_and_the_manifest_travels`) |
| R8 an API restart mid-run loses nothing; a restarted API does not reap the run | `test_pipeline_service_runners.py::test_an_api_restart_mid_run_loses_nothing`, `test_run_watchdog.py::test_a_restarted_api_does_not_reap_a_detached_run` passed |
| R9 lost heartbeat / dead call / ended without results -> failed with the reason; finished worker never overruled; Modal unreachable judged by heartbeat | `tests/serving/test_run_watchdog.py`: 10 passed |
| R9 cancel by call id (also after a restart); queued reads "waiting for gpu" | `test_cancel_by_call_id_after_a_restart`, `test_a_queued_run_reads_waiting_for_a_gpu`, `test_cancel_while_it_runs` passed |
| R10 GPU list L4 then A10G, cap at the dearest, GPU recorded, functions use the list + secret | `tests/serving/test_modal_gpu_choice.py`: 5 passed |
| Nothing else broke | `tests/api tests/control tests/serving tests/storage tests/streamlit`: 327 passed; `tests/pipeline/test_run_spec.py` 9 passed |

**Decisions / deviations**
- T5.1 `modal deploy` was not run (owner OK needed, W8). The code targets the deployed app `albumify-curator` / `albumify-curator-cpu`.
- Grace is 90 s and the API watchdog runs every 30 s, so a stopped heartbeat fails the job within 2 minutes (R9).
- "Waiting for a GPU" is the stage between the spawn and the worker's first write; Modal's own queue state is not read.
- `tests/e2e_fake_modal_api.py` (Playwright stand-in) now fakes the detached call with a local child worker; the Playwright E2E tests were not run here (they start an API; W8).

## 2026-11-06 — W4 Worker writes the database landed

`Job` gained `progress` (JSON), `heartbeat_at`, `call_id`, `gpu` (added to older DBs by `_add_missing_columns`).
`serving/db_progress.py` `DbProgressWriter` writes step, in-step fraction, photo n of N and the heartbeat to the job
row (fractions throttled to one write per 2 s, heartbeat thread every 5 s, a failed write never stops the run);
`progress.json` is gone (`JobStore.write_progress/read_progress/progress_path` removed). `serving/worker_persister.py`
`WorkerPersister` runs in the worker after the pipeline: job `running -> loading`, result files to the object store,
`ResultsLoader` rows + renders, the album run's rows via `RunPersister` with original paths (`write_run_rows`, the one
implementation `PipelineService` also falls back to), job `succeeded`; on any failure the worker ends the job and the
run `failed` with the reason. The manifest now carries each original's pixel size, so `PathMap` needs no original
(the L4 has none). The local child process gets the job DB URL by env var (never argv) and runs
`runners.worker_curator`. `JobService.run` reads the job row after the runner returns; `PipelineService.live` reads
only the job row. Configure & Run shows "43% (307/715)" for the current step on both machines and the GPU name.

| Claim | Evidence |
|---|---|
| R5 progress + heartbeat in the job row, throttled, never fatal | `tests/serving/test_db_progress.py`: 6 passed |
| R5 the app view is the same on both machines, no progress.json | `tests/api/test_pipeline_service_runners.py::test_in_step_progress_reads_the_same_on_both_machines[local,modal_gpu]` passed |
| R6 worker writes job rows + run rows, original paths, 0 staged paths, files in R2, on "PC" and "L4" (originals deleted, moto R2) | `tests/serving/test_worker_persister.py`: 12 passed (5 tests x 2 machines + 2) |
| R6 a failed write fails job + run, no partial run rows | `test_a_failed_run_write_fails_the_job_and_leaves_no_run_rows[pc,l4]` passed |
| Nothing else broke | `tests/api tests/control tests/serving tests/storage tests/streamlit`: 306 passed |

**Decisions / deviations**
- Job rows and run rows are two transactions, not one (two databases when `DATABASE_URL` is unset). A failure in either fails the job; `RunPersister` still leaves no partial run rows.
- A runner whose worker did not write its results (in-process test fakes, the pre-W5 Modal path) still gets them loaded by `JobService`, so both paths stay tested until W5 retires the copy-back.
- The L4 worker (`modal_app._curate_job`) switches to `worker_curator` in W5, together with the deployed app and the secret.

## 2026-11-06 — W3 Files on R2 landed

`sim_bench/storage/object_store.py`: `ObjectStore` (put, get, exists, list, delete_prefix), `LocalObjectStore` (the
job folders, layout unchanged) and `R2ObjectStore` (boto3, objects under `jobs/<job_id>/`). `make_object_store`
picks R2 only when every `R2_*` is set. `JobStore.objects` carries it; `LocalRenders` writes display images and
face crops through it, `ResultsLoader` lists it (one listing per folder, not one check per face), `RunEmbeddings`
reads `results/*` from it, and `/jobs/{id}/photos/.../image` + `/faces/.../crop` serve this job's own key only.
`DELETE /jobs/{id}` (`JobService.delete_job`) deletes the rows and every stored file of the job.

| Claim | Evidence |
|---|---|
| R7 store contract on both stores, unsafe keys refused | `tests/storage/test_object_store.py`: 30 passed (moto for R2; incl. 8 unsafe-key cases x 2 stores) |
| R7 another job's key / a made-up key is 404, on both stores | `tests/api/test_jobs_router.py::test_other_jobs_photo_is_404[local]` and `[r2]` passed |
| R7 images and crops served from R2 | `test_images_and_crops_are_served[r2]` passed |
| R14 deleting a job deletes its files | `test_delete_prefix` (both stores, 1,005-object case on R2) + `test_delete_job_removes_rows_and_stored_files[local,r2]` passed |
| SQLite default unchanged | `tests/api tests/control tests/serving tests/storage`: all passed |

**Decisions / deviations**
- Keys are job-relative (`<job_id>/display/a.jpg`); the R2 store adds the `jobs/` prefix, the local store maps a key to `<jobs_root>/<key>`, so a local run's files stay exactly where they were and the DB keys (`display/a.jpg`, `faces/12.jpg`) are unchanged.
- The staged 768 px inputs stay in the local input store and the Modal inputs Volume (a spec non-goal); `inputs/...` keys are still served from there.
- `JobStore.write_displays` (worker-side renders, debug only) is kept, not removed: it is behind `write_renders=True`, which no runner sets.

## 2026-11-06 — W2 Postgres parity landed

`tests/_db.py` gives each DB test a fresh database: a new SQLite file by default, or, in the parity run, the
throwaway Neon schema (emptied once per test). The control, API-session and router tests that build databases
use it (`control_db_url` fixture, `fresh_database_url`, `fresh_api_engine`). `tests/storage/test_neon_parity.py`
(`-m needs_neon`, excluded from the default run) reads `DATABASE_URL` from `.env` inside its fixture, creates
`test_<random>`, runs the child pytest on it through `ALBUMIFY_TEST_DATABASE_URL` and drops the schema at the end;
output is scrubbed of the URL and password.

| Claim | Evidence |
|---|---|
| R3 control + jobs API pass on Neon Postgres | parity run of `tests/control` + `tests/api/test_jobs_router.py` on a `test_*` schema: **97 passed, 0 failed** (1,377 s; schema dropped) |
| Both table sets in one Postgres schema | `test_neon_parity.py::test_both_table_sets_live_in_one_postgres_schema` passed on Neon (33 s) |
| Column migrations portable | `_migrate_new_columns` compiles the model's type per dialect; `_add_missing_columns` reads all columns in one query (`get_multi_columns`) |
| SQLite default unchanged | `tests/api tests/control tests/serving tests/storage`: 243 passed |

**Decisions / deviations**
- Postgres fixes needed: none so far (types, JSON, timezones passed as written). The whole `tests/api tests/control tests/serving` parity run is repeated at the end of W6.
- The schema is set per connection (libpq `options=-csearch_path=...`), which Neon's pooler does not pass on, so the fixture connects to the direct host (the pooled host name minus `-pooler`).
- Each Neon round trip is ~180 ms from this PC, so the parity run is slow (~14 s per DB test). `ControlDB` start now makes ~2 schema queries instead of ~15 (also helps API start on Neon).
- `tests/api/test_people_service_bbox.py` keeps its in-memory SQLite (pure unit test of bbox shapes).

## 2026-11-06 — W1 One database setting landed

`sim_bench/storage/database.py` (`DatabaseConfig`, `load_env`, `redact_url`) is the one engine factory: the API
session, `ControlDB` and (W6) the feature cache build their engines from it. `DATABASE_URL` set puts every Albumify
table in that database; unset, the API stays on `sim_bench.db` and the jobs on `control.db`. `.env` is read only
at API startup and in the local worker child. The job table `people` is now `job_people`; an older `control.db`
is renamed in place on open (only a `people` table with a `job_id` column, never the API's).

| Claim | Evidence |
|---|---|
| R1 one URL, psycopg 3 driver, pre-ping, both table sets in one DB | `tests/storage/test_database_config.py`: 10 passed (incl. `test_api_and_control_share_one_url_when_set`) |
| R2 `job_people`, rows kept, API `people` untouched | `tests/control/test_jobs.py::ut_ControlDB::test_job_people_table` passed |
| R15 unset = the two SQLite files; the suite never reads `.env` | `test_unset_is_sqlite`, `test_unset_keeps_the_two_sqlite_files`, `ut_LoadEnv` (3); `tests/conftest.py` sets `ALBUMIFY_NO_DOTENV=1` and drops `DATABASE_URL`/`R2_*` |
| Nothing else broke | `tests/api tests/control tests/serving tests/storage`: 243 passed (baseline 232 + 11 new) |

**Decisions / deviations**
- `redact_url` hides user and password (`scheme://***@host/db`), the spec-112 `_redact` format; `_redact` is kept as an alias.
- SQLite foreign keys stay off for the API's `sim_bench.db` (as before); `ControlDB` keeps them on. Postgres enforces them for both (W2 checks).
- Tests run with pydeps on `PYTHONPATH` (psycopg 3.3.8, boto3 1.43, moto 5.2.5) because nothing may be installed into the live `.venv`; `requirements.txt` lists them for the real install.

## 2026-11-06 — Drafted

Owner, after the L4 path was explained ([diagram](https://claude.ai/artifact/K3WQv7SkBsConZVnJZhh9f)): "from my perspective,
L4 should run and write results to DB. Also, DB is used for polling status and telemetry. If we lost connection we just
query DB for status and results." Then: "go with A" and "move everything to Neon".

Accounts were set up and checked the same day:

| Claim | Evidence |
|---|---|
| Neon reachable with the `.env` URL | Scratch check: PostgreSQL 18.6, pooled host, temp table write/read: PASS |
| R2 bucket usable with the scoped token | Scratch check: put, read back (match), delete in `albumify`: PASS |
| Modal secret exists | `modal secret list`: `albumify-cloud`, created 2026-11-06 11:20 |
| Leaked password not in use | `.env` password differs from the one pasted in chat |

Code map (read-only) found: nothing in the API loads `.env`; the API DB URL is always SQLite; `control.db` already
supports Postgres; two tables are both named `people`; status reaches the app through the job row, a local
`progress.json` and an in-memory dict; no Postgres driver or S3 client is installed.
