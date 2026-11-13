# spec-117 — Tasks (One cloud database: Neon + R2)

**Status:** In Progress 2026-11-06 — owner approved; W1-W6 built in worktree `D:/sim-bench-117` (branch `spec/117-neon-r2`).
**Spec:** [spec.md](spec.html) · [updates](updates.html) · roadmap stage **C** · [ROADMAP.md](../../ROADMAP.html)

Design lives in the spec (§3 plan, §4 class diagram, §5-§6 file tables). This file orders the work.
**Rule while a local album run is in progress:** no `.py` edits under the repo (the API's `--reload` kills the run).

## W0 — Owner setup (done 2026-11-06)
- [x] T0.1 Neon project `albumify` (AWS us-east-1), pooled `DATABASE_URL` in `.env`; password reset after it was pasted in chat.
- [x] T0.2 R2 bucket `albumify`, token scoped to it; `R2_*` in `.env`.
- [x] T0.3 Modal secret `albumify-cloud` (`DATABASE_URL` + `R2_*`).
- [x] T0.4 Check: Neon query + temp table, R2 put/read/delete: both PASS (scratch script, no values printed).

## W1 — One database setting (R1, R2, R15)
- [~] T1.1 `requirements.txt`: `psycopg[binary]`, `boto3`, `moto`; install into `.venv` (listed; install deferred: the live `.venv` must not change during the album run, tests use a side folder).
- [x] T1.2 `sim_bench/storage/database.py` `DatabaseConfig`; `.env` loaded once at API start (`api/main.py`) and in the worker.
      Test `tests/storage/test_database_config.py` (unset -> SQLite path as today; Postgres URL -> psycopg driver, pre-ping; redaction).
- [x] T1.3 `api/database/session.py` and `control/db.py` build their engines from `DatabaseConfig`.
- [x] T1.4 `control/models.py`: `Person.__tablename__ = "job_people"`, with a one-time rename of the existing SQLite table.
      Test `tests/control/test_jobs.py::test_job_people_table`.

## W2 — Postgres parity (R3)
- [x] T2.1 `needs_neon` pytest marker: creates a throwaway Neon branch (or schema), runs `tests/api`, `tests/control`, `tests/serving`.
- [x] T2.2 Make `_migrate_new_columns` and `_add_missing_columns` portable (SQLAlchemy inspector, no raw SQLite DDL).
- [x] T2.3 Fix what fails on Postgres (types, JSON, timezones); record each fix in updates.md.

## W3 — Files on R2 (R7, R14)
- [x] T3.1 `sim_bench/storage/object_store.py`: `LocalObjectStore`, `R2ObjectStore`. Test `tests/storage/test_object_store.py` (moto).
- [x] T3.2 `LocalRenders`, `ResultsLoader`, `RunEmbeddings`, `routers/jobs.py` use the store; per-job key check kept
      (`tests/api/test_jobs_router.py::test_other_jobs_photo_is_404` on both stores).
- [x] T3.3 Job deletion deletes `jobs/<id>/` in the store (`test_delete_prefix`).

## W4 — Worker writes the database (R5, R6)
- [x] T4.1 `Job` columns `progress` (JSON), `heartbeat_at`, `call_id`, `gpu`.
- [x] T4.2 `serving/db_progress.py` `DbProgressWriter`; steps' `report_progress` reaches it (throttled 2 s); `progress.json` retired.
      Test `tests/serving/test_db_progress.py`.
- [x] T4.3 `serving/worker_persister.py`: manifest carries original path + size; remap in the worker; `ResultsLoader` + `RunPersister`
      in one transaction. Test `tests/serving/test_worker_persister.py` (SQLite + fake store).
- [x] T4.4 `JobService.run` and `PipelineService.live/get_status` read the job row only.

## W5 — Detached L4 runs (R8, R9, R10)
- [x] T5.1 `modal deploy` of `albumify-curator` with secret `albumify-cloud`; `gpu=["L4", "A10G"]`; run record stores the GPU. (deployed 2026-11-14 from `c56d6db`, owner OK)
- [x] T5.2 `serving/detached_modal.py`: `start` (spawn by name, store call id), `status`, `cancel`.
- [x] T5.3 `serving/run_watchdog.py`: heartbeat older than 2 min or call ended without rows -> `failed` with the reason;
      queued -> "Waiting for a GPU". Test `tests/serving/test_run_watchdog.py`. Spend cap at the dearest GPU: `test_modal_gpu_choice.py`.

## W6 — One feature cache (R12)
- [x] T6.1 `UniversalCacheHandler.load_many`; steps read once per step. Test `tests/pipeline/test_cache_handler_batch.py`.
- [x] T6.2 Worker and local runs use the cache through `DATABASE_URL`; the `albumify-cache` Volume copy is removed.

## W7 — Move the data (R4)
- [x] T7.1 `scripts/migrate_to_neon.py` (`NeonMigrator`): plan, copy, verify; refuses a non-empty target.
      Test `tests/scripts/test_migrate_to_neon.py` (SQLite to SQLite). Real run: counts per table into updates.md. (script + test done; the real run waits for the owner)

## W8 — Prove on real data (R8, R11, R13, R14) — owner OK before each L4 run
- [~] T8.1 Slovakia26 on this PC against Neon + R2; Albums/Results/People/Explore; page timings. (2026-11-13: run from the app works; timings fail, SIGHTING-152; gallery bug SIGHTING-151)
- [x] T8.2 Slovakia26 on the L4: cut the PC's network 2 min, restart the API once; run finishes; results in Albums. (2026-11-14: run `f7d8ba85`, after three failed runs fixed in SIGHTING-153; the PC side was stopped, the Wi-Fi itself not cut)
- [x] T8.3 L4 re-run after the local run: cached steps recompute nothing. (2026-11-14: `fd52013c` after `f7d8ba85`, local `8ab479f9` after both; picks differ by 3 of ~330, SIGHTING-154)
- [ ] T8.4 Secret scan (`git grep`, logs, run records, image env); Neon and R2 sizes.
- [ ] T8.5 Full suite (SQLite), `needs_neon` set, Budapest gate, `/code-review` -> REVIEW.md.
