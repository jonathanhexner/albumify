# spec-117 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress 2026-11-06 (owner approved; W1-W6 being built in worktree `D:\sim-bench-117`)

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
