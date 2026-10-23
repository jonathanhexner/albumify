# spec-112 — Tasks (Live DB + storage, runner as config)

**Status:** In Progress — Phase A A1–A5 built 2026-10-10 (63 new tests); A6 real-data proof closes with spec-115 W5. Phase A2 moved to spec-115.
**Spec:** [spec.md](spec.html) · [updates](updates.html) · roadmap stage **C** · [ROADMAP.md](../../ROADMAP.html)

## Design notes

- **Rows in the DB, bytes in storage, compute cache local.** Storage map: spec §2.
- **Job layout unchanged.** `jobs/<id>/{input/, album.json, display/, run.json}` from spec-110 plus a
  new `results/` (`run.db`, `embeddings.npy`, face crops). `JobCurator` keeps working on a local
  directory; `JobSync` moves files in and out of the object store.
- **ObjectStore interface** (`sim_bench/serving/object_store.py`): `put_file`, `get_file`,
  `list(prefix)`, `exists`, `delete_prefix`, `download_url(key, ttl)`. Phase A: `LocalObjectStore`.
  Phase B: `R2ObjectStore` (`boto3`).
- **Worker writes storage only.** DB writes happen in `JobService` / the loader (API process or CLI).
- **Control DB = separate engine** (`sim_bench/control/`), never `sim_bench.db`, never the per-run
  `run_db` factory (they must not be cross-imported, see `sim_bench/run_db/_session.py`).
- **Loader reuses the run schema read-side:** open `results/run.db` with the `run_db` models, map to
  control-DB rows keyed by `(job_id, photo_id)` / `(job_id, face_id)`.
- **Runner** protocol: `run(job_id) -> RunOutcome`. `LocalRunner` runs `JobCurator` in a child process
  (spec-104: torch memory is only freed by process exit). `ModalRunner`: Phase A shells out to the
  existing driver path; Phase B calls the deployed function via `modal.Function.from_name`.
- New deps: Phase A none expected; Phase B `boto3`, `psycopg[binary]`, `moto` (fake S3, tests).

# Phase A — local, offline

## A1 — Object store + results bundle
- [x] TA1.1 **Moved to TB1.1.** In Phase A the local `JobStore` directory *is* the storage; an object-store interface only matters once R2 exists. Every runner leaves the job's files in the local store (Modal: `ModalVolumeRemote.download`). | 2026-10-10
- [x] TA1.2 `curate_job` writes `results/bundle.json` + `embeddings.npy` + `faces/<id>.jpg`. **Not RunExporter** (gap recorded in updates.md): it is face-centric, has no capture time/GPS/album status, needs FC-app `ClusterResult` objects, keys by absolute path. New `sim_bench/album/results_bundle.py` (`ResultsBundler.calc`); photos carry pipeline decisions so pre-scoring drops keep their reason. | 2026-10-10
- [x] TA1.3 `tests/album/test_results_bundle.py` (9), `tests/serving/test_curate_job.py` (+2), `tests/serving/test_job_store.py` (+2). Real-pipeline bundle: TA6. | 2026-10-10

## A2 — Control DB
- [x] TA2.1 `sim_bench/control/{db,models}.py`: `jobs`, `job_events`, `photos`, `faces`, `people`,
      `scenes`, `user_edits`; engine from `DATABASE_URL` (SQLite `connect_args` only for SQLite);
      portable types. Schema creation: read `alembic/env.py` first; separate Alembic config or
      `create_all` — decide and record. | 2026-10-10 `create_all` (Alembic targets sim_bench.db only); `pool_pre_ping` for Neon; SQLite FK pragma
- [x] TA2.2 Job status set `queued → staging → running → loading → succeeded | failed`; invalid
      transitions raise. (AC4)
- [x] TA2.3 `tests/control/test_jobs.py` (13). | 2026-10-10

## A3 — Loader
- [x] TA3.1 `ResultsLoader.calc(job_id)`: results bundle + `album.json` → rows; replaces the job's
      rows in one transaction (idempotent). (AC2)
- [x] TA3.2 `AlbumBuilder`: tables (+ `user_edits`) → `Album`, via the spec-106 schema. (AC3, AC5)
- [x] TA3.3 `tests/control/test_loader_builder.py` (15): counts, idempotent re-load, built album == worker album, promote/demote/rename, edits survive re-load. | 2026-10-10

## A4 — Runner config + JobService
- [x] TA4.1 `configs/serving.yaml` (`runner`, `pipeline`, `device`); env override `ALBUMIFY_RUNNER`.
- [x] TA4.2 `JobService.submit(source, runner)`: stage → record → dispatch → events → load. (AC4)
- [x] TA4.3 `LocalRunner` (child process); `ModalRunner` (stage → curate_job → `Volume.read_file` copy-back). Download verified read-only on the real `slovakia48` job (15 files). | 2026-10-10
- [x] TA4.4 `tests/control/test_job_service.py` (16). | 2026-10-10

## A5 — API + viewer
- [x] TA5.1 `sim_bench/api/routers/jobs.py`: `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/album`,
      `GET /jobs/{id}/photos/{photo_id}/display`, `POST /jobs/{id}/edits`. Typed Pydantic responses.
      | 2026-10-10 also `GET /jobs`, `/photos`, `/faces`, `/faces/{id}/crop`, `POST /jobs/{id}/load`; image route is `/photos/{id}/image`
- [x] TA5.2 API tests (SQLite + local store + fake runner).
- [x] TA5.3 Streamlit page (built: `app/streamlit/pages/jobs.py`, with promote / remove / rename buttons; Playwright check in TA6.4) —: job list → picks, dropped with reasons, people
      (real `st.button` grids for any clickable thumbnails).

## A6 — Prove Phase A
- [x] TA6.1 Slovakia26 via `JobService` on `modal_gpu`; loaded (715 photos, 987 faces). (AC1, AC2) | 2026-10-11; re-run 2026-10-23 from the app (spec-115 W5): 251 picks, 80 people, crops on every person.
- [x] TA6.2 Compare `GET /jobs/{id}/album` with the worker's `album.json`. (AC3) | 2026-10-11 identical
- [x] TA6.3 Forced failure; promote + rename edit round-trip. (AC4, AC5) | 2026-10-23 failure end states proven on real runs (L4 cancel, spec-115 W5); edits covered by API tests; a real-data edit click is a spec-115 follow-up
- [x] TA6.4 Playwright screenshots of the viewer on the Slovakia job; updates.md entry. | 2026-10-23 spec-115 final check: all four tabs, People 410/410 crops load

# Phase A2 — moved to [spec-115](../115-album-workbench/tasks.html)

W1 (done 2026-10-23) and W2-W5 are tracked there. TA6.1/TA6.3/TA6.4 close with its W5.

# Phase B — cloud, configuration only

## B0 — Accounts and secrets (owner)
- [ ] TB0.1 Neon project (free plan); pooled connection string.
- [ ] TB0.2 Cloudflare R2 bucket `albumify`; API token scoped to it.
- [ ] TB0.3 Local `.env` (gitignored, verified): `DATABASE_URL`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
      `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`.
- [ ] TB0.4 `modal secret create albumify-r2 ...` (R2 keys only). (AC7)

## B1 — Cloud backends
- [ ] TB1.1 `R2ObjectStore` + `moto` tests; `needs_r2` round-trip test on the real bucket.
- [ ] TB1.2 Run the A2/A3 test suites against Neon (`needs_neon` marker).
- [ ] TB1.3 Worker reads/writes R2 via `JobSync`; `albumify-jobs` Volume retired for job data;
      `modal deploy`; `ModalRunner` → `Function.from_name`.

## B2 — Prove Phase B
- [ ] TB2.1 Slovakia26 via `POST /jobs`, `runner=modal_gpu`, on Neon + R2: AC1–AC3 again. (AC6)
- [ ] TB2.2 Secret audit: `git grep`, image env, API responses; no-`.env` run of the suite. (AC8)

# Close out
- [ ] TC.1 `docs/architecture/{data_flow,classes,db_schemas}.html`; `sim_bench/serving/README.md`.
- [ ] TC.2 CHANGES_LOG; `updates.md` with evidence; `/code-review` → REVIEW.md.
