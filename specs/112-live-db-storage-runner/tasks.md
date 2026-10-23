# spec-112 — Tasks (Live DB + storage, runner as config)

**Status:** In Progress — Phase A A1–A5 built 2026-10-10 (63 new tests); A6 real-data proof folded into Phase A2's proof (W5). Phase A2 (album workbench) approved 2026-10-23 (O3 = 7 days, O4 = pipeline name only); W1 in progress.
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
- [x] TA6.1 Slovakia26 via `JobService` on `modal_gpu`; loaded (715 photos, 987 faces). (AC1, AC2) | 2026-10-11. Re-run needed after SIGHTING-132 + the face→person fix.
- [x] TA6.2 Compare `GET /jobs/{id}/album` with the worker's `album.json`. (AC3) | 2026-10-11 identical
- [ ] TA6.3 Forced failure; promote + rename edit round-trip. (AC4, AC5) → done in TW5.4
- [>] TA6.4 Playwright screenshots of the viewer on the Slovakia job; updates.md entry. | 2026-10-11 four tabs captured; People thumbnails pending the re-run → TW5.1

# Phase A2 — album workbench (spec §4 items 10–17, AC9–AC15)

Order: the cache and input changes first (they change what a job folder contains), the app last.

## W1 — Stable inputs and a safe cache
- [x] TW1.1 Feature-cache lookup includes `model_version`; test: a cached entry from another model
      version is a miss. (risk: stale cache) | 2026-10-23 the check already existed (spec-079, opt-in per
      step) but had no test: `tests/pipeline/test_cache_model_version.py` (4)
- [x] TW1.2 Content store `InputStore`: `put(original) -> sha` (hash of original bytes + staging
      version), `path(sha)`, `has(sha)`; local root `~/.sim_bench/inputs/`. Staging writes there and the
      job gets `input/manifest.json` (`photo_id -> sha`) instead of copies. | 2026-10-23
      `sim_bench/serving/input_store.py`; keys `<sha16>/<name>.jpg` under `<jobs_root>/_inputs`; `JobStore`
      manifest + `resolve()` (API file route); older jobs with `input/*.jpg` still work.
      `tests/serving/test_input_store.py` (13)
- [x] TW1.3 `JobCurator` reads inputs through the manifest; the pipeline sees the stable
      `inputs/<sha>.jpg` paths. Test: a second local job on the same photos hits the cache (no
      recompute of a cached step). | 2026-10-23 no temp copy; `discover_images` takes given paths;
      **`run_pipeline` now attaches the feature cache** (it had none). Tests +5. Real: 12 Budapest photos,
      two local jobs: job 2 staged 0; IQA 1017->26 ms, AVA 1980->0, scene embedding 7719->10; same picks

## W2 — Results-only copy-back, display and crops on the PC
- [ ] TW2.1 Worker stops writing `display/` and `faces/` (flag kept for the local debug path only if
      needed); writes `album.json`, `run.json`, `results/`.
- [ ] TW2.2 Loader step `LocalRenders`: display images from the job's inputs, face crops from bbox +
      margin; written into the job folder before the DB load. Same code for every runner. (AC10, AC12)
- [ ] TW2.3 `ModalVolumeRemote.download` copies `album.json`, `run.json`, `results/*` only.
- [ ] TW2.4 Tests: job folder file set identical for local and fake-Modal runners; crops match bboxes.

## W3 — Modal: inputs Volume, cache Volume, retention
- [ ] TW3.1 Volume `albumify-inputs` (content-addressed); `stage` uploads only missing hashes
      (one listing, then batch upload). (AC11)
- [ ] TW3.2 Volume `albumify-cache`; the worker points its `sim_bench.db` there, `reload()` before,
      `commit()` after each job. L4 jobs run one at a time (the API queues). (AC11)
- [ ] TW3.3 Retention: scheduled Modal function deletes inputs unused for 7 days (O3); API
      `DELETE /jobs/{id}/remote-inputs` + button "Delete my photos from Modal". (AC15)
- [ ] TW3.4 Tests with the fake remote: second stage uploads 0; delete removes only that job's hashes
      not used by another job.

## W4 — The app
- [ ] TW4.1 Jobs page "Run album" form (folder, start, n, runner, pipeline) → `POST /jobs`; L4 path:
      estimate (photos to upload, GPU minutes from the last runs' rate) + confirm; photo limit from
      `configs/serving.yaml`. (AC9)
- [ ] TW4.2 Auto-refresh while the job is live (`st.fragment(run_every=...)`): stage, latest event,
      elapsed. (AC9)
- [ ] TW4.3 Cancel: `POST /jobs/{id}/cancel`; `LocalRunner` kills the child, `ModalRunner` cancels the
      function call; job → `failed` "cancelled by owner". (AC13)
- [ ] TW4.4 Re-run button refills the form; on success, a link opens Compare runs on the pair. (AC12)
- [ ] TW4.5 `run.json` records pipeline config, commit, runner/device, model versions; Jobs page shows
      them; Compare runs lists differing settings. (AC14)
- [ ] TW4.6 Playwright (real API + app): start a local `minimal_pipeline` job from the form, watch it
      finish, cancel a second one, re-run → Compare opens. Screenshots stay on disk.

## W5 — Prove Phase A2 (owner approval before each L4 run)
- [ ] TW5.1 Slovakia26 from the form on `modal_gpu`: AC9, AC10 (copy-back time and file count), plus
      Phase A's AC1–AC3 again (closes TA6.1/TA6.4 with the post-SIGHTING-132 pipeline).
- [ ] TW5.2 Same photos re-run on the L4: AC11 (0 uploads, cache hits, GPU time).
- [ ] TW5.3 Local vs L4 on a 48-photo slice: AC12 via Compare runs.
- [ ] TW5.4 Forced failure + promote/rename round-trip (TA6.3) and Cancel on both runners (AC4, AC5, AC13).
- [ ] TW5.5 Budapest gate not affected (no v2 tab change) — state it; full suite; `updates.md` entry.

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
