# spec-112 — Live DB + storage, runner as config

**Status:** In Progress (Phase A; Phase A2 "album workbench" approved 2026-10-23, W1 in progress)
**Owner:** Jonathan / Claude
**Priority:** P1 (Stage C critical path — every later product piece reads from here)
**Roadmap:** Stage C · see [ROADMAP.md](../../ROADMAP.html)
**Relation:** takes over [[spec-106]] T2.1 (object store) and the DB half of T3.1 (`jobs` /
`job_events`); builds on the [[spec-110]] job layout (`input/`, `album.json`, `display/`, `run.json`)
and reuses the normalized per-run schema written by `sim_bench/run_db/exporter.py` (`RunExporter`).
spec-106 keeps presigned browser upload (T2.2), the state-machine logic, the browser-facing API,
and the Next.js front end.

**Decisions (owner, 2026-10-10):**
- Neon Postgres (free plan) for the DB, Cloudflare R2 for storage.
- **Local first:** Phase A builds everything on local SQLite + a local folder; Phase B switches to
  Neon + R2 by configuration only.
- Full storage map in v1: photos, faces, people, scenes and user edits are DB rows, not only
  fields inside `album.json`.
- Face embeddings stay in the worker's results file in storage (not DB columns) for v1.

**Decisions (owner, 2026-10-23) — Phase A2:**
- A stable way to run and examine albums, locally or on the L4, comes before Gate 1: the picks are
  judged in this app.
- Cloud and local must behave the same: only where the computation runs differs. No pictures come back
  from the worker; the app shows the PC's own copies.
- Re-runs on the L4 must reuse what was already computed (the feature cache stays with the worker).

## 1. Problem

The same pipeline (`execute_spec`) runs in two places that never share results:

```
FastAPI app:    local folder -> execute_spec -> sim_bench.db on the PC        -> Streamlit
Modal worker:   Modal Volume -> execute_spec -> album.json on the Modal Volume -> (nothing)
```

Product data is also scattered over three stores: `sim_bench.db` (JSON-blob results mixed with the
feature cache), one `face_clustering.db` file per face-clustering run (normalized, but loose files),
and `album.json` (the curated result, in no DB). So the 715-photo Slovakia26 GPU run is invisible
to the app, nothing is queryable across jobs, and "where it runs" is baked into each path.

## 2. Goal

**One rule: rows go in the DB, bytes go in storage, the compute cache stays with the compute** (on the
PC for local runs, on a Modal Volume for L4 runs; Phase A2). Where the pipeline runs is a setting.

```
            API (control plane) --- writes ---> DB   (SQLite file in Phase A, Neon in Phase B)
                    |
                    |  runner: local | modal_gpu | modal_cpu        <- config
          +---------+---------+
     local runner        Modal worker (L4)
          |                   |
          +-- read input/, write results/ + album.json + display/ --+
                              |
                         STORAGE  (local folder in Phase A, R2 in Phase B)
```

The **worker writes storage only, never the DB.** It writes a results bundle; the API loads the
bundle into the DB. The worker then needs no DB credentials, and local and Modal runs behave the same.

### Storage map

| DB tables (rows) | Storage keys (bytes) | Local only (not product data) |
|---|---|---|
| `jobs` — id, status, runner, n_inputs, timings, error | `jobs/<id>/input/<photo>.jpg` (768 px) | pipeline feature cache (`universal_cache`) |
| `job_events` — job_id, ts, kind, message | `jobs/<id>/display/<photo>.jpg` | model weights (Modal Volumes) |
| `photos` — (job_id, photo_id), storage key, taken_at, gps, scores, `status` pick/dropped, `drop_reason`, scene_id, order | `jobs/<id>/faces/<face>.jpg` (crops) | legacy `sim_bench.db` tables (unchanged) |
| `faces` — (job_id, face_id), photo_id, bbox, person_id, quality scores | `jobs/<id>/results/run.db` + `embeddings.npy` (worker output) | |
| `people` — (job_id, person_id), label, face_count | `jobs/<id>/album.json` (export, see below) | |
| `scenes` — (job_id, scene_id), time range, n_photos | | |
| `user_edits` — job_id, ts, kind (promote / rename / merge / split), payload | | |

`album.json` stops being the source of truth: the API builds it from the tables (the worker's copy
remains as a debug artifact and for byte-comparison in spec-110 AC4).

## 3. Non-goals

- Browser upload / presigned URLs, rate limits, TTL purge job (spec-106 P2/P3/P6).
- Next.js front end (spec-106 P4).
- Migrating the existing `sim_bench.db` tables (Albumify blobs, metric rows, people, overrides).
  They stay local and untouched; the local tools keep using them.
- Embeddings in the DB / cross-trip person search (`pgvector`) — v1.5.

## 4. Scope

**Phase A — local, offline (no accounts):**
1. **Object store interface** with a local-directory backend; the spec-110 job layout as keys.
2. **Worker results bundle:** the worker's job run (`curate_job`) also calls `RunExporter` and writes
   `results/` (normalized SQLite + embeddings + face crops) next to `album.json`.
3. **Control-plane DB** at `DATABASE_URL` (default `sqlite:///~/.sim_bench/control.db`), separate
   engine from `sim_bench.db`, SQLAlchemy models portable to Postgres, every table keyed by `job_id`.
4. **Loader:** results bundle + `album.json` → `photos`, `faces`, `people`, `scenes` rows. Idempotent
   (re-load replaces the job's rows).
5. **Runner config:** `runner: local | modal_gpu | modal_cpu` in `configs/serving.yaml` (env override).
   A `JobService` stages inputs, records the job, dispatches, records events, loads results.
   In Phase A `modal_gpu` keeps using the `albumify-jobs` Volume and the driver pulls results to the
   local store.
6. **FastAPI endpoints:** `POST /jobs` (local folder), `GET /jobs/{id}` (status + events),
   `GET /jobs/{id}/album` (built from tables), `GET /jobs/{id}/photos/{photo_id}/display`,
   `POST /jobs/{id}/edits` (promote / rename person).
7. **Viewer (O1):** thin read-only Streamlit page over the API: job list → picks, dropped with reasons,
   people.

**Phase A2 — album workbench (owner, 2026-10-23):** run, watch, re-run and examine albums from the app,
locally or on the L4, without moving pictures back and forth.

```
 today                                            Phase A2
 -----                                            --------
 stage: resize to 768 px into jobs/<id>/input/    stage into a content store: inputs/<sha>.jpg
 upload all 715 again for every run               upload only photos the worker has not seen
 L4 cache inside the container: lost at exit      L4 cache on a Modal Volume: kept across runs
 download 993 files one by one (19 min)           download ~5 files (album.json, run.json, results/)
 display/ + face crops come back from Modal       made on the PC from its own 768 px copies + bboxes
 start a job: API / scripts only                  "Run album" form, live status, Re-run, Cancel
 uploaded photos stay on the Volume forever       deleted after N days or on demand (O3)
```

10. **Run from the app.** Jobs page form: folder, photo range (start, n), runner (local / L4 / Modal CPU),
    pipeline. Calls the existing `POST /jobs`. The L4 path shows an estimate (photos to upload,
    estimated GPU minutes) and asks for confirmation; a per-job photo limit (config) refuses larger jobs.
11. **Live status:** the page refreshes itself while a job is queued/staging/running/loading and shows the
    stage, the latest event and elapsed time. **Cancel** stops a queued or running job (local: kill the
    child; Modal: cancel the function call); the job ends `failed` with "cancelled by owner".
12. **Re-run:** a button on any finished job refills the form with its source, range and pipeline; the new
    job is a separate job (old results stay), and Compare runs opens on the pair when it finishes.
13. **Content-addressed inputs.** Staging writes `inputs/<sha256>.jpg` (hash of the original file bytes +
    the staging version, 768 px q85), and the job keeps a manifest `photo_id -> sha`. Locally under
    `~/.sim_bench/inputs/`; on Modal on a Volume `albumify-inputs`. The runner uploads only hashes the
    Volume lacks. Because the pipeline cache keys images by path, a stable path per photo is what lets
    a re-run hit the cache (true locally too: today every new job id is a new path).
14. **Worker feature cache on a Volume** (`albumify-cache`): the worker's `sim_bench.db` feature cache
    (faces, embeddings, quality, scene features) lives on a Volume and is committed after each job, so a
    re-run recomputes only steps whose inputs or settings changed.
15. **Results-only copy-back.** The worker writes only `album.json`, `run.json` and `results/` (`bundle.json`,
    embeddings); no `display/`, no `faces/` crops. The PC builds display images and face crops from its
    own 768 px copy plus the bundle's bboxes (simple bbox crop with margin, not the worker's aligned crop).
    Local runs use the same code path, so both runners leave identical job folders.
16. **Run settings recorded.** `run.json` gains the pipeline config as run, the code commit, the runner and
    device, and the model names/versions; the Jobs page shows them, and Compare runs shows what differs.
17. **Input retention on Modal:** `albumify-inputs` entries not used for N days are deleted (scheduled
    Modal function); a "Delete my photos from Modal" button removes them now (O3).

**Phase B — cloud, configuration only:**
8. R2 backend for the object store (`boto3`, S3 protocol); short-lived download URLs.
9. `DATABASE_URL` → Neon; the worker reads/writes R2 (Volume retired for job data); `modal deploy`
   so the API calls the worker by name; `modal.Secret` with R2 keys only.

## 5. Acceptance criteria

**Phase A**
- **AC1** A Slovakia26 job run on Modal (`runner=modal_gpu`) is loaded into the local DB and visible
  in the viewer: every pick, every dropped photo with its reason, people, scenes.
- **AC2** `photos` / `faces` / `people` / `scenes` row counts match the job's `album.json` and
  results bundle (no photo or face lost in loading); re-loading the same job leaves the counts
  unchanged.
- **AC3** `GET /jobs/{id}/album` built from tables equals the worker's `album.json` (same picks,
  order, reasons, scores).
- **AC4** A failed run (bad input or raised exception) ends `failed` with the error in `job_events`,
  not stuck in `running`.
- **AC5** A promote / rename edit through the API is stored in `user_edits` and reflected in the
  next `GET /jobs/{id}/album`.

**Phase A2**
- **AC9** From the app alone (no script, no curl), a Slovakia26 job is started on the L4, its status
  updates without a manual refresh, and the album opens when it finishes.
- **AC10** No picture is downloaded from Modal: the job's copy-back is at most 10 files, and the Jobs page
  still shows every pick, dropped photo and face crop. Copy-back for Slovakia26 takes under 1 minute
  (was 19.4).
- **AC11** A re-run of the same photos on the L4 uploads 0 photos and reuses the cache: face detection and
  embedding are not recomputed (cache hits in the worker log; GPU time well below the first run).
- **AC12** A local job and an L4 job on the same photos leave the same set of files in their job folders;
  Compare runs on the pair reports the same picks (spec-110 CPU/GPU parity).
- **AC13** Cancel ends a running job `failed` ("cancelled by owner") within 30 s, on both runners.
- **AC14** `run.json` names the pipeline config, commit, runner/device and model versions; Compare runs
  shows the differing settings of the two jobs.
- **AC15** "Delete my photos from Modal" leaves no file of that job's photos on `albumify-inputs`.

**Phase B**
- **AC6** Switching only `DATABASE_URL` and the storage root to Neon + R2, the same job runs with
  `runner=modal_gpu` and passes AC1–AC3 against the cloud DB and bucket.
- **AC7** The worker holds no DB credentials (the `modal.Secret` has R2 keys only).
- **AC8** No secret in git, the Modal image, or any API response. With no `.env`, the app and the
  test suite run fully on local SQLite + local folder.

## 6. Open decisions

- **O3 — decided (owner, 2026-10-23): 7 days.** Inputs on Modal unused for 7 days are deleted; the cache
  Volume holds derived features, not photos, and is kept.
- **O4 — decided (owner, 2026-10-23): pipeline name only** in v1; per-parameter overrides come later with
  the profile work.

- **O2** R2 lifecycle (auto-delete) rule: set now (e.g. delete `jobs/*` after 30 days, showcase trips
  under `seed/` never expire) or with spec-106 TTL work. Rec: with spec-106; nothing anonymous is
  uploaded before then.

## 7. Dependencies & risks

- **Phase A2 — stale cache:** the cache stores `model_version` but lookups have ignored it before
  (spec-079: stale detections without pose). A persistent L4 cache makes that worse, so A2 makes the
  lookup include the model version, with a test, before the cache Volume goes live.
- **Phase A2 — one writer:** a SQLite cache on a Modal Volume is safe for one job at a time. v1 runs L4
  jobs one at a time (the API queues the second); parallel L4 jobs need a different cache store.
- **Phase A2 — Phase B overlap:** `albumify-inputs` is a Volume in Phase A2; Phase B moves inputs to R2
  with the same content keys, so nothing is designed twice.
- **Phase A2 spend:** the proof needs two L4 runs of Slovakia26 (first run, then the re-run for AC11), about
  10 GPU-minutes for the first, less for the second. Owner approval before each.

- **Phase B owner setup:** Neon project + connection string; Cloudflare account, R2 bucket, scoped
  token; local `.env`; `modal secret create`. Claude cannot create accounts.
- **Local default-pipeline runs are blocked** by SIGHTING-127 (RAM++ segfault). Phase A proves the
  local runner with `minimal_pipeline`; full-pipeline evidence comes from Modal runs.
- **Slovakia re-run needed:** the 2026-10-09 run predates the results bundle, so AC1 needs one new
  Modal run of the trip (about 6 min on the L4).
- **Schema rekey:** `RunExporter` keys images by file path; the DB keys by `(job_id, photo_id)`.
  The loader translates; the exporter itself is not changed for the face-clustering app.
- **SQLite → Postgres portability:** booleans stored as INTEGER and timestamps as TEXT in the run
  schema; the control DB uses portable SQLAlchemy types and Phase B runs the same tests on Neon.
- Neon free plan sleeps after 5 min idle (about 1 s wake) and suspends at the monthly compute cap.
  R2 free plan: 10 GB; one trip at 768 px plus crops is about 0.1–0.2 GB.

## 8. Definition of done

AC1–AC8 verified (Phase A on local, Phase B on real Neon + R2 + Modal), and AC9–AC15 for Phase A2
(real Slovakia26 L4 runs from the app, Playwright on the Jobs page); `updates.md` names the
evidence; `docs/architecture/{data_flow,classes,db_schemas}.html` updated; CHANGES_LOG;
`/code-review` → REVIEW.md with no High findings. See `tasks.md`.
