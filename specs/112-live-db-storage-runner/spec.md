# spec-112 — Live DB + storage, runner as config

**Status:** In Progress (Phase A)
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

**One rule: rows go in the DB, bytes go in storage, compute cache stays local.**
Where the pipeline runs is a setting.

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

**Phase B**
- **AC6** Switching only `DATABASE_URL` and the storage root to Neon + R2, the same job runs with
  `runner=modal_gpu` and passes AC1–AC3 against the cloud DB and bucket.
- **AC7** The worker holds no DB credentials (the `modal.Secret` has R2 keys only).
- **AC8** No secret in git, the Modal image, or any API response. With no `.env`, the app and the
  test suite run fully on local SQLite + local folder.

## 6. Open decisions

- **O2** R2 lifecycle (auto-delete) rule: set now (e.g. delete `jobs/*` after 30 days, showcase trips
  under `seed/` never expire) or with spec-106 TTL work. Rec: with spec-106; nothing anonymous is
  uploaded before then.

## 7. Dependencies & risks

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

AC1–AC8 verified (Phase A on local, Phase B on real Neon + R2 + Modal); `updates.md` names the
evidence; `docs/architecture/{data_flow,classes,db_schemas}.html` updated; CHANGES_LOG;
`/code-review` → REVIEW.md with no High findings. See `tasks.md`.
