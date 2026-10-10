# spec-112 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — Phase A built (A1–A5); real-data proof (A6) pending a Modal re-run of Slovakia26.

## 2026-10-10 — Phase A A1–A5 built (offline); real-data proof pending

**What exists now** (all local, no accounts):

```
worker (JobCurator) --writes--> album.json + results/bundle.json + embeddings.npy + faces/<id>.jpg
JobService: queued -> staging -> running (runner = local | modal_gpu | modal_cpu) -> loading -> succeeded|failed
ResultsLoader: files -> control DB rows (jobs, job_events, photos, faces, people, scenes)
AlbumBuilder:  rows + user_edits -> Album (== worker album.json when there are no edits)
API /jobs ...  -> Streamlit "Jobs" page (picks, dropped by reason, removed-before-scoring, people)
```

**Decisions made while building**
- **Results bundle instead of `RunExporter`.** The exporter is face-centric (its `images` table is
  derived from faces and has no capture time, GPS or album status), needs the face-clustering app's
  in-memory `ClusterResult` objects, and keys by absolute path. A purpose-built bundle
  (`sim_bench/album/results_bundle.py`) reads the context directly; each photo row carries the
  pipeline's own decisions, so photos removed before scoring keep their reason (SIGHTING-128 data,
  without choosing the reason code: they are `in_album = false`, `status = null`).
- **ObjectStore deferred to Phase B** (TA1.1 → TB1.1): in Phase A the local job directory is the
  storage, and every runner copies the job's files there.
- **Album contract:** `Person.label` added (owner-given name; "" by default). Additive.
- **Control DB** uses `create_all` (Alembic manages only `sim_bench.db`).
- **Viewer is not read-only:** promote / remove / rename buttons write `user_edits` (AC5 asked for
  the API; the page uses it).

**Blocked**
- **SIGHTING-131 (cleared):** the dev box's commit charge sat at its limit (126.6 / 127.9 GB), so
  every local model load failed (`bad allocation`, torch access violation in DINOv2). It cleared on
  its own the same evening without a reboot; `tests/pipeline` then passed (183) including the two
  tests that had crashed. Intermittent, cause unknown.
- **AC1 needs one Modal run** of Slovakia26 with the bundle (the 2026-10-09 run predates it).

| How it was verified | Evidence |
|---|---|
| Bundle mapping: ids, scores, NaN, straightened ids, decisions of pre-scoring drops, faces → person, embeddings | `tests/album/test_results_bundle.py` (9) |
| Worker writes the bundle; a failed run writes none; store round-trip | `tests/serving/test_curate_job.py` (+2), `tests/serving/test_job_store.py` (+2) |
| Status set, AC4 (a crash ends `failed` with the error), re-run, password never logged | `tests/control/test_jobs.py` (13) |
| AC2 counts + idempotent re-load; AC3 built album == worker album; AC5 promote/demote/rename, edits survive re-load | `tests/control/test_loader_builder.py` (15) |
| Same flow for `local` and `modal_gpu`; 768 px staging; failures; child-process contract; Modal stage→curate→download order; config + env | `tests/control/test_job_service.py` (16) |
| API: submit → succeeded with history; AC3 + AC5 over HTTP; images/crops served; 404/409/422 | `tests/api/test_jobs_router.py` (8) |
| Regression: serving + album + control + api suites | 132 passed |
| Modal copy-back on a real Volume (read-only, no compute) | `ModalVolumeRemote.download("slovakia48")` → 15 files (album, 13 displays, run.json), `input/` skipped |
| Regression after the memory cleared: `tests/pipeline` | 183 passed |
| Real pipeline → bundle → DB → viewer | **Not yet** (TA6: Modal re-run of Slovakia26) |

## 2026-10-10 — Spec amended: local first, full storage map

- **Local first (owner):** Phase A builds everything on a local SQLite file + local folder, no
  accounts; Phase B switches to Neon + R2 by configuration only.
- **Full storage map in v1 (owner):** `photos`, `faces`, `people`, `scenes`, `user_edits` become DB
  rows keyed by `job_id`; bytes (inputs, displays, face crops, worker results) in storage; feature
  cache and model weights stay local. `album.json` becomes an export built from the tables.
- **Reuse found:** `RunExporter` (`sim_bench/run_db/exporter.py`) already writes the normalized
  per-run schema (images, faces, face_scores, clusters, scene clusters, filter/merge decisions,
  embeddings, crops) but only the face-clustering path calls it. The worker will call it and write a
  `results/` bundle; a loader maps it into the shared DB.
- **Embeddings** stay in the results bundle for v1 (no `pgvector`).
- **Not affected:** the 2026-10-09 Slovakia run predates the bundle; Phase A AC1 needs one re-run.

## 2026-10-10 — Spec drafted

- **Why:** the Slovakia26 full GPU run (2026-10-09) could not be seen in the FastAPI app. Both
  paths call the same `execute_spec`, but each persists to its own place (`sim_bench.db` on the PC
  vs `album.json` on a Modal Volume) and hard-wires where it runs.
- **Decision (owner):** Neon Postgres (free plan) + Cloudflare R2. Where the pipeline runs becomes a
  `runner` setting. Considered: Vercel Blob (weaker Python support, egress billed on every GPU read),
  Supabase (pauses after 1 week idle — bad for a portfolio URL).
- **Scope taken from spec-106:** T2.1 (object store) and the DB half of T3.1 (`jobs`, `job_events`).
- **Next:** owner approves the spec and does P0 (Neon project, R2 bucket + token, `.env`,
  `modal secret`).

| How it was verified | Evidence |
|---|---|
| — | Nothing implemented yet |
