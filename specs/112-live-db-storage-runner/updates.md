# spec-112 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — Phase A proven on Slovakia26 via Modal for AC1–AC4 (AC5 by tests); SIGHTING-132 fixed by spec-113, the re-run is folded into Phase A2's proof. Phase A2 (album workbench) approved 2026-10-23 (O3 = 7 days, O4 = pipeline name only); W1 in progress.

## 2026-10-23 — Phase A2 drafted: album workbench (run, watch, re-run, examine; no pictures back from Modal)

Owner: a stable app to run and examine albums, locally or on cloud GPUs, comes before Gate 1. Asked
whether pictures travel back and forth and whether the cache stays on the L4. Checked in the code:
- **Pictures go both ways today.** Up: 715 resized copies per run (unavoidable once; the L4 needs pixels).
  Down: 993 files, of which 147 display copies and 842 face crops (19.4 of ~31 min); only
  `album.json`, `run.json` and `results/` (3 files) carry information the PC lacks.
- **The L4 keeps only model weights.** The feature cache (`sim_bench.db`) lives in the container and is
  lost at exit, so every L4 re-run recomputes everything. Locally too, a new job id means new input
  paths and the path-keyed cache misses.
- The app never reads from Modal (local files + control DB only). Uploaded inputs stay on the
  `albumify-jobs` Volume after the job.

Plan: spec §4 items 10–17, AC9–AC15, tasks W1–W5. Earlier 3–4 day estimate was inflated by bundling
hardening (jobs outside the API process); that is out of A2. Open: O3 (input retention, rec 7 days), O4
(re-run settings: pipeline name only in v1).

| Claim | Evidence |
|---|---|
| Download is mostly pictures | `~/.sim_bench/jobs/slovakia26`: display 147, faces 842, results 3, album.json, run.json |
| Worker cache not persistent | `sim_bench/serving/modal_app.py`: Volumes only for models + HF cache; `sim_bench.db` under the container's home |
| Cache keyed by path | `sim_bench/api/database/models.py` `universal_cache` unique (`image_path`, `feature_type`, `model_name`) |

## 2026-10-11 — Real-data proof: Slovakia26 (715 photos) through JobService on the L4

`JobService.submit(source=D:/Slovakia26, job_id=slovakia26, runner=modal_gpu)`: one call, same code path as
`runner=local`. Job `succeeded`; history `queued → staging → running → loading → succeeded`.

| Phase | Time |
|---|---|
| staging (768 px, local) | 1.6 min |
| upload + GPU container + pipeline (411 s = 0.57 s/img) | ≈ 10 min |
| **copy-back: 993 files, one `Volume.read_file` each** | **19.4 min** |
| load into the DB | 0.3 s |

**Found by looking at the page, not the counts**
- **Face → person mapping was empty on real data** (all 987 faces `person_id = null`). The bundler matched by
  `face_id`; the real `people_clusters` comes from the deprecated `cluster_people`, whose members have
  `original_path` + `face_index` and no `face_id`. Fixed: match by (photo id, face index), which both shapes
  carry; `face_index` added to the bundle; regression test with the legacy shape. Needs a re-run to repopulate.
- **SIGHTING-132 (new, pre-existing):** that deprecated clusterer is in every default run. `identity_refinement`
  lists it in `depends_on`, and the builder's auto-resolve adds it, so it overwrites the spec-079 chain's
  people (80 → 118). Owner's call before the re-run.
- **"Removed before scoring" was mislabeled.** All 89 were scored, then removed by `filter_quality`
  ("Sharpness 0.08 < threshold 0.2"). The viewer showed the first non-"passed" decision (a person
  *detection*); it now shows only `rejected` decisions, and the tab is "Removed by filters".
- **145 faces without a crop** are all gate-rejected faces (79 blur, 66 per-photo cap), none in a person;
  the worker now logs that at INFO unless a face in a person lacks a crop.
- **Copy-back dominates wall time.** Proposed: one Modal call that returns the job's output as a single
  archive (Phase B's R2 path removes the issue too).

| How it was verified | Evidence |
|---|---|
| AC1 visible in the viewer: picks, dropped by reason, removed with reason, people | `reports/2026-10-11_spec112_jobs_page/{1_picks,2_dropped,3_removed,4_people}.png` (Playwright, all job images loaded, no page errors). People thumbnails missing → the mapping bug above |
| AC2 every photo and face is a row; re-load idempotent | 715 photos (626 in album: 147 pick / 423 near_dup / 56 set_aside; 89 removed), 987 faces, 118 people, 96 scenes == bundle + album; re-load: identical counts, 0 warnings |
| AC3 album from tables == worker album | `GET /jobs/slovakia26/album` == `album.json` (created_at excluded): **True** |
| AC4 failures end `failed` | tests only (`tests/control/test_job_service.py`); no real failure occurred |
| AC5 edits | tests only (fixtures + API); not exercised on the real job, to leave the owner's album untouched |
| Mapping fix | `tests/album/test_results_bundle.py::test_people_from_the_legacy_clusterer_are_matched_by_photo_and_face_index`; 114 passed (album, serving, control, jobs API) |

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
