# spec-110 — Tasks (Containerized worker on Modal, CPU)

**Status:** In Progress — AC1/AC2 met; **AC3 met** on the owner's 48-photo Modal GPU job run
(storage-reference input to a valid normalized `album.json`). AC4 failed on that run only on pick
order, the cause (SIGHTING-126) is fixed, and one re-run closes it. AC5 unproven. New gap:
SIGHTING-128 (album omits photos dropped before scoring). Evidence: [updates](updates.html).
**Builds on:** the `sim_bench/serving/modal_app.py` scaffold (spec-109). This spec owns the **worker
structure + CPU image + storage-reference curate**; spec-109 owns the GPU variant.
**Decision 2026-10-09:** owner chose the shared GPU image for acceptance rather than building a separate CPU dependency set. `curate_cpu` retains the CPU proof; account auth and GPU billing are working.

## P0 — Prove the Modal account/path
- [x] T0.1 `modal setup` (user) → `modal run …/modal_app.py` → `gpu_smoke` returns. (AC1) | 2026-10-09 NVIDIA L4, CUDA available

## P1 — CPU image
- [x] T1.1 Define the shared `modal.Image`: pipeline deps that work together on a clean container
      (torch-CUDA, onnxruntime-gpu, insightface, ultralytics, transformers, timm, pyiqa+numpy<2, ram --no-deps,
      mediapipe, opencv-headless) + `add_local_python_source(sim_bench, face_cluster, geo_cluster)`.
- [x] T1.2 First build on Modal; iterate until the pipeline imports cleanly in-container. (AC2) | 2026-10-09 imports and full CPU/GPU runs
- [>] T1.3 Upload model weights to the Volume (`modal volume put`); confirm reuse across runs. (AC5)
      | 3.1 GB uploaded; RAM++ loads from Volume. **2026-10-09:** the leak is fixed in code —
      `_prepare_caches()` sets `HF_HUB_CACHE`, `TORCH_HOME` (torch.hub: DINOv2/GeoCalib/pyiqa) and
      `INSIGHTFACE_HOME`, and the job functions commit `models_vol`. `INSIGHTFACE_HOME` is read via
      `sim_bench/pipeline/utils/insightface_home.py` by **both** `FaceAnalysis` call sites — detection
      (`insightface_pipeline/face_analyzer.py`) *and* embeddings
      (`face_embedding/insightface_native.py`, found while closing this task; missing it would have
      kept buffalo_l downloading for `extract_face_embeddings`). 9 tests.
      Reuse itself stays unproven until two consecutive remote runs show the second not downloading.

## P2 — Storage-reference curate (CPU)
- [x] T2.1 Replace inline-bytes with `curate(job_id)`: pull the job's 768px images from storage/Volume →
      `execute_spec(...)` → write `album.json` + keeper displays back. Idempotent. | 2026-10-09
      `job_store.py` (a job *is* a directory: `input/`, `album.json`, `display/`, `run.json`),
      `curate_job.py` (`JobCurator.calc`: scratch copy → plan → `normalize()` → write),
      `pipeline_runner.py` (`SpecBuilder`: config + per-step device placement, lifted out of
      `modal_app._curate`). Modal side: `stage_job` / `curate_job` / `curate_job_cpu` / `fetch_album`
      on volume `albumify-jobs`. 41 tests in `tests/serving/`.
- [>] T2.2 Run on a small real trip (≤40 imgs staged in storage); assert a valid `album.json`
      equivalent to the local CPU run (within SIGHTING-122 tolerance). (AC3)
      **Met 2026-10-09 on Modal** (owner's run, GPU, 48 Slovakia26 images, `--job-id slovakia48`):
      48 staged produced an `album.json` with 13 picks, 25 `near_dup` dropped-with-reason, 7 scenes,
      10 people, a full theme, and every score field populated on every pick; displays written;
      `fetch_album` read it back. Evidence: `reports/_modal_job_slovakia48.json`.
      Local no-spend proof (12 images, `minimal_pipeline`): `reports/_local_job_localproof12.json`.
      **Caveat:** the full pipeline still cannot run locally (`RamTagger` segfaults, SIGHTING-127), so
      a local-vs-remote comparison on the *default* pipeline is impossible on this box; the remote
      CPU/GPU pair from the inline runs stays the equivalence evidence.
      **Found by this run:** SIGHTING-128 — 48 staged but `counts.uploaded = 38`; photos dropped before
      scoring are missing from the album. Needs the owner's call on the reason code.
- [>] T2.3 Re-run same `job_id` → identical result (idempotence). (AC4)
      Proven locally after fixing SIGHTING-126 (scene ids + pick order were not stable across runs):
      two consecutive `JobCurator` runs over the same staged inputs now return identical albums
      (`created_at` excluded). The owner's 48-photo GPU run, made *before* the fix, independently
      reproduced the bug: identical pick set, scene ids, scores, dropped reasons and theme, differing
      only in pick order, so `idempotent: false`. **One re-run of `--job-id slovakia48 --twice` closes
      AC4**; nothing further to build.

## P3 — Review
- [ ] T3.1 CHANGES_LOG + `/code-review` → REVIEW.md; hand the proven worker to spec-109 + spec-106.

## Out of scope
GPU / device=cuda / latency targets (spec-109) · control plane + front end (spec-106) · parallelism (spec-109).
