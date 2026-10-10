# spec-110 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — AC1, AC2 met; **AC3 met on Modal** (owner's 48-photo GPU job run wrote a valid
normalized `album.json` from storage-reference input). AC4 **failed on that run for the reason since
fixed** (SIGHTING-126, pick order) and needs one re-run to close. AC5 still unproven (no two-run
download comparison yet). New gap found by the same run: SIGHTING-128 — the album omits photos
dropped before scoring (48 staged, 38 in the album), which needs an owner decision on the reason code.

## 2026-10-09 — Owner's 48-photo GPU run on the job path: AC3 met, AC4 reproduced the ordering bug, and a new contract gap

The owner ran the new storage-reference path on Modal (GPU, `--job-id slovakia48 --twice`) while the
ordering fix was still being written, which turned out to be the most useful possible timing: the
remote run independently reproduced SIGHTING-126 and proved everything around it.

- **AC3 met.** 48 images staged to the `albumify-jobs` Volume, `curate_job` read them by id, and a
  valid normalized `album.json` came back: 13 picks, 25 `near_dup` dropped-with-reason, 7 scenes,
  10 people, a full album theme, and **every** score field populated on every pick
  (composite / sharpness / aesthetic / occlusion / tilt) — i.e. `normalize()` reads the right
  `PipelineContext` fields on the real 38-step pipeline, which the local `minimal_pipeline` proof
  could not show. Displays written for all 13 picks; `fetch_album` read the result back out.
- **AC4 not yet met, and the diff is exactly SIGHTING-126.** Between the two runs: pick *set*, scene
  ids, composite scores, dropped set, dropped reasons and theme were **all identical**; only the
  **pick order** differed, so `idempotent` reported `False`. That is the same instability found
  locally, from the same cause, and the fix landed after this run — so closing AC4 needs one re-run,
  not new work. (`album_matches_store: False` is the same artifact: the store held run 2's album while
  the comparison used run 1's.)
- **Warm container is ~1.3x faster:** 227.0 s pipeline on the cold run, 172.2 s on the second
  (254.2 s / 173.6 s driver wall). Per-run model downloading is what AC5 has to rule out.
- **New: SIGHTING-128 — the album is not carrying every photo.** 48 staged, `counts.uploaded = 38`,
  `set_aside = 0`: ten photos dropped by pre-scoring gates appear nowhere in `album.json`, which
  contradicts the show-everything principle the contract states. Filed with a fix sketch; it needs an
  owner decision on the reason code (new `FILTERED` code vs. reusing `NOT_SELECTED`) because it
  changes the spec-106 P0 shape.

| How it was verified | Evidence |
|---|---|
| AC3 on Modal, GPU, real trip | [48-photo job report](../../reports/_modal_job_slovakia48.json) — `ok=True` twice, `device=cuda`, 48 staged -> `album.json` with 13 picks / 25 dropped / 7 scenes / 10 people / theme / all score fields set |
| AC4 blocked only by pick order | same report: pick sets, scene ids, scores, dropped reasons and theme identical across the two runs; order differs -> `idempotent: false`. Fixed after the run by the canonical ordering in `album/normalizer.py` (6 tests) |
| Device really reached the GPU steps | same report, `flipped` = 11 steps incl. `triage_content`, `score_ava`, `insightface_detect_faces`, `extract_scene_embedding`, `extract_face_embeddings`, `caption_*` |
| SIGHTING-128 | same report: `counts.uploaded = 38` vs 48 staged inputs, `set_aside = 0`, 13 + 25 = 38 |

**To close AC3/AC4/AC5 (owner, one command, then repeat it):**
```bash
PYTHONUTF8=1 .venv/Scripts/python scripts/modal_curate_driver.py     --src D:/Slovakia26 --start 300 --n 48 --job-id slovakia48 --twice
```
Expect `AC4 idempotence ...: True` and `album.json in store matches the returned album: True` now that
ordering is canonical; run it a second time and check the log for model downloads (AC5).

## 2026-10-09 — Storage-reference curate built: job store, `album.json`, and an album that is actually reproducible

The worker no longer takes image bytes. A job is a directory and the worker is handed its id.

```
stage_job(job_id, images) -> /jobs/<id>/input/*     (CPU, light image)
curate_job(job_id)        -> /jobs/<id>/album.json  (GPU) | curate_job_cpu (free tier)
fetch_album(job_id)       -> {album, run, displays}
```

- **New modules, all testable off Modal** (this is the point — the worker body used to only be
  observable by paying for a container): `sim_bench/serving/job_store.py` (layout + `job_id`/filename
  validation, since both become paths), `sim_bench/serving/curate_job.py`
  (`JobCurator.calc`: read inputs → **scratch copy** → plan → `normalize()` → write album/displays/run
  record), `sim_bench/serving/pipeline_runner.py` (`SpecBuilder`: `configs/pipeline.yaml` → steps +
  per-step device placement, lifted verbatim out of `modal_app._curate`, which is now thin).
- **The run uses a scratch copy of `input/`** because `straighten_images` writes derivatives beside
  its inputs: without the copy, a second run of the same job would see N + derivatives. That one
  decision is what makes AC4 meaningful rather than accidental.
- **Found and fixed SIGHTING-126 — the album was not reproducible.** Two consecutive runs over the
  same staged images returned the same four picks with the same composite scores, but with the scene
  ids swapped and the picks in a different order: `extract_scene_embedding`'s dict order varies
  between runs, HDBSCAN numbers clusters by row order, and `select_best` iterates that dict. Fixed in
  `album/normalizer.py` (not in the clusterer — cluster ids are inherently arbitrary; the album is
  where a stable contract is owed): scenes renumbered by their earliest photo, picks ordered by EXIF
  capture time (id fallback), `beat`/`beaten_by` sorted. Scene 0 is now the trip's first scene.
- **AC5 cause fixed in code:** `_prepare_caches()` sets `HF_HUB_CACHE`, `TORCH_HOME` (torch.hub:
  DINOv2/GeoCalib/pyiqa) and `INSIGHTFACE_HOME`, the last honored by a 4-line change in
  `InsightFaceFaceAnalyzer._load_app`; the job functions commit `models_vol`. buffalo_l was landing in
  `~/.insightface` — container scratch — so it was re-downloaded every cold start.
- **Blocked locally, not by this work:** the full 38-step pipeline cannot run on the dev box at all —
  `RamTagger(device="cpu")` segfaults (SIGHTING-127, filed). The local proof therefore uses
  `minimal_pipeline`; the default-pipeline comparison must be the Modal CPU run.
- **Not done:** no remote run of the job path (owner's call on spend). AC3/AC4/AC5 need
  `--job-id ... --twice` on Modal, twice, to also show the second run skipping model downloads.

| How it was verified | Evidence |
|---|---|
| Storage-reference path end-to-end on real photos | `scripts/modal_curate_driver.py --src D:/Slovakia26 --start 300 --n 12 --job-id localproof12 --pipeline minimal --local-root reports/_local_jobs --twice` → [local job report](../../reports/_local_job_localproof12.json): 12 staged, 4 picks, 8 dropped with reasons, 2 scenes, displays written, 13.5 s pipeline |
| AC4 idempotence (local) | same command: `AC4 idempotence (created_at excluded): True`; before the SIGHTING-126 fix the same command reported `False` with permuted scene ids |
| Album ordering is order-insensitive | `tests/album/test_normalizer.py::ut_AlbumDeterminism` — 6 tests incl. permuted cluster ids + reversed selected order → identical album; 12 tests in the file |
| Worker orchestration + storage contract | `tests/serving/` — 41 tests: `test_pipeline_runner.py` (9, device placement incl. the SIGHTING-123 regression guard), `test_job_store.py` (19, incl. `job_id`/filename traversal), `test_curate_job.py` (13, incl. inputs-never-mutated, scratch cleanup, failed-step path) |
| InsightFace reads the Volume | `tests/pipeline/test_insightface_model_root.py` — 3 tests (env set → `root` passed; unset → upstream default; CUDA provider still requested) |
| No regression elsewhere | `pytest tests/album tests/serving tests/pipeline tests/triage tests/quality_assessment tests/image_quality_models` → **303 passed**, 47 deselected (6:18). One integration test (`test_extract_face_embeddings.py::test_full_extraction_pipeline`) failed once in an earlier run of the same suite and passed in isolation and on re-run — flaky under model-load pressure, not related to these changes |
| Local full-pipeline block is pre-existing | `python -c "from sim_bench.triage.content import RamTagger; RamTagger(device='cpu')"` → exit `-1073741819`, no pipeline involved (SIGHTING-127) |

## 2026-10-09 — Follow-up audit: correct AC3 and remove stale account blocker
- Inspected `_curate`: accepts `(filename, bytes)` pairs and returns picks/clusters/theme/timing with
  temporary file paths. It does not accept `job_id`, call the album normalizer, or persist keeper displays
  and `album.json`. The earlier "AC3 met" claim meant pipeline success and was too broad for AC3.
- Updated spec/task headers and ROADMAP to the recorded owner decision: shared GPU image for acceptance;
  CPU comparison path retained. Authentication and billing are working; implementation gates remain open.

| How it was verified | Evidence |
|---|---|
| Real-data execution, partial AC3 | [CPU JSON](../../reports/_modal_curate_last.json) + [GPU retry JSON](../../reports/_modal_curate_gpu12_retry2.json): both ok, same 12 input filenames and four pick filenames, identical theme/set_aside |
| Storage/result gap | `sim_bench/serving/modal_app.py::_curate`: inline bytes → temporary directory → pipeline summary; no normalize/persist call or job ID |
| Local regressions | 37 passed, 1 slow test deselected across device routing, portable checkpoints, triage, scorer cache behavior, config/spec validation and normalizer; full command scope recorded in [109 updates](../109-parallel-gpu-execution/updates.html) |
| AC5 remains partial | 3.1 GB uploaded; RAM++ Volume load proven, but buffalo_l downloads outside the persistent HF cache (see [GPU retry log](../../reports/_modal_curate_gpu12_retry2.log)) |

## 2026-10-09 — GPU retry succeeds after remaining device-routing and rebuilt-image fixes
- Full GPU pipeline on the same 12 Slovakia26 images: `ok=True`, same four selected filenames as CPU, restored matching theme/facets; active CUDA providers confirmed for InsightFace detection and embeddings.
- Fixed GeoCalib wrapper device forwarding and RAM++ input placement. First rebuilt-image retry failed due to unpinned PyArrow 26 requiring NumPy 2; pinned PyArrow 21.0.0 to the local venv's working version. 19 regression tests + 9 pipeline/config tests pass.
- Evidence: `reports/_modal_curate_gpu12_retry2.json`, `.log`; Modal run `ap-9Jh59QFjqss55pWj7gORxH`. Pipeline 142.186 s; driver wall 313.8 s. See spec-109 updates for per-step comparison and remaining face-model startup bottleneck.
- This is the inline-byte scaffold proof. Storage-reference input, normalized persisted `album.json`, same-job idempotence, and complete model-cache reuse remain open.

## 2026-10-09 — Modal account live; full pipeline image imports cleanly (AC2 met, CPU, free tier)
- Owner ran `modal setup` (workspace `jonathan-hexner`). Token on this machine.
- First `modal run` built both images but **L4 GPU functions need a payment method** on the account
  (blocks AC1 `gpu_smoke` until the owner adds one). Also: on Windows set `PYTHONUTF8=1`, or the Modal
  CLI dies on a `✓` glyph with a charmap error.
- CPU functions run free, so a scratch `import_check` function on the full image was used instead. First
  build surfaced two real image bugs: `pydantic` (and pandera/SQLAlchemy/hdbscan/umap/sixdrepnet/geocalib…)
  missing, and pip pulled transformers 5.x which refuses torch 2.4 and disables PyTorch (RAM++ would die).
- Fix: `full_image` now mirrors the venv pins (torch 2.4.0+cu121, transformers 4.57.1, numpy 1.26.4,
  protobuf 3.20.3, mediapipe 0.10.9, one opencv variant, GeoCalib + RAM++ from pinned git commits with
  `--no-deps`) and mounts `configs/` which `curate()` reads. **Verified:** every third-party package +
  `sim_bench.pipeline.steps.all_steps`, `sim_bench.album.normalizer`, `face_cluster.pipeline` import inside
  the container (run `ap-hrWjtVpCsHvNLOhXAp9Mno`).
- Model weights (3.1 GB: RAM++, AVA, ArcFace, Siamese, occlusion probes) uploaded to Volume `albumify-models`,
  mounted at `/root/models` (= `<repo>/models`; steps resolve weights relative to the repo root). First
  upload from Git Bash landed under `C:/Program Files/Git/` (MSYS path rewrite) — redone via the Python API.
- `curate` refactored: shared `_curate(images, pipeline, device)` + `curate` (L4) and `curate_cpu` on a
  separate CPU-only app (`app.run()` registers every function, so a GPU function in the same app blocks
  free-tier runs). Driver: `scripts/modal_curate_driver.py` (768px downsample, keeps EXIF).
- Device flip now covers every step whose `config_schema` declares `device` (RAM++ triage and AVA were
  missed by the yaml-only flip) and validates each flip against the typed config first.
- **SIGHTING-123** found by that flip: `score_ava`'s typed config forbade `device` although the step reads
  it — AVA could never run on GPU. Fixed (`ScoreAVAConfig.device`); 26 pipeline config/spec tests green.
- Data-path proof on the free CPU container (12 contiguous Slovakia26 images): spec validation passes,
  discover → triage (RAM++ loads from the Volume) → semantic → geo run; first stop was `score_iqa`
  needing `PyWavelets` (added, pinned 1.9.0). Re-running.
- **AC3 preliminary CPU proof (free tier), 2026-10-09:** after two more container-only fixes (`select_best` could not
  unpickle a `WindowsPath` inside the Siamese checkpoint's training config -> portable
  `checkpoint_io.load_checkpoint`; `PyWavelets` for `score_iqa`), the full 38-step `default_pipeline` ran
  end to end on 12 contiguous Slovakia26 images in the container: `ok=True`, 4 picks, 0 set aside (no
  screenshots in that burst), 201 s pipeline = **16.8 s/img on 4 vCPU** (vs 9.0 s/img on the dev box).
  Top CPU costs in-container: InsightFace detect 5.6 s, face embeddings 3.7 s, RAM++ 2.6 s, GeoCalib 2.4 s.
  Result JSON: `reports/_modal_curate_last.json`. Full AC3 still needs storage-reference input and
  normalized persisted album.json; AC4 (idempotent re-run) still to do.
- **AC1 met:** owner added a payment method; `gpu_smoke` -> `cuda_available: True, NVIDIA L4, torch 2.4.0+cu121`.
- **AC4 on the GPU job flow (48 imgs, same job_id twice), 2026-10-09:** after the SIGHTING-126 ordering fix,
  picks/order/scores are byte-identical; the ONLY remaining diff is the GeoCalib tilt angle in the 2nd
  decimal (-0.84 vs -0.76 deg) = GPU floating-point nondeterminism (cuDNN autotune / TF32). Fix staged:
  `_deterministic_torch()` (cudnn deterministic, no benchmark, TF32 off) in both curate bodies; to be
  confirmed with the next `--twice` run. Note: the driver's "album matches store" line compares run 1 to
  the store, which holds run 2 — run 2 vs store is identical. Results `reports/_modal_job_slovakia48_pb4.json`.
- **AC3 at scale, 2026-10-09:** `curate_job(slovakia_full)` over all 715 photos (97 MB staged inline in one
  `stage_job` call — worked, but is the ceiling for inline staging): 341 s pipeline = 0.48 s/img, wall
  521 s. Valid album.json (147 picks, 423 near_dup, 56 set_aside). Report
  `reports/2026-10-09_slovakia_full_gpu/`. Surfaced: 89 photos absent from album.json (SIGHTING-128).
- **AC4 re-measured with deterministic torch (48 imgs, twice):** picks/order/scores byte-identical; `score.tilt`
  STILL differs (-0.54 vs -1.40 deg) -> not cuDNN/TF32; GeoCalib itself is nondeterministic on CUDA ->
  **SIGHTING-130** with three options (owner's call). Warm run 50 s. `reports/_modal_job_slovakia48_det.json`.
- **Fixed cost solved (SIGHTING-129):** protobuf 4.25 -> warm 48-image run 49 s (was 172 s).
- Decision (owner, 2026-10-09): go GPU directly; spec-110's acceptance criteria are satisfied on the GPU
  image rather than a separate CPU image, to avoid building the dependency set twice.

## Purpose
Prove the serving data path on CPU before touching GPU: image build with the full dependency set,
code mount, storage-reference `curate(job_id)`, valid `album.json` out, idempotent re-run.

## Verification plan (acceptance criteria)
- AC1 `gpu_smoke` returns from a Modal run.
- AC2 pipeline imports cleanly inside the container.
- AC3 a small real trip (≤40 images staged in storage) produces a valid `album.json`.
- AC4 same `job_id` twice → identical result.
- AC5 model weights reused from the Volume across runs.

## Timeline
- 2026-10-05 — spec created (split out of spec-109).

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [CPU JSON](../../reports/_modal_curate_last.json) | successful 12-photo full pipeline on 4 vCPU, 201.504 s; inline-byte summary only |
| [GPU retry JSON](../../reports/_modal_curate_gpu12_retry2.json) · [log](../../reports/_modal_curate_gpu12_retry2.log) | successful L4 retry, matching picks/theme; 142.186 s pipeline / 313.8 s driver wall; model-cache gap visible |
| [Modal explainer](../109-parallel-gpu-execution/modal_explainer.html) · [Q&A](../109-parallel-gpu-execution/modal_q_and_a.html) | worker architecture and storage-reference design |
