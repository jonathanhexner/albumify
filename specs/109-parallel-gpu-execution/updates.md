# spec-109 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · [modal_explainer.html](modal_explainer.html) · [modal_q_and_a.html](modal_q_and_a.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — re-scoped 2026-10-05 to GPU enablement + parallelism only (spec-110 proves the worker first).

## 2026-10-09 — Follow-up audit: trackers reconciled, saved outputs compared, regressions re-run
- Reconciled ROADMAP and spec-106/109/110 task statuses with the latest run: Modal auth/billing are
  working, the image runs, and the 12-photo GPU proof is complete. No spec is marked Implemented.
- Saved CPU versus GPU retry outputs have identical ordered input filenames, selected filename sets,
  album_theme, and set_aside. Pick order and temporary paths differ; exact JSON idempotence is unproven.
- Next performance work: profile face-model initialization versus inference, persist non-HF weights,
  reuse/preload sessions, then measure 100/300-photo latency and cost. Storage/job durability belongs to 110.

| How it was verified | Evidence |
|---|---|
| Saved CPU/GPU output comparison | [CPU JSON](../../reports/_modal_curate_last.json), [GPU retry JSON](../../reports/_modal_curate_gpu12_retry2.json): 12 identical inputs, same four picks/theme/set_aside; 201.504 → 142.186 s pipeline; GPU driver wall 313.8 s |
| CUDA provider and bottleneck evidence | [GPU retry log](../../reports/_modal_curate_gpu12_retry2.log): active CUDA providers; detection + embeddings 101.850 s / 142.186 s = 71.6% |
| Current local regression run | **37 passed, 1 slow test deselected**: checkpoint_io, tilt_device, ram_tagger_device, content_triage, triage_steps, score_tilt, score_occlusion, albumify_default_spec, pipeline_spec, typed_step_configs, album normalizer; `-m 'not slow and not e2e and not budapest'` explicitly includes offline tests under the needs_data-marked tilt module |
| Limits | Read-only comparison of existing remote artifacts plus local tests; no new Modal benchmark/deployment. Storage-reference result, same-job idempotence, complete cache reuse, full-size latency, and measured cost remain pending |

## 2026-10-09 — GPU retry: two remaining routing gaps fixed; rebuilt-image dependency failure
- Fixed `estimate_tilt` dropping `device` before `estimate_tilt_raw`, and RAM++ input tensors staying on CPU after its model moved to CUDA (SIGHTING-124).
- Enabled Modal output and worker INFO logging. First retry stopped at `triage_content`: newly resolved PyArrow requires NumPy 2, incompatible with the pinned NumPy 1.26.4 (SIGHTING-125). Pinned `pyarrow==21.0.0` to the working local version; second retry succeeded.
- **Local verification:** 19 focused tests (GeoCalib/RAM++ device routing, triage, scorer cache contracts, portable checkpoints) and 9 pipeline/config tests pass; `git diff --check` passes.
- First retry evidence: `reports/_modal_curate_gpu12_retry.json`, `reports/_modal_curate_gpu12_retry.log`; Modal run `ap-4EGJfnlJ7n0Sc0fPVaAaqc`.
- **Real GPU verification:** `reports/_modal_curate_gpu12_retry2.json` and `.log`; Modal run `ap-9Jh59QFjqss55pWj7gORxH`. `ok=True`; exactly the same 12 inputs and four selected filenames as the CPU reference. Restored theme matches CPU: outdoors with family/friends, facets food=1/nature=3/people=8. Both InsightFace detection and embedding logs confirm active CUDA providers; no pipeline model-failure warnings.
- Pipeline **142.186 s = 11.85 s/img**, versus 202.862 s previous GPU / 201.504 s CPU (1.43x / 1.42x faster). Driver wall **313.8 s**, including startup and image preparation; pipeline improvement is not a wall-time improvement.
- Per-step previous GPU → retry: RAM++ 11.439 → 12.497 s (previous run returned empty tags), occlusion 18.103 → 6.944 s, GeoCalib 55.518 → 5.994 s, DINOv2 11.226 → 6.019 s, InsightFace detection 53.544 → 56.936 s, embeddings 43.391 → 44.914 s.
- **Remaining bottleneck:** face detection + embedding account for 101.850 s (72% of pipeline time), despite CUDA being active. Logs show two complete buffalo_l model initializations and downloads outside the persistent HF cache. Profile initialization versus inference, reuse/preload model sessions, and persist non-HF caches before claiming the 100-photo latency gate. Same-job idempotence and persistent deployment remain unverified.

## 2026-10-09 — First GPU run: the pipeline barely used the GPU (P1 finding, fixes staged, NOT yet re-run)
- `gpu_smoke` passed (NVIDIA L4, torch 2.4.0+cu121). `curate` on 12 Slovakia26 images on the L4:
  **16.9 s/img — identical to the 4-vCPU CPU container (16.8)**. Picks identical by filename. Result:
  `reports/_modal_curate_gpu12.json`. Per step vs CPU container: RAM++ 2.7x faster, YOLO 2.7x; InsightFace
  detect 1.3x, embeddings 1.0x; GeoCalib tilt 0.5x, occlusion CLIP 0.6x, DINOv2 0.7x (slower).
- Root causes found by reading each step's device path:
  1. the GPU function had no `cpu=` request (default fractional core) -> every CPU-bound step starved;
  2. `InsightFaceFaceAnalyzer` hard-coded `providers=['CPUExecutionProvider']` (device only changed ctx_id);
  3. the image installed both `onnxruntime` (via insightface) and `onnxruntime-gpu`; the CPU wheel shadows
     the CUDA provider;
  4. `extract_scene_embedding` dropped `device` when building the DINOv2 extractor;
  5. `OcclusionScorer` hard-coded `clip.load(device="cpu")`; 6. GeoCalib `_get_model` hard-coded `.to("cpu")`.
- **Staged fixes (code edited, imports verified, NOT yet run on Modal):** `curate` gets `cpu=4.0`; CUDA+CPU
  providers in `face_analyzer.py` (logs the active provider); image layer uninstalls both ORT wheels and
  installs `onnxruntime-gpu==1.19.2` last, `torch` imported before ORT; DINOv2 device passthrough;
  `OcclusionScorer(device=)` + `score_occlusion` device knob; `TiltScorer(device=)` + `score_tilt` device
  knob. `score_iqa` is rule-based (no model) — nothing to move.
- **Next:** re-run `scripts/modal_curate_driver.py --start 300 --n 12` (GPU), confirm the InsightFace
  provider log says CUDA, compare per-step times; then AC4 idempotence; then storage-reference input.

## 2026-10-09 — GPU run v2 (after the six fixes): GPU engaged, but 12 images is too few to read per-image cost
- Same 12 images, same picks. InsightFace log confirms `active=['CUDAExecutionProvider', ...]`.
  GeoCalib tilt 28.7 s -> 7.1 s (x4), DINOv2 x1.2, RAM++ x2.1, YOLO x2.3. Face stack got SLOWER
  (detect 67.6 -> 75.1 s, embeddings 44.6 -> 62.2 s). Total 15.5 s/img (CPU container 16.8).
  Result: `reports/_modal_curate_gpu12_v2.json`.
- Reading: every step's time includes its model's cold load from the network-backed Volume plus
  first-inference setup, so at N=12 fixed costs dominate and the per-image GPU cost is not yet visible.
  For InsightFace specifically, ORT's default EXHAUSTIVE cuDNN algorithm search runs on first inference
  per shape for each of buffalo_l's 5 nets, twice (detect step + embedding step reload).
- Staged: `cudnn_conv_algo_search=HEURISTIC` in both InsightFace loaders; next run uses N=48 to separate
  fixed vs per-image cost (if detect time stays ~75 s at 4x images, it is warm-up; if it 4x's, it is per-image).

## 2026-10-09 — GPU run v3 (48 images): the cost model is ~175 s fixed + 0.88 s/img
- 48 contiguous Slovakia26 images on the L4 (HEURISTIC cuDNN search): ok, 13 picks, 8 scene clusters,
  0 set aside, wall 255 s. Result `reports/_modal_curate_gpu48.json`.
- Comparing 12 vs 48 images per step gives fixed (warm-up) vs marginal (per image) cost:

  | step | fixed s | per-image s |
  |---|---|---|
  | insightface_detect_faces | 72 | 0.28 |
  | extract_face_embeddings | 62 | ~0 |
  | triage_content (RAM++) | 13 | 0.11 |
  | score_tilt (GeoCalib) | 4 | 0.24 |
  | score_occlusion | 9 | 0.06 |
  | extract_scene_embedding | 6 | 0.05 |
  | **total** | **175** | **0.88** |

- Reading: per image the GPU is ~19x the dev box (0.88 vs 16.8 s). The problem is the fixed 175 s, of
  which 134 s is InsightFace warm-up — and buffalo_l is loaded TWICE (detect step, then the embedding
  step builds its own FaceAnalysis). 100 photos today ≈ 175 + 88 = 4.4 min; 715 ≈ 13.6 min.
- Implication for the live product: the fixed cost must move out of the request path — load once per
  container (`@modal.enter`), keep the container warm during upload (spec-106 warm-up ∥ upload), and share
  one InsightFace instance between the two steps. Probing where the 134 s go next.

## 2026-10-09 — Probe: the 134 s InsightFace warm-up is a model DOWNLOAD, twice per run
- Isolated timing on the L4 (`ort_warmup_probe`): `buffalo_l_cached: False`;
  `FaceAnalysis("buffalo_l")` construct = **71 s (CPU provider) / 61 s (CUDA)**; `prepare` 0-1 s; first
  inference 0.6 s; next 5 inferences **0.07 s total (14 ms/img)**. torch CUDA init 4.7 s.
- So `~/.insightface` is container-local and buffalo_l is re-downloaded from GitHub on every run, once
  by the detect step and once by the embedding step. GPU inference itself is ~20x faster than CPU.
- Fix staged: `_persist_home_caches()` symlinks `~/.insightface` and `~/.cache` (torch hub, clip,
  pyiqa, ultralytics) onto the `/cache` Volume before any model import; `TORCH_HOME` set. First run
  populates the cache, every later container reuses it. Re-running 48 images twice (populate, then warm).

## 2026-10-09 — Job flow on GPU (48 imgs, twice) + the real root cause of the warm-up: pure-Python protobuf
- Storage-reference flow (`stage_job` -> `curate_job` -> `fetch_album`, built in a parallel session) ran
  end to end on the L4: album.json with picks + drop reasons + display copies. Run 1 227 s, run 2 (warm)
  172 s; detect 89 -> 64 s, embed 70 -> 61 s even with buffalo_l on the Volume. Result
  `reports/_modal_job_slovakia48.json`.
- AC4: same 13 picks, identical scores, **different order** -> SIGHTING-126 (fixed in the normalizer by the parallel session: chronological picks + canonical scene ids).
- ORT session creation measured at < 1 s per model (all providers/thread settings). The 60 s is
  `onnx.load` inside InsightFace's constructors under **pure-Python protobuf** (3.20.3 has no C++ wheel
  for py3.11): 38 s + MemoryError on the 144 MB landmark model locally -> SIGHTING-129. Affects the dev
  box identically. Fix candidate (protobuf 4.25 + mediapipe 0.10.14) being measured in a container variant.

## 2026-10-09 — protobuf 4 applied: warm 48-image GPU run 172 s -> 49 s (1.0 s/img all-in)
- Probe (free CPU container, standalone images): protobuf 3.20.3 = `python` impl, FaceAnalysis construct
  38.7 s; protobuf 4.25.3 = `upb`, **1.1 s**; mediapipe 0.10.14 + onnx 1.22 import fine.
- Applied to the worker image (+ venv, requirements.txt, CLAUDE.md rule; owner approved). Job flow,
  48 images, twice: run 1 73 s (wall 92), **run 2 49 s (wall 51)**. detect 88.7 -> 4.7 s, embeddings
  69.6 -> 1.8 s. Same 13 picks. Result `reports/_modal_job_slovakia48_pb4.json`.
- Remaining fixed cost (~25 s): RAM++ 10 s (2.8 GB checkpoint from the Volume), GeoCalib 12 s,
  occlusion CLIP 6 s — all model loads; `@modal.enter` warm containers (P2) hide them.
- Projection: 100 photos ≈ 1.5-2 min cold, under 1.5 min warm — inside the 3-5 min bar without any
  parallelism. Full 715-photo Slovakia roll ≈ 11-12 min.
- AC4 idempotence still reports False on the job flow — being diffed (SIGHTING-126 fix is in the tree).

## How it was verified so far
| Item | Evidence |
|---|---|
| T0.1 CPU baseline of the REAL default pipeline (38 steps) | `reports/2026-10-04_spec109_baseline/`: **9.0 s/img** on 57 Slovakia images; RAM++ triage 3.0 s + GeoCalib tilt 2.6 s = 62%, face stack 2.0 s = 23% |
| T0.2 "defer face embeddings" idea | Rejected with evidence: `select_best` defaults to face sub-clusters, which need embeddings on all faces |
| P2 Modal worker (`sim_bench/serving/modal_app.py`) | L4 smoke, image imports, and full CPU/GPU inline-byte runs passed 2026-10-09. GPU retry: 142.186 s / 12 images, matching CPU picks/theme. Persistent deployment and 100/300-photo gate pending |

## Decisions
- Host = Modal (owner approved 2026-10-04). L4 GPU, ~1–2 cents per album, $0 idle.
- Whole pipeline runs on one GPU box; CPU-only steps are ~2% of time, not worth splitting.
- Storage-reference input and the warm-up trigger moved to spec-110 / spec-106.

## Open
- Account/auth blocker cleared; storage-reference result, idempotence, complete model reuse and persistent deployment remain in spec-110/106.
- Complete per-step runtime device evidence (MediaPipe may be CPU-only), face-model profiling/preload, 100/300-photo latency, measured cost, and parallelism/fallback gates.

## Timeline
- 2026-10-05 — re-scoped to GPU-only; CPU proof split into spec-110.
- 2026-10-04 — baseline report; Modal approved; worker scaffolded; two explainer HTMLs.
- 2026-10-01 — PRD created from the Slovakia26 findings.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [GPU retry JSON](../../reports/_modal_curate_gpu12_retry2.json) · [log](../../reports/_modal_curate_gpu12_retry2.log) | 12-photo successful L4 run; matching CPU pick filenames/theme; 142.186 s pipeline, 313.8 s driver wall; CUDA providers active |
| [Modal CPU JSON](../../reports/_modal_curate_last.json) | same 12 inputs; four picks; 201.504 s pipeline on 4 vCPU; comparison reference, not persisted album.json |
| [CPU baseline summary](../../reports/2026-10-04_spec109_baseline/summary.html) | full 38-step pipeline, 57 Slovakia imgs: 9.0 s/img; RAM++ 3.0 s + GeoCalib 2.6 s = 62%, face stack 2.0 s = 23%. Raw per-step numbers in `timing.json` (timing only, no image report) |
| [Slovakia26 analysis](../../reports/2026-10-01_slovakia_analysis/report.html) | the earlier lean `faces` number (2.2 s/img, face stack 75%) this baseline corrected |
| [Modal explainer](modal_explainer.html) | what Modal is, our architecture and implementation, deploy lifecycle, cost table |
| [Modal Q&A](modal_q_and_a.html) | whole pipeline on one GPU box; storage-reference input; reactive warm-up |
| Worker scaffold | `sim_bench/serving/modal_app.py` + `README.md` (commands: setup → smoke → volume put → curate → deploy) |
