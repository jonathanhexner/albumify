# Albumify Modal GPU worker (spec-109 P2 / spec-106 serving)

Runs the real curation pipeline on a **Modal serverless GPU** (L4, scale-to-zero ≈ $0 idle).
This is the remote-GPU execution target the CPU baseline (9 s/img, non-interactive) proved we need.

> **Division of labor:** the code here is complete, but **you** deploy it — it runs on *your* Modal
> account/billing and needs *your* browser auth. Claude cannot authenticate or deploy on your behalf.

## Prerequisites (you, once)
```bash
.venv/Scripts/python -m modal setup          # browser login to your Modal account
```
Then add a **payment method** at modal.com (Settings -> Billing): GPU functions refuse to start without
one, even inside the free credit allowance. CPU functions run without it.

**Windows:** run every `modal` command with `PYTHONUTF8=1` set (PowerShell: `$env:PYTHONUTF8="1"`),
or the CLI crashes on a check-mark glyph (`charmap codec can't encode '✓'`). Upload volumes from
PowerShell or the Python API, not Git Bash: MSYS rewrites the remote path `/` into `C:/Program Files/Git/`.

## Step 1 — prove the GPU path (cheap, ~30 s, do this FIRST)
```bash
.venv/Scripts/python -m modal run sim_bench/serving/modal_app.py
# expect: GPU smoke: {'cuda_available': True, 'device': 'NVIDIA L4', 'torch': '2.4.0+...'}
```
If that prints `cuda_available: True`, Modal + your account + GPU all work. Only then invest in the
heavy image + model upload.

## Step 2 — upload the models to the Volume (once, ~GBs)
The pipeline's weights (InsightFace buffalo_l, DINOv2, YOLO, AVA, occlusion CLIP, GeoCalib, RAM++,
Siamese) + the HF cache live on Modal Volumes, not in the image. Upload them once:
```bash
# model artifacts under ./models/ -> the albumify-models volume
.venv/Scripts/python -m modal volume put albumify-models ./models /models
# (RAM++ / HF-downloaded models populate albumify-hf-cache on first curate run)
```
See `MODEL_MANIFEST` below for exactly which files the steps expect.

## Step 2b — free-tier proof of the data path (no GPU needed)
`curate_cpu` is the same pipeline on a CPU container (4 vCPU, 16 GB): proves image + volumes + config +
result shape before spending on GPU (spec-110 AC3/AC4). ~9 s/img, so keep it to <=20 images.
```bash
.venv/Scripts/python -m modal run sim_bench/serving/modal_app.py::curate_cpu   # via a small driver, see scripts/
```

## Step 2c — the storage-reference job path (spec-110 T2.1, what the control plane will call)

`curate` takes image bytes; **`curate_job` takes a `job_id`**. A job is a directory on the
`albumify-jobs` Volume:

```
/jobs/<job_id>/input/*.jpg    staged 768 px images  (written by stage_job; never written by a run)
/jobs/<job_id>/album.json     the normalized Album   (spec-106 contract)
/jobs/<job_id>/display/       display copies of the picks
/jobs/<job_id>/run.json       device, timing, per-step device placement
```

Prove it on a real trip, CPU first, with the idempotence check (`--twice` curates the same job
twice and diffs the albums):

```bash
# CPU (free tier)
PYTHONUTF8=1 .venv/Scripts/python scripts/modal_curate_driver.py     --src D:/Slovakia26 --start 300 --n 12 --job-id slovakia12 --cpu --twice
# then GPU (same job, same inputs)
PYTHONUTF8=1 .venv/Scripts/python scripts/modal_curate_driver.py     --src D:/Slovakia26 --start 300 --n 12 --job-id slovakia12 --twice
```

Expect `ok=True`, `AC4 idempotence ...: True`, and an `album.json` with picks + dropped-with-reason.
Run it **a second time** to check AC5: the models must not download again (`_prepare_caches()` points
HuggingFace, `torch.hub` and InsightFace at the mounted Volumes, and the job functions commit them).

**No Modal account needed to test the same path** — `--local-root` runs `JobStore` + `JobCurator`
locally, which is how the contract is regression-tested without spend:

```bash
PYTHONUTF8=1 .venv/Scripts/python scripts/modal_curate_driver.py     --src D:/Slovakia26 --start 300 --n 12 --job-id localproof12     --pipeline minimal --local-root reports/_local_jobs --twice
```

`minimal` here because `triage_content` cannot load RAM++ on the local venv at all (SIGHTING-127);
the full 38-step comparison has to run on Modal.

## Step 3 — first `curate` run (iterate on the image here)
The full image's dep set is finicky (numpy<2 for pyiqa, transformers for RAM++, onnxruntime-gpu for
InsightFace, geocalib). Expect the FIRST build to surface a missing/clashing dep — fix in
`full_image` in `modal_app.py`, re-run. Validate on a small real trip:
```bash
.venv/Scripts/python -c "import modal; \
  f = modal.Function.from_name('albumify-curator','curate'); \
  imgs = [(p.name, p.read_bytes()) for p in __import__('pathlib').Path('D:/Slovakia26').glob('*.jpg')][:40]; \
  print(f.remote(imgs))"
```
Check: `ok: True`, sensible `selected`, and `timing.total_ms / n` ≈ the per-image GPU latency (the
number that must land a 100-photo album under the 3–5 min bar).

## Step 4 — deploy (persistent endpoint)
```bash
.venv/Scripts/python -m modal deploy sim_bench/serving/modal_app.py
```
`curate` is then callable from the spec-106 control-plane API (`modal.Function.from_name(...)`).

## MODEL_MANIFEST — what `curate` needs on the volume
| path on volume | used by | source |
|---|---|---|
| `/models/album_app/ava_resnet50.pt` | score_ava | repo `models/` |
| `/models/occlusion/clip_b32_gmax_v1.npz` | score_occlusion | repo `models/` |
| `/models/album_app/siamese_comparison_model.pt` | select_best (siamese) | repo `models/` |
| `/models/ram/ram_plus_swin_large_14m.pth` | triage_content (RAM++) | HF (2.8 GB) |
| InsightFace `buffalo_l`, DINOv2, YOLO, GeoCalib | detect/embed/scene/tilt | HF cache (first run) |

## Cost (L4 @ ~$0.80/hr, per-second, scale-to-zero)
~GPU-seconds per album × $0.000222/s. If GPU lands ~0.3–0.5 s/img, a 100-photo album ≈ 30–50 GPU-s
≈ **1–2¢/album**, $0 at rest. Set a Modal budget alarm.

## What is NOT done here
- The spec-106 control-plane API that calls `curate` (bounded upload → enqueue → this worker).
- **AC4/AC5 on the job path.** The 2026-10-09 48-photo GPU run proved AC3 (storage-reference input to
  a valid `album.json`: 13 picks, 25 dropped-with-reason, 7 scenes, full theme, all score fields set;
  `reports/_modal_job_slovakia48.json`). Its two runs differed *only* in pick order, which the
  canonical ordering in `album/normalizer.py` fixed afterwards — so AC4 needs one re-run, and AC5 a
  further run showing no model downloads.
- **SIGHTING-128:** that run also showed `album.json` omitting photos dropped before scoring
  (48 staged, `counts.uploaded = 38`). Needs a decision on the reason code before the front end can
  rely on "every photo is in the album".
- Production deployment and the 100/300-photo GPU latency gate. The 2026-10-09
  12-photo retry passed with the same picks and theme as CPU: 142.186 s pipeline, 313.8 s driver wall
  time; CUDA providers confirmed. Evidence: `reports/_modal_curate_gpu12_retry2.json` and `.log`.
  Face-model initialization/inference remains the performance bottleneck; non-HF downloads still
  need persistent caching. See `specs/109-parallel-gpu-execution/updates.md`.
- MiniCPM-V captions (spec-108 P2) on this worker — same `curate` env can host it once the pipeline runs.
