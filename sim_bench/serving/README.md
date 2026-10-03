# Albumify Modal GPU worker (spec-109 P2 / spec-106 serving)

Runs the real curation pipeline on a **Modal serverless GPU** (L4, scale-to-zero ≈ $0 idle).
This is the remote-GPU execution target the CPU baseline (9 s/img, non-interactive) proved we need.

> **Division of labor:** the code here is complete, but **you** deploy it — it runs on *your* Modal
> account/billing and needs *your* browser auth. Claude cannot authenticate or deploy on your behalf.

## Prerequisites (you, once)
```bash
.venv/Scripts/python -m modal setup          # browser login to your Modal account
```

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
- Live validation of the full image + the per-image GPU latency number (needs your deploy).
- MiniCPM-V captions (spec-108 P2) on this worker — same `curate` env can host it once the pipeline runs.
