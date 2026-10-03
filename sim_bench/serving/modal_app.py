"""Albumify curation on a Modal serverless GPU (spec-109 P2 / spec-106 serving worker).

This is the remote-GPU execution target: a Modal app that runs the real pipeline on an L4 GPU,
scale-to-zero (≈$0 idle). It is deployed to the USER's Modal account — Claude cannot auth or deploy it.

Two functions, smoke-first (prove the path before the heavy model upload):
  * `gpu_smoke`   — trivial: confirm CUDA is visible on the Modal GPU. Deploy + run this FIRST.
  * `curate`      — the real run: execute the pipeline on GPU over uploaded images → album result.

Models (~GBs: RAM++, InsightFace, DINOv2, YOLO, GeoCalib, CLIP, AVA) and the HF cache live on a Modal
Volume, uploaded ONCE — not baked into the image, not re-downloaded per cold start.

Quickstart (YOU run these — your account/billing):
    .venv/Scripts/python -m modal setup                       # browser auth, one time
    .venv/Scripts/python -m modal run sim_bench/serving/modal_app.py::gpu_smoke   # prove GPU
    # then upload models to the volume + deploy (see sim_bench/serving/README.md)
"""
from __future__ import annotations

import modal

APP_NAME = "albumify-curator"
GPU_TYPE = "L4"  # cheap + enough for inference; bump to A10G/A100 only if timing demands

app = modal.App(APP_NAME)

# Persistent volumes: model weights + HF cache (uploaded once, mounted every run).
models_vol = modal.Volume.from_name("albumify-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("albumify-hf-cache", create_if_missing=True)

# --- light image: just enough to prove the GPU path (smoke) ---
smoke_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",  # CUDA build resolved by Modal's GPU base
)

# --- full image: the pipeline's dependency set (CUDA). Mirrors the working .venv pins.
#     NOTE: this dep set is finicky (numpy<2 for pyiqa, transformers for RAM++, geocalib --no-deps,
#     onnxruntime-gpu for InsightFace). Expect to iterate on the FIRST deploy; that's why smoke
#     is separate and deployed first. ---
full_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")  # opencv / mediapipe runtime libs
    .pip_install(
        "torch==2.4.0", "torchvision",
        "onnxruntime-gpu", "insightface", "ultralytics",
        "transformers", "timm", "pillow", "pyyaml", "numpy<2",
        "opencv-python-headless", "mediapipe", "pyiqa", "scikit-learn", "hf_xet",
    )
    .pip_install("fairscale")  # RAM++ inference dep
    .pip_install("git+https://github.com/xinyu1205/recognize-anything.git",
                 extra_options="--no-deps")  # RAM++ (don't let it pull a conflicting torch/transformers)
    # sim_bench code is added as importable source (no pip install of the repo needed).
    .add_local_python_source("sim_bench", "face_cluster", "geo_cluster")
)

MODELS_DIR = "/models"
HF_CACHE = "/cache/hf"


@app.function(image=smoke_image, gpu=GPU_TYPE, timeout=120)
def gpu_smoke() -> dict:
    """Prove the Modal GPU path end-to-end: is CUDA visible, which device?"""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
    }


@app.function(image=full_image, gpu=GPU_TYPE, timeout=1800,
              volumes={MODELS_DIR: models_vol, "/cache": hf_cache_vol})
def curate(images: list[tuple[str, bytes]], pipeline: str = "default") -> dict:
    """Run the real curation pipeline on GPU over uploaded images → album result.

    Args:
        images: list of (filename, jpeg_bytes) — the uploaded trip (already bounded by the API).
        pipeline: config key from configs/pipeline.yaml (default_pipeline / minimal_pipeline).
    Returns: {selected, scores, scene_clusters, set_aside, album_theme, image_captions, timing}.
    """
    import os, tempfile, yaml
    os.environ["HF_HUB_CACHE"] = HF_CACHE
    # point the model loaders at the Volume (steps read these paths / env where applicable)
    os.environ.setdefault("ALBUMIFY_MODELS_DIR", MODELS_DIR)

    from pathlib import Path
    from sim_bench.pipeline.run import execute_spec
    from sim_bench.pipeline.spec import PipelineSpec
    from sim_bench.pipeline.context import PipelineContext

    work = Path(tempfile.mkdtemp())
    for name, data in images:
        (work / name).write_bytes(data)

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    key = "default_pipeline" if pipeline == "default" else f"{pipeline}_pipeline"
    steps = doc[key]
    step_configs = {s: (doc.get(s) or {}) for s in steps}
    # GPU: flip every step that takes a device knob to cuda.
    for s, c in step_configs.items():
        if isinstance(c, dict) and "device" in c:
            c["device"] = "cuda"

    ctx = PipelineContext(source_directory=work)
    result = execute_spec(PipelineSpec(steps=steps, step_configs=step_configs), ctx, fail_fast=True)
    if not result.success:
        return {"ok": False, "failed_step": result.failed_step, "error": result.error_message}

    return {
        "ok": True,
        "selected": [str(p) for p in (ctx.selected_images or [])],
        "scene_clusters": {int(k): v for k, v in (ctx.scene_clusters or {}).items()},
        "set_aside": ctx.set_aside or [],
        "album_theme": ctx.album_theme or {},
        "image_captions": ctx.image_captions or {},
        "timing": {"total_ms": result.total_duration_ms,
                   "steps": [{"step": r.step_name, "ms": r.duration_ms} for r in result.step_results]},
    }


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` → runs the GPU smoke and prints the result."""
    print("GPU smoke:", gpu_smoke.remote())
