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

import logging
import modal

APP_NAME = "albumify-curator"
GPU_TYPE = "L4"  # cheap + enough for inference; bump to A10G/A100 only if timing demands

app = modal.App(APP_NAME)
# Separate CPU-only app: lets the free-tier data-path proof run while GPU functions are unbillable.
cpu_app = modal.App(APP_NAME + "-cpu")

# Persistent volumes: model weights + download caches (uploaded once, mounted every run) and the
# job store (a directory per job: staged inputs in, album.json out — spec-110 T2.1).
models_vol = modal.Volume.from_name("albumify-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("albumify-hf-cache", create_if_missing=True)
jobs_vol = modal.Volume.from_name("albumify-jobs", create_if_missing=True)
# spec-115 W3: photos stored once by content key (shared by every job, purged after 7 days unused) and
# the pipeline's feature cache, kept across runs so a re-run recomputes only what changed.
inputs_vol = modal.Volume.from_name("albumify-inputs", create_if_missing=True)
cache_vol = modal.Volume.from_name("albumify-cache", create_if_missing=True)

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
    .apt_install("libgl1", "libglib2.0-0", "git", "build-essential")  # opencv/mediapipe libs; hdbscan builds
    # CUDA torch first so nothing below drags a different torch in.
    .pip_install("torch==2.4.0", "torchvision==0.19.0")
    # Mirror the working .venv pins (requirements.txt + pip freeze, 2026-10-09). Pins that matter:
    #   numpy<2 (pyiqa), transformers<5 (5.x refuses torch 2.4 and disables PyTorch -> RAM++ dies),
    #   protobuf 3.20 (mediapipe), one opencv variant only (SIGHTING-111).
    .pip_install(
        "numpy==1.26.4", "scipy==1.16.1", "scikit-learn==1.7.1", "scikit-image==0.26.0", "PyWavelets==1.9.0",
        "pandas==2.3.3", "pydantic==2.12.5", "pandera==0.31.1", "SQLAlchemy==2.0.46",
        "pyyaml", "tqdm", "pillow==11.3.0", "pillow-heif==1.2.1", "jinja2", "matplotlib",
        "opencv-contrib-python==4.11.0.86",
        "protobuf==3.20.3", "mediapipe==0.10.9",
        "insightface==0.7.3",
        "ultralytics==8.4.12", "sixdrepnet==0.1.6",
        "transformers==4.57.1", "timm==1.0.21", "pyiqa==0.1.15.post2", "hf_xet",
        "hdbscan==0.8.40", "umap-learn==0.5.11", "networkx==3.5",
        "clip-anytorch", "kornia==0.8.3", "kornia_rs==0.1.14", "fairscale==0.4.13",
    )
    # GeoCalib + RAM++ from git, --no-deps so they cannot replace the pinned torch/transformers.
    .pip_install("git+https://github.com/cvg/GeoCalib@97b8968e7798a66bf04fcf791fb535624241bda7",
                 extra_options="--no-deps")
    .pip_install("git+https://github.com/xinyu1205/recognize-anything.git@7cb804a8609e9f4b1a50b7f31436d2df40bb9481",
                 extra_options="--no-deps")
    # insightface depends on the CPU `onnxruntime` wheel, which shares the module name with
    # `onnxruntime-gpu` and shadows the CUDA provider. Make the GPU build the only one, last.
    .run_commands("pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.19.2")
    .pip_install("pyarrow==21.0.0", extra_options="--no-deps")
    # SIGHTING-129: protobuf 3.20.3 has no C++ wheel for py3.11 -> pure-Python parser -> InsightFace's
    # onnx.load() of its 144/174 MB models cost ~60 s per construction (x2 per run). protobuf 4.25 (upb)
    # measured 38.7 s -> 1.1 s. mediapipe >= 0.10.14 accepts protobuf 4; numpy re-pinned so it stays < 2.
    .pip_install("mediapipe==0.10.14", "protobuf==4.25.3", "numpy==1.26.4")
    # Pipeline config read by curate() at <repo>/configs/pipeline.yaml; packages land in /root.
    .add_local_dir("configs", remote_path="/root/configs")
    # sim_bench code is added as importable source (no pip install of the repo needed).
    .add_local_python_source("sim_bench", "face_cluster", "geo_cluster")
)

MODELS_DIR = "/root/models"  # == <repo>/models in-container: steps resolve weights relative to the repo root
HF_CACHE = "/cache/hf"          # transformers / RAM++ weights
TORCH_CACHE = "/cache/torch"    # torch.hub (DINOv2) + torchvision/pyiqa checkpoints
INSIGHTFACE_HOME = MODELS_DIR + "/insightface"  # buffalo_l lands in <home>/models/<name>
JOBS_DIR = "/jobs"              # job store root (see sim_bench/serving/job_store.py)
INPUTS_DIR = "/inputs"          # shared input store (see sim_bench/serving/input_store.py)
FCACHE_DIR = "/fcache"          # feature cache Volume: holds sim_bench.db between runs


def _prepare_caches() -> None:
    """Point every model downloader at a mounted Volume (spec-110 T1.3 / AC5).

    Each of these libraries defaults to a path under $HOME, which on a serverless container is
    scratch that dies with the container — so the weights were re-downloaded on every cold start
    while the Volumes sat unused. Must run BEFORE the first import of the libraries that read them.
    """
    import os

    os.environ["HF_HUB_CACHE"] = HF_CACHE          # transformers (RAM++, CLIP)
    os.environ["TORCH_HOME"] = TORCH_CACHE         # torch.hub: DINOv2, GeoCalib, pyiqa checkpoints
    os.environ["INSIGHTFACE_HOME"] = INSIGHTFACE_HOME  # buffalo_l (see face_analyzer._load_app)
    for path in (HF_CACHE, TORCH_CACHE, INSIGHTFACE_HOME):
        os.makedirs(path, exist_ok=True)


@app.function(image=smoke_image, gpu=GPU_TYPE, timeout=120)
def gpu_smoke() -> dict:
    """Prove the Modal GPU path end-to-end: is CUDA visible, which device?"""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
    }


def _deterministic_torch() -> None:
    """Same inputs -> same album.json (spec-110 AC4). Measured 2026-10-09: two identical GPU runs
    differed only in GeoCalib tilt angles (-0.84 vs -0.76 deg) — cuDNN autotuning / TF32 rounding.
    Costs a little speed on conv-heavy steps; correctness of the contract wins."""
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _curate(images: list[tuple[str, bytes]], pipeline: str, device: str) -> dict:
    """Shared body of curate (GPU) / curate_cpu (free-tier proof of the data path).

    Args:
        images: list of (filename, jpeg_bytes) — the uploaded trip (already bounded by the API).
        pipeline: config key from configs/pipeline.yaml (default_pipeline / minimal_pipeline).
    Returns: {selected, scores, scene_clusters, set_aside, album_theme, image_captions, timing}.
    """
    import tempfile
    import torch  # noqa: F401  (import before onnxruntime: ORT-gpu 1.19 resolves CUDA/cuDNN libs from torch's wheels)
    logging.basicConfig(level=logging.INFO)
    _prepare_caches()
    _deterministic_torch()

    from pathlib import Path
    from sim_bench.serving.curate_job import run_pipeline
    from sim_bench.serving.pipeline_runner import SpecBuilder, SpecInputs

    work = Path(tempfile.mkdtemp())
    for name, data in images:
        (work / name).write_bytes(data)

    # Device placement is resolved by SpecBuilder (unit-tested off Modal in tests/serving/).
    plan = SpecBuilder().calc(SpecInputs(pipeline=pipeline, device=device))
    ctx, result = run_pipeline(work, plan)
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


_VOLUMES = {MODELS_DIR: models_vol, "/cache": hf_cache_vol}
# The job functions additionally mount the job store, the input store and the feature cache.
_JOB_VOLUMES = {**_VOLUMES, JOBS_DIR: jobs_vol, INPUTS_DIR: inputs_vol, FCACHE_DIR: cache_vol}
_STAGE_VOLUMES = {JOBS_DIR: jobs_vol}
_INPUT_VOLUMES = {INPUTS_DIR: inputs_vol, JOBS_DIR: jobs_vol}


@app.function(image=full_image, gpu=GPU_TYPE, timeout=1800, cpu=4.0, memory=16384, volumes=_VOLUMES)
def curate(images: list[tuple[str, bytes]], pipeline: str = "default") -> dict:
    """Run the real curation pipeline on GPU over uploaded images -> album result."""
    try:
        return _curate(images, pipeline, device="cuda")
    finally:
        hf_cache_vol.commit()  # persist any model weights downloaded on first run


@cpu_app.function(image=full_image, timeout=3600, cpu=4.0, memory=16384, volumes=_VOLUMES)
def curate_cpu(images: list[tuple[str, bytes]], pipeline: str = "default") -> dict:
    """Same pipeline on a CPU container (free tier): proves image + volumes + data path
    before any GPU spend (spec-110 AC3/AC4). Slow (~9 s/img); use on <=20 images."""
    try:
        return _curate(images, pipeline, device="cpu")
    finally:
        hf_cache_vol.commit()


# ---------------------------------------------------------------------------------------------
# Storage-reference path (spec-110 T2.1): the worker is handed a job_id, not image bytes.
#
#   stage_job(job_id, images)  -> /jobs/<job_id>/input/*       (cheap CPU container, light image)
#   curate_job(job_id)         -> /jobs/<job_id>/album.json    (GPU)  | curate_job_cpu (free tier)
#   fetch_album(job_id)        -> the album dict + run record  (cheap CPU container)
#
# Staging is a separate function so re-running curate_job on the SAME job re-reads the same inputs
# — that is what AC4 (idempotence) means here. The control plane (spec-106) will replace
# stage_job with its own upload path; the curate contract stays the same.
# ---------------------------------------------------------------------------------------------

stage_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "sim_bench", "face_cluster", "geo_cluster")


@cpu_app.function(image=stage_image, timeout=900, volumes=_STAGE_VOLUMES)
def stage_job(job_id: str, images: list[tuple[str, bytes]], replace: bool = True) -> dict:
    """Write a job's analysis-tier images into the job store. No models, no GPU."""
    from sim_bench.serving.job_store import JobStore

    logging.basicConfig(level=logging.INFO)
    jobs_vol.reload()
    store = JobStore(JOBS_DIR)
    written = store.stage_inputs(job_id, images, replace=replace)
    jobs_vol.commit()
    return {"job_id": job_id, "staged": [p.name for p in written],
            "input_dir": str(store.paths(job_id).input_dir)}


def _curate_job(job_id: str, pipeline: str, device: str, keys: list[str] | None = None,
                commit: str | None = None) -> dict:
    """Shared body of curate_job (GPU) / curate_job_cpu — see sim_bench/serving/curate_job.py.

    ``keys`` (spec-115 W3): the job's photos in the shared input store, uploaded once by the control
    plane. The feature cache is copied from its Volume to local disk for the run (SQLite on a network
    Volume is not safe to write in place) and copied back only when the run finishes: a cancelled run's
    cache writes are lost, never half-written (R4)."""
    import dataclasses
    import os
    import shutil
    import time
    from pathlib import Path

    import torch  # noqa: F401  (import before onnxruntime; see _curate)
    logging.basicConfig(level=logging.INFO)
    _prepare_caches()
    _deterministic_torch()

    from sim_bench.serving.curate_job import FEATURE_CACHE_ENV, CurateJobInputs, JobCurator
    from sim_bench.serving.input_retention import mark_used
    from sim_bench.serving.job_store import JobStore
    from sim_bench.serving.run_settings import COMMIT_ENV

    os.environ.pop(FEATURE_CACHE_ENV, None)   # warm containers keep env between calls (review L5)
    os.environ[COMMIT_ENV] = commit or ""     # no git checkout in the container
    jobs_vol.reload()  # see inputs committed by stage_job after this container started
    store = JobStore(JOBS_DIR, inputs_root=INPUTS_DIR)
    local_cache = Path("/tmp/feature_cache.db")
    if keys is not None:
        inputs_vol.reload()
        cache_vol.reload()
        store.write_input_manifest(job_id, keys)
        mark_used(Path(INPUTS_DIR), job_id, keys, time.time())
        shared = Path(FCACHE_DIR) / "sim_bench.db"
        if shared.is_file():
            shutil.copyfile(shared, local_cache)
        os.environ[FEATURE_CACHE_ENV] = str(local_cache)
    runner = "modal_gpu" if device == "cuda" else "modal_cpu"
    result = JobCurator(store).calc(CurateJobInputs(job_id=job_id, pipeline=pipeline, device=device,
                                                    runner=runner))
    if keys is not None and local_cache.is_file():
        tmp = Path(FCACHE_DIR) / "sim_bench.db.part"
        shutil.copyfile(local_cache, tmp)
        os.replace(tmp, Path(FCACHE_DIR) / "sim_bench.db")
    return dataclasses.asdict(result)


def _commit_job_volumes() -> None:
    jobs_vol.commit()      # persist album.json / run.json / results/
    inputs_vol.commit()    # "used" marks
    cache_vol.commit()     # feature cache
    hf_cache_vol.commit()  # persist weights downloaded on first run
    models_vol.commit()    # ... including buffalo_l under INSIGHTFACE_HOME (AC5)


@app.function(image=full_image, gpu=GPU_TYPE, timeout=1800, cpu=4.0, memory=16384,
              volumes=_JOB_VOLUMES)
def curate_job(job_id: str, pipeline: str = "default", keys: list[str] | None = None,
               commit: str | None = None) -> dict:
    """Curate a job on GPU -> album.json + results in the job store (spec-110 AC3, spec-115 W3)."""
    try:
        return _curate_job(job_id, pipeline, device="cuda", keys=keys, commit=commit)
    finally:
        _commit_job_volumes()


@cpu_app.function(image=full_image, timeout=3600, cpu=4.0, memory=16384, volumes=_JOB_VOLUMES)
def curate_job_cpu(job_id: str, pipeline: str = "default", keys: list[str] | None = None,
                   commit: str | None = None) -> dict:
    """Same, on a CPU container: the free-tier correctness proof. Slow (~9 s/img)."""
    try:
        return _curate_job(job_id, pipeline, device="cpu", keys=keys, commit=commit)
    finally:
        _commit_job_volumes()


@cpu_app.function(image=stage_image, timeout=900, volumes=_INPUT_VOLUMES, schedule=modal.Period(days=1))
def purge_inputs() -> dict:
    """Daily (spec-115 R13, owner O3): delete photos no job used in 7 days, and the per-job photo
    copies older jobs left on albumify-jobs. Runs only once the cpu app is deployed."""
    import time
    from pathlib import Path

    from sim_bench.serving.input_retention import purge

    logging.basicConfig(level=logging.INFO)
    inputs_vol.reload()
    jobs_vol.reload()
    res = purge(Path(INPUTS_DIR), time.time(), legacy_jobs_root=Path(JOBS_DIR))
    inputs_vol.commit()
    jobs_vol.commit()
    return {"deleted": res.deleted, "kept": res.kept}


@cpu_app.function(image=stage_image, timeout=300, volumes=_INPUT_VOLUMES)
def delete_inputs(job_id: str, keys: list[str]) -> dict:
    """The owner's "delete my photos from Modal" (R13): drop this job's claim on its photos and delete
    every photo no other job used in the last 7 days."""
    import time
    from pathlib import Path

    from sim_bench.serving.input_retention import release

    inputs_vol.reload()
    res = release(Path(INPUTS_DIR), job_id, keys, time.time())
    inputs_vol.commit()
    return {"deleted": res.deleted, "kept": res.kept}


@cpu_app.function(image=stage_image, timeout=300, volumes=_STAGE_VOLUMES)
def fetch_album(job_id: str) -> dict:
    """Read a finished job's album + run record back out of the store (no models needed)."""
    import json

    from sim_bench.serving.job_store import JobStore

    jobs_vol.reload()
    store = JobStore(JOBS_DIR)
    paths = store.paths(job_id)
    run = json.loads(paths.run_path.read_text(encoding="utf-8")) if paths.run_path.is_file() else None
    return {"job_id": job_id, "album": store.read_album(job_id), "run": run,
            "displays": sorted(p.name for p in paths.display_dir.iterdir())
            if paths.display_dir.is_dir() else []}


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` → runs the GPU smoke and prints the result."""
    print("GPU smoke:", gpu_smoke.remote())
