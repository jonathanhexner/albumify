# spec-110 — Containerized curation worker on Modal (CPU)

**Status:** Draft — awaiting approval
**Owner:** Jonathan / Claude
**Priority:** P1 (proves the serving data plane before the GPU optimization)
**Relation:** the worker [[spec-106]]'s control plane calls; [[spec-109]] later makes it fast on GPU.
Splits the old spec-109 "Modal worker" scaffold into *prove-it-runs* (here) vs *make-it-fast* (109).

## 1. Problem

Before chasing GPU speed, we need to know the **serving data-plane path works at all**: can we build a
container image with the pipeline's finicky deps (numpy<2, transformers, insightface, geocalib,
mediapipe, pyiqa, ram), mount our code, pull a job's images from storage, run the real pipeline, and
write a valid `album.json` back — **on Modal**? The dep set fought us locally; the honest unknown is
whether it assembles in a clean container. Proving this **on CPU** isolates "does it run" from "is it
fast" (spec-109), removing the GPU-image + `device=cuda`-per-step unknowns from the critical first step.

## 2. Goal / non-goals

**Goal:** a Modal worker that, on **CPU**, runs the full pipeline end-to-end over a job's images pulled
from storage and writes the normalized result back — correctness proven on a small real trip.

**Non-goals:** GPU, `device=cuda`, latency/throughput targets (CPU is slow — fine; this is a correctness
proof, not a perf test); the control plane / front end (spec-106); parallelism (spec-109).

## 3. Scope

- **CPU image** (`modal.Image`): debian_slim + the pipeline deps resolved to work together on a clean
  container — torch (CPU), onnxruntime (CPU), insightface, ultralytics, transformers, timm, pyiqa
  (numpy<2), ram (`--no-deps`), mediapipe, opencv-headless. Code via `add_local_python_source`.
  **Expect iteration on the first build** — that's the point of doing it early and alone.
- **Storage-reference `curate(job_id)`** (replaces the inline-bytes scaffold): pull the job's 768px
  images from a Volume/object prefix → `execute_spec(..., device=cpu)` → write `album.json` +
  keeper displays back to storage. **Idempotent** (re-running reads the same inputs, overwrites result).
- **Volumes**: model weights + HF cache, uploaded once.
- **`gpu_smoke`** stays as the "Modal account works" proof (runs first).

## 4. Acceptance criteria

- **AC1** `gpu_smoke` returns from Modal on the user's account (path proven).
- **AC2** The CPU image builds on Modal with the full pipeline importable.
- **AC3** `curate(job_id)` on a small real trip (≤40 imgs in storage) returns a **valid `album.json`**
  (picks + scene_clusters + set_aside + theme), equivalent to the local CPU run (within SIGHTING-122 tol).
- **AC4** `curate` is idempotent: re-running the same `job_id` reproduces the result.
- **AC5** Models load from the Volume, not re-downloaded per run (after first populate).

## 5. Dependencies & risks

- Needs the user's Modal auth (account/billing) — deploy/verify cannot be done from the dev box.
- Risk: dep conflicts in the container (we hit them locally) → iterate the image; CPU keeps it simple
  (no CUDA provider, no device plumbing).
- Risk: `add_local_python_source` misses a package → import error surfaces on first `curate`; fix + re-run.

## 6. Definition of done
AC1–AC5 verified on the user's Modal account; the worker + CPU image + storage-ref `curate` committed;
REVIEW.md. Hands a **proven worker** to spec-109 (GPU) and spec-106 (control plane). See `tasks.md`.
