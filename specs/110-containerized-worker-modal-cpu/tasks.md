# spec-110 — Tasks (Containerized worker on Modal, CPU)

**Status:** Draft — no implementation until spec.md approved. Rationale: `spec.md`.
**Builds on:** the `sim_bench/serving/modal_app.py` scaffold (spec-109). This spec owns the **worker
structure + CPU image + storage-reference curate**; spec-109 owns the GPU variant.
**Needs:** user Modal auth (`modal setup`) — deploy/verify is the user's account.

## P0 — Prove the Modal account/path
- [ ] T0.1 `modal setup` (user) → `modal run …/modal_app.py` → `gpu_smoke` returns. (AC1)

## P1 — CPU image
- [ ] T1.1 Define the CPU `modal.Image`: pipeline deps that work together on a clean container
      (torch-cpu, onnxruntime, insightface, ultralytics, transformers, timm, pyiqa+numpy<2, ram --no-deps,
      mediapipe, opencv-headless) + `add_local_python_source(sim_bench, face_cluster, geo_cluster)`.
- [ ] T1.2 First build on Modal; iterate until the pipeline imports cleanly in-container. (AC2)
- [ ] T1.3 Upload model weights to the Volume (`modal volume put`); confirm reuse across runs. (AC5)

## P2 — Storage-reference curate (CPU)
- [ ] T2.1 Replace inline-bytes with `curate(job_id)`: pull the job's 768px images from storage/Volume →
      `execute_spec(..., device=cpu)` → write `album.json` + keeper displays back. Idempotent.
- [ ] T2.2 Run on a small real trip (≤40 imgs staged in storage); assert a valid `album.json`
      equivalent to the local CPU run (within SIGHTING-122 tolerance). (AC3)
- [ ] T2.3 Re-run same `job_id` → identical result (idempotence). (AC4)

## P3 — Review
- [ ] T3.1 CHANGES_LOG + `/code-review` → REVIEW.md; hand the proven worker to spec-109 + spec-106.

## Out of scope
GPU / device=cuda / latency targets (spec-109) · control plane + front end (spec-106) · parallelism (spec-109).
