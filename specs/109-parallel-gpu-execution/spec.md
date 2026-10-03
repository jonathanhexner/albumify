# spec-109 — Parallel GPU Execution (hit the interactive runtime bar)

**Status:** In Progress — 2026-10-04. **P0 started.** T0.1 baseline timing running (full default_pipeline).
**T0.2 defer-embedding REVISED by a dependency finding:** `select_best` uses `use_face_subclusters=True`
(default) → it consumes `context.face_clusters` for identity-aware per-person selection, which needs
`extract_face_embeddings` on ALL faces. **So embeddings are genuinely on the album-critical path and
cannot be trivially deferred** without changing the default album (identity-aware → scene-only). The
~34% "defer" win in the original draft was based on a wrong assumption. Real levers: **P1 parallelism +
P2 GPU** (the face stack is needed, so speed it up rather than skip it). Deferral is only viable as an
opt-in "scene-only selection" mode — tracked, not default.
**Owner:** Jonathan / Claude
**Priority:** P1 (gates the live product's usability; sizes [[spec-106]] serving)
**Relation:** reuses [[spec-104]] isolation; its output is consumed by [[spec-106]] (the service).

## 1. Problem (measured)

The full pipeline on CPU is far over the interactive bar. Measured on Slovakia26 (2026-10-01,
768px, CPU, the lean `faces` variant):

- **~2.2 s/image**; the **face stack is 75%** (`insightface_detect_faces` 908 ms/img +
  `extract_face_embeddings` 753 ms/img). Scene/people/quality are the other 25%.
- This **excludes** rotation (`score_tilt` GeoCalib ~1.9 s/img) + occlusion CLIP, which the full
  `default_pipeline` adds on top.
- Extrapolated: 100 photos ≈ 3.7 min (faces only); +triage +rotation +occlusion → **10+ min.**

**Target:** a 100-photo album should complete in **≤ 3–5 min** (aim < 1 min). Runtime — not memory
(spec-104 capacity) — is the ceiling.

## 2. Goal / non-goals

**Goal:** execute the pipeline fast enough for interactive use via (a) remote **GPU** execution and
(b) **parallelism** over the embarrassingly-parallel per-image steps, plus (c) deferring work that the
album ranking doesn't need.

**Non-goals:** the serving app / queue / upload path (spec-106 consumes this); auto-scaling policy
and tiering (later — see note); changing curation *results* (must stay equivalent).

## 3. Scope

- **Remote GPU worker** — ONNX CUDA EP + torch CUDA for the InsightFace / DINOv2 / YOLO / AVA stack.
  Estimated ~10–30× → ~0.1–0.2 s/img → 100 photos in ~20–40 s (to be validated on real hardware).
- **Parallelize per-image steps** — score / detect / embed shard across workers (or GPU batches); the
  **cross-image steps** (scene clustering, face clustering, select_best) run once on the gathered
  results. CPU fallback: shard across processes (~1 worker / 4 GB, spec-104 capacity).
- **Defer face embedding** — `extract_face_embeddings` (34% of runtime) produces *identity* for the
  "who's here" view, **not** needed for album pick ranking. Run detection on all; embed only the
  survivors + for the people view. Removes ~a third of the critical path.
- **Rotation cost** — `score_tilt` (GeoCalib) is heavy; make it GPU-only or optional; `straighten_images`
  is already terminal (winners only).
- **Benchmark** — per-step GPU timing, end-to-end for 100/300 photos, and **$/album** (Modal L4
  $0.000222/s → est. ~1–3¢/100-photo album; validate).

## 4. Approach

Add a GPU execution path (device-aware step config) and a parallel executor that fans the per-image
steps across workers/batches, then barriers into the cross-image steps. Land the defer-embedding change
as an ordering/scope change in the pipeline. Validate on a cloud GPU (we only have CPU torch locally).

## 5. Acceptance criteria

- **AC1** 100-photo **full** pipeline (incl. rotation) completes **≤ 3–5 min** (target < 1 min) on the
  chosen GPU.
- **AC2** Measured **$/album** reported (est. ~1–3¢ on L4); idle cost $0 (scale-to-zero via spec-106 host).
- **AC3** Defer-embedding produces an **equivalent album** (same picks within SIGHTING-122 tolerance)
  and still powers the people view.
- **AC4** Parallel per-image steps give near-linear speedup to the worker count; cross-image steps
  unaffected; results equivalent to serial.
- **AC5** CPU parallel fallback measured (for a possible free tier).

## 6. Dependencies, risks & the tiering note

- Reuses spec-104 isolation; feeds spec-106 serving; uses the engine as-is (+ defer-embedding change).
- Risk: GPU numbers are **estimates** until benchmarked on real hardware → AC1/AC2 require a cloud-GPU run.
- Risk: parallelism races on shared context → per-image steps must be pure; cross-image steps barrier.
- **Tiering (noted, deferred):** at portfolio volume GPU is ~1–3¢/album, so v1 can give **everyone** the
  fast GPU path — no free/paid split needed yet. A free-CPU / paid-GPU tier only earns its complexity at
  real scale; the CPU-parallel fallback (AC5) is what a free tier would use.

## 7. Definition of done
GPU path + parallel executor + defer-embedding implemented; AC1–AC5 benchmarked on real GPU; $/album
reported; REVIEW.md; CHANGES_LOG. See `tasks.md`.
