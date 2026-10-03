# spec-109 — Tasks (Parallel GPU Execution)

**Status:** Draft — no implementation until spec.md approved. Rationale: `spec.md` (Slovakia26 timing).

## P0 — Baseline & defer-embedding (CPU, measurable now)
- [>] T0.1 Re-run the **full** `default_pipeline` (incl. rotation/occlusion) on the Slovakia subset for true per-step timing (today's numbers are the lean `faces` variant). | 2026-10-04 running
- [~] T0.2 Defer `extract_face_embeddings` — **REVISED (not a free win).** Dependency finding 2026-10-04:
      `select_best` default `use_face_subclusters=True` consumes `context.face_clusters` (identity-aware
      per-person selection) → needs embeddings on ALL faces BEFORE selection. Deferral would change the
      default album (identity-aware → scene-only). So embeddings stay on the critical path; deferral is
      only an **opt-in scene-only selection mode** (future), not the default. The speed lever moves to
      P1 (parallelize the face stack) + P2 (GPU), since the stack is genuinely needed.
- **Gate (revised):** honest full-pipeline per-step baseline captured; defer-embedding reframed + documented.

## P1 — Parallel per-image execution
- [ ] T1.1 Identify pure per-image steps (score/detect/embed) vs cross-image steps (scene/face cluster, select_best).
- [ ] T1.2 Parallel executor: fan per-image steps across workers (≈1/4 GB, spec-104 capacity) / GPU batches; barrier into cross-image steps.
- [ ] T1.3 Verify results equivalent to serial; measure near-linear speedup.
- **Gate:** CPU parallel speedup ≈ worker count; identical album.

## P2 — Remote GPU path
- [ ] T2.1 Device-aware config: ONNX CUDA EP + torch CUDA for InsightFace / DINOv2 / YOLO / AVA.
- [ ] T2.2 Benchmark on a cloud GPU (L4/T4): per-step ms/img, end-to-end 100/300 photos.
- [ ] T2.3 Make `score_tilt` (GeoCalib) GPU-only/optional; confirm rotation included within budget.
- **Gate:** 100-photo full pipeline ≤ 3–5 min (target < 1 min) on GPU.

## P3 — Cost + fallback
- [ ] T3.1 Measure $/album on the chosen GPU (est. ~1–3¢ on L4); confirm idle $0.
- [ ] T3.2 Characterize the CPU-parallel fallback (potential free tier).
- **Gate:** $/album reported; fallback measured.

## P4 — Ship
- [ ] T4.1 CHANGES_LOG + `/code-review` → REVIEW.md; hand the execution path to spec-106.

## Out of scope
The serving app/queue/upload (spec-106) · auto-scaling & tiering policy · any change to curation results.
