# spec-104 — Staged process isolation for the pipeline (SIGHTING-117 durable fix)

**Status**: Draft — awaiting approval. Phase 0 (cheap release fix) already landed 2026-07-30.
**Owner**: Claude / user
**Supersedes**: the "stream/batch InsightFace detections" framing of SIGHTING-117 (measured wrong).
**Related**: SIGHTING-117, spec-040 (unified pipeline framework), `feedback_pipeline_is_config`.

---

## 1. Problem (measured, not assumed)

The full album pipeline OOM-kills on real-size trips. Three memory probes (2026-07-30, on the
768px working sets) established the real cause, which **corrects** the sighting's recorded text:

- It is **NOT** detections (tiny) and **NOT** aligned crops (Budapest 67 MB, Austria 149 MB).
- It **IS stacked model memory that `release()` never returns to the OS.** Full `faces` pipeline:
  peak **1735 MB** on 122 imgs (Budapest), **2090 MB** on 474 (Austria). Peak is
  ~model-count-bound; image count adds only a few hundred MB. The `default` pipeline (adds
  CLIP-occlusion + GeoCalib-tilt + Siamese) projects to **3 GB+** → OOM on a low-RAM box.

| step | leaked & kept | backend | release()? | RSS freed |
|---|---|---|---|---|
| extract_scene_embedding (DINOv2) | +425 MB | torch | ✅ nulls handle | **0** |
| extract_face_embeddings | +363 MB | ONNX | ❌ was missing → **fixed Phase 0** | 0→~360 recovered |
| score_ava (resnet50) | +187 MB | torch | ✅ | **0** |
| detect_persons (YOLO) | +164 MB | torch | ✅ | **0** |
| insightface_detect_faces | +33 MB | ONNX | ✅ | **freed 471** ✅ |

**Root fact:** InsightFace ONNX sessions drop on `gc`, but **torch CPU models do not return
memory to the OS on Windows** — no `torch.cuda.empty_cache` equivalent for CPU, no `malloc_trim`
on Windows. `del + gc.collect()` frees the Python object; RSS stays as a high-water mark and
fragments (models don't reuse each other's freed pages).

**Proven remedy:** process isolation. Prototype (`isolation_proto.py`): DINOv2 work done in a
child process that exits → **parent RSS delta +0 MB** (results handed back via disk); same work
in-process → **+1105 MB** retained. Process death is the only reliable `free()` for torch here.

## 2. Goal

Make the executor own a **two-tier resource-management contract** so peak RSS is bounded by
**one stage's models**, not the sum of the whole pipeline — and so a step's OOM/crash no longer
kills the whole run. Directly answers "should the pipeline control release/isolation, incl. on error?": **yes.**

### The contract (executor-owned, both tiers under one error umbrella)

- **Tier 1 — `release()` (already exists).** Default no-op; model-owning steps null their handle.
  Executor calls it in `finally` after every step (success OR failure). Lightweight, same-process.
  Sufficient for ONNX; **insufficient for torch** (measured).
- **Tier 2 — isolation boundary (new).** A declared **contiguous stage** of steps runs in a child
  process that **exits** when the stage finishes. The OS reclaims 100% of the stage's model
  memory. Also under unified error handling: a stage exception → clean failed `StepResult`; a
  stage **crash/OOM** (non-zero child exit) → the **parent survives** and reports the failure
  (today an OOM kills everything with no traceback).

Both tiers are declarative and controlled by the executor — step authors don't manage processes.

## 3. Design

### 3.1 Stage granularity (why not per-step)

Large intermediates must stay **inside** one child. `align_faces` produces 149 MB of crops that
`extract_face_embeddings` consumes; if those were separate isolated steps we'd pickle 149 MB
across the boundary. So the isolation unit is a **contiguous stage** grouping the steps that share
big intermediates. Only small, declared **exports** cross back (embeddings/scores/records/clusters
— all a few MB). Per-step isolation is the degenerate case (stage of one).

Proposed stages for the album pipeline (image-level, face-level, scene-level — the user's split):
```
image_features : detect_persons, insightface_detect_faces, filter_faces, score_face_frontal,
                 detect_face_orientation, align_faces, insightface_score_*, extract_face_embeddings,
                 score_iqa, score_ava, score_occlusion, score_tilt, extract_scene_embedding
                 exports: face_records(+embeddings), *_scores, scene_embeddings, insightface_faces(meta)
                 kept-inside (never exported): aligned_faces (crops)
face_clustering: quality_gate, build_face_knn_graph, cluster_face_components, select_face_exemplars,
                 merge_face_clusters, attach_holdout_faces, apply_diameter_cap, assign_people_clusters,
                 identity_refinement, cluster_by_identity        exports: people_clusters, cluster_*
scene_org      : build_scene_distance?, cluster_scenes, select_best, straighten_images
                 exports: selected_images, composite_scores, scene_clusters
```
(Exact grouping tuned in Phase 2 so each stage loads a disjoint model set.)

### 3.2 Marshaling across the boundary

The child gets a **fresh** `PipelineContext` seeded with: `source_directory`, `image_paths`, the
stage's declared **imports** (picklable values produced by earlier stages), and a locally
constructed `cache_handler` (disk-backed, reconstructable from its path — NOT pickled). The child
runs the stage's steps via the existing executor loop (so Tier-1 `release()` still runs between
steps inside the child, bounding the child's own peak). The child returns a dict of the stage's
declared **exports**; the parent merges them into its long-lived context.

Not crossed: `on_progress` (relayed via a Queue → parent invokes the real callback), non-picklable
handles, and large kept-inside intermediates (crops). `spawn` context (Windows default); worker is
a module-level function in `sim_bench.pipeline`.

### 3.3 Error / crash handling (the "also in case of an error" ask)

- Stage step raises → child catches, returns serialized `{error_type, message, traceback}` →
  parent builds a failed `StepResult` exactly like the inline path; `fail_fast` respected.
- Child dies (OOM / segfault) → non-zero exit, no result on the queue → parent detects, emits a
  failed `StepResult("<stage>", "child process exited N — likely OOM")`. **Parent stays alive.**
- Tier-1 `release()` remains best-effort in `finally`; a `release()` bug never masks the result.

### 3.4 Off by default (zero behavior change)

No `isolated_stages` declared → executor runs exactly as today, byte-identical. Isolation is opt-in
per pipeline spec (album pipeline turns it on; FC-v2 anchor runs can stay inline).

## 4. Acceptance criteria

- **AC1 (memory):** `faces` pipeline on Austria (474) staged peak RSS **< 1.2 GB** (from 2.1 GB).
- **AC2 (equivalence, BINDING):** staged run output byte-identical to inline on the Budapest anchor
  — `n_clusters == 15`, `n_faces == 340`, same cluster sizes, same `selected_images`.
- **AC3 (error):** a step that raises inside an isolated stage → run fails gracefully, `failed_step`
  correct, **parent process alive**.
- **AC4 (crash):** a stage child that OOMs/`os._exit(1)` → parent reports failure, stays alive.
- **AC5 (off = identical):** no stages declared → byte-identical to today (regression guard).
- **AC6 (scale):** Germany (788) `default` pipeline completes without OOM on the dev box.
- **AC7:** `tests/face_clustering/e2e_budapest/` stays green (V2 baseline gate).

## 5. Non-goals

- GPU/CUDA memory (CPU is the shipping target). CUDA `empty_cache` is out of scope.
- Rewriting steps' internals. Isolation is an executor concern; steps are unchanged except the
  (existing) `release()` hook.
- A distributed / multi-machine runner. Single host, multiple short-lived child processes.

## 6. Risks

- **Pickling cost / non-picklable exports.** Mitigated: exports are small + declared; a startup
  check asserts every declared export key is picklable before a run.
- **Per-stage torch re-import (~seconds).** Acceptable vs minute-scale runs; 3 stages = 3 spawns.
- **Two code paths (inline vs isolated).** Mitigated by AC5 (off=identical) + AC2 (on=identical
  output) as permanent regression guards.

## 7. Phasing

- **Phase 0 (DONE 2026-07-30):** add missing `extract_face_embeddings.release()` (~360 MB); probes;
  re-diagnose + reprice SIGHTING-117.
- **Phase 1:** executor subprocess-stage primitive (spawn worker, marshal imports/exports, progress
  relay, error + crash handling) behind a spec flag; default off. Unit + error + crash tests.
- **Phase 2:** declare image/face/scene stages for the album pipeline; AC1/AC2/AC5 gates.
- **Phase 3:** wire Albumify arm + API to staged mode for large trips; measure Austria/Germany (AC6).
- **Phase 4:** docs (`docs/architecture/data_flow.html`, `overview.md`), REVIEW.md, flip to Implemented.

## 8. Open questions (for approval)

1. Stage grouping: three stages (image/face/scene) as above, or finer? (affects spawn count vs peak)
2. Import/export declaration: explicit per-stage lists (safer, more boilerplate) vs auto-derived
   from steps' `requires`/`produces` (less boilerplate, risk of pickling something large)?
3. Should staged mode auto-enable above an image-count threshold, or always be an explicit flag?
