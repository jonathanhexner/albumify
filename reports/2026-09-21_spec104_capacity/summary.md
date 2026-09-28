# spec-104 capacity measurement — peak RSS vs album size (2026-09-21)

**Goal**: measure how peak memory scales with album size N and what per-step process isolation
(spec-104 v1, `isolate_steps=True`) costs vs the in-process path, on the real `faces` pipeline
(YOLO / InsightFace / MediaPipe / DINOv2 / AVA / pyiqa, CPU, no feature cache).

**Method**: `scripts/spec104_capacity_measure.py` — N = 100/300/500 Germany images (768 px),
each config in a fresh spawn child, monitor samples peak **process-tree** RSS (parent + step
children). Windows 10 dev box, 32 GB RAM.

| N | mode | peak MB | dur s | s/img | faces |
|---|---|---|---|---|---|
| 100 | isolated | 2,578 | 381 | 3.8 | 170 |
| 300 | isolated | 2,696 | 532 | 1.8 | 455 |
| 500 | isolated | 2,845 | 817 | 1.6 | 794 |
| 100 | in-process | 2,379 | 240 | 2.4 | 170 |
| 300 | in-process | 2,449 | 473 | 1.6 | 454 |
| 500 | in-process | 2,532 | 743 | 1.5 | 796 |

**Findings**:
1. **Flat scaling**: peak ≈ model floor + tiny linear term — isolated **2,511 MB + 0.67 MB/img**,
   in-process **2,341 MB + 0.38 MB/img**. Models are ~93% of peak; context is KB/image; no N² term
   active (spec-103 `build_scene_distance` is the only N×N producer and is gated off).
2. **Capacity is runtime-bound, not memory-bound**: 16 GB ≈ 20k images/run, 32 GB ≈ 44k — but at
   ~1.6 s/img CPU, 20k images ≈ 9 h. Production sizing: ~4 GB/worker, k ≈ RAM/4 GB capped by cores.
3. **Isolation costs +~300 MB peak and +10% time at N=500** (parent baseline alongside child +
   marshal copies). Cheap for the crash/OOM-survival guarantees.
4. **In-process no longer loses on peak** post-SIGHTING-117 `release()` fixes — v1 isolation buys
   robustness, not a peak-RSS win, on this pipeline.
5. **Anomaly**: face counts diverge slightly between modes (±1–2 at N=300/500) → SIGHTING-122;
   the REVIEW.md §5 equivalence test must root-cause or carry a tolerance.

Full input set: `D:\albumify_vs_vlm\germany\imgs768` (788 imgs). Report: [report.html](report.html).
