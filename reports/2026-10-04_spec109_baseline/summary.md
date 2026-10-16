# spec-109 T0.1 — full pipeline CPU timing baseline (2026-10-04)

Full `default_pipeline` (38 steps, incl. occlusion CLIP + GeoCalib tilt) on 75 Slovakia images
(57 scored after triage set aside 11), CPU, 768 px. **Total 512 s = 9.0 s/image.**

## Per-step cost (slowest), s/image

| step | s/img | model |
|---|---|---|
| triage_content (RAM++) | **3.0** | RAM++ swin-L |
| score_tilt (GeoCalib) | **2.6** | GeoCalib |
| insightface_detect_faces | 1.1 | SCRFD |
| extract_face_embeddings | 0.9 | ArcFace |
| score_occlusion | 0.5 | CLIP ViT-B/32 |
| extract_scene_embedding | 0.3 | DINOv2 |
| detect_persons | 0.2 | YOLOv8 |
| score_ava / select_best / score_iqa / … | ≤0.1 each | — |

## Findings

1. **9 s/img on CPU → non-interactive.** 100 photos ≈ 15 min, 300 ≈ 45 min. The live product's
   ≤3–5 min bar is impossible on CPU. **GPU (P2) is mandatory, not optional.**
2. **The cost moved.** vs the 2026-10-01 lean `faces` run (face stack = 75%), the full pipeline is now
   dominated by the two heaviest *added* models: **RAM++ triage (3.0 s) + GeoCalib tilt (2.6 s) = 62%**
   of the 9 s/img. The face stack (detect + embed = 2.0 s) is 23%.
3. **All top costs are GPU-friendly models** (RAM++, GeoCalib, SCRFD/ArcFace, CLIP, DINOv2, YOLO) —
   GPU should cut the per-image time ~10–30× across the board.
4. **Defer-embedding is NOT a free win** (dependency finding): `select_best` default
   `use_face_subclusters=True` consumes `face_clusters` → needs embeddings on all faces before
   selection. Embeddings stay on the critical path; speed them up (P1/P2), don't skip them.

## Implication

The baseline confirms the live product cannot run on CPU. The real blocker is **choosing + provisioning
a GPU host (Modal, per spec-106)** — then P1 (parallelize per-image steps) + P2 (GPU) deliver the
≤3–5 min target. Cheap per-image reconsideration also worth a look: GeoCalib tilt (2.6 s) only yields a
tie-breaker penalty — making it GPU-only or optional is low-hanging.

Repro: `.venv/Scripts/python scripts/spec109_baseline_timing.py` (reads `_slovakia_triage_work`).
