# spec-097 — Occlusion Scoring in the Pipeline (penalize, don't delete)

**Created**: 2026-07-07 · **Status**: Draft · **Priority**: P2
**Depends on**: spec-096 (benchmark; winner = CLIP tile probe 0.83 scene PR-AUC,
pending adjudicated refit). Design discussed + user-approved in session 2026-07-07.

## Goal
Photos with lens occlusion (finger/strap blobs) get penalized in Albumify's
selection ranking — scaled by severity and by whether the MAIN SUBJECT is covered.
Never auto-delete; a human sees the ranking.

## Stage 1 — flat occlusion penalty (ships first)
1. **`score_occlusion` pipeline step** (universal_cache-backed, spec-053 thin-step):
   - model behind a small interface: v1 = **CLIP tile probe** (ViT-B/32 embeddings,
     global+tile-max logistic head; weights = small artifact checked into
     `models/occlusion/`, versioned in cache metadata).
   - writes `context.occlusion_scores {path: P(occluded)}` and
     `context.occlusion_tiles {path: [9 tile scores]}` (localization for UI/Stage 2).
2. **`occlusion_penalty` in `select_best`** (spec-084 halves pattern, next to
   person_penalty): `penalty = w * P * area_factor(n_tiles ≥ τ)`. Config in
   pipeline.yaml; default gentle (L1-corner blob dents, never disqualifies).
3. Surfacing: Albumify Results shows the penalty component (existing pattern);
   Analysis Studio gains an `occlusion` column in image_quality.

## Stage 2 — subject-aware severity
Subject mask ladder (cheapest-first; updated 2026-07-07 after the CLIP-semantic panels):
1. faces ∪ person boxes (already in context: insightface + detect_persons;
   covered 111/122 Budapest — the primary source)
2. else: **CLIP-semantic saliency** — tile↔global embedding similarity ("which
   region is most what-this-photo-is-about"); reuses the occlusion probe's tile
   embeddings, and drew visibly better subject boxes than classical saliency
   in the research panels
3. last resort: spectral-residual × center prior (cv2.saliency, zero-torch)
4. (only if all above fail in practice: U2-Net-lite, ~5MB)

Blur/occlusion region: tile scores from Stage 1 (learned, preferred) and/or the
LoG **flat-but-noisy** box (`occlusion_bench/saliency.blur_bbox`).

```
penalty = w_subject * P * overlap(occluded_region, subject_mask)   # photo-ruiner
        + w_bg      * P * area(occluded_region - subject_mask)     # cosmetic
```
Maps to severity: background-only blob → L1-ish; subject partly covered → L2;
subject mostly covered → L3.

## Research evidence (2026-07-07, `D:\occlusion_dataset\research_saliency\index.html`)
122 Budapest images, 3 saliency variants (SR / fine-grained / SR×center-prior)
with dual boxes (green=salient, red=blur region):
- SR×center gives the most stable subject boxes on consumer photos.
- Naive "flattest region" localization finds SKY, not fingers. Fix = the
  **flat-but-noisy** rule (defocused occluders sit on a sensor-noise floor;
  sky is flat AND near-silent) — the same sign-flip the spec-096 LR discovered.
- Physics bands OVERLAP (dark finger ≈ hazy sky at med/g 0.09-0.13): no single
  threshold separates them. Accepted: benign sky boxes are harmless because the
  Stage-2 penalty is overlap-driven (sky never overlaps the subject).
  Verified: both real fingers boxed; their subject-overlap ≈ 0 (correct — corner
  occlusions not covering the person).
- Durable localization = learned (tile probe now; per-patch head trained on the
  FREE synthetic masks later), not physics thresholds.

## Rules
- Penalty driver is the LEARNED P(occluded), never raw blurriness (bokeh safety).
- Production model choice finalized only after spec-096 adjudicated refit.
- Every score cached (image, feature_type=occlusion_<model_ver>).

## Out of scope
Auto-delete; crop-suggestions; video; retraining pipelines (spec-096 owns models).
