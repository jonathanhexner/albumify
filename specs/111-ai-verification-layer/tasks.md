# spec-111 — Tasks (AI Verification Layer)

**Status:** Draft — no implementation until spec.md approved. Rationale: `spec.md`.
**Shape (decided):** verifier *steps* after each decision (verify-early, correct-forward); thin steps +
`sim_bench/verification/` domain module; VLM via the spec-108 seam, grounded in our models; **no
meta-orchestrator, no loop**. Each checkpoint ships only if it beats the deterministic baseline.

## P0 — Shared infrastructure (the layer's spine)  ✅ DONE 2026-10-05
- [x] T0.1 `sim_bench/verification/types.py`: `Correction` (kind/targets/reason/detail) + `VerifyResult`
      (checkpoint, n_reviewed, escalated, corrections) + correction kinds (merge/split/promote/demote/
      unset_aside/reassign). `escalation.py`: `borderline(items, margin, band)` — the cost-bounding
      trigger (escalate only items within `band` of a decision fence; confident ones stand).
- [x] T0.2 Correction *vocabulary* + surfacing contract defined (Correction.reason shown + reversible,
      spec-107). Per-checkpoint forward-application lands with each checkpoint (P1+).
- [x] T0.3 **Baseline-eval harness** `eval.py`: `evaluate(baseline, verified, truth)` → `EvalResult`
      with agreement-with-human for both + helped/hurt breakdown of changed items; `.improved` is the
      AC1 ship gate (raises agreement AND helped >= hurt). 9 tests green.
- **Gate:** ✅ the spine is in — every checkpoint plugs its margin-fn into `borderline`, returns a
      `VerifyResult`, and must pass `evaluate(...).improved` before it's enabled.

## P1 — Checkpoint #1: verify_face_clusters (fold in spec-105)
- [x] T1.1 `sim_bench/verification/face_clusters.py`: pure `merge_candidates` (pairs just above the merge
      threshold via `borderline`) + `split_candidates` (high internal spread) + `verify_face_clusters(...)`
      with an INJECTED adjudicator (VLM in prod, fake in tests) → MERGE/SPLIT `Correction`s. `cosine_distance`.
- [x] T1.2 AC1 gate mechanism: `evaluate_clustering()` — pairwise reduction (co-clustered? vs same-person?)
      → reuses P0 `evaluate()`. 18 verification tests green (candidates, verify, gate helps/hurts).
- [ ] T1.3 (needs VLM+data) real adjudicator wrapping the VLM seam on face crops; run the AC1 gate on
      **Budapest 340→15** ground truth — enable the checkpoint only if it beats baseline.
- [ ] T1.4 (needs integration) thin pipeline step after `cluster_face_components` that applies MERGE/SPLIT
      forward into `context.face_clusters` + surfaces corrections.
- [x] T1.5 spec-105 superseded → this is checkpoint #1.

## P2 — Checkpoint: verify_selection (best-of-cluster + soft-K)
- [ ] T2.1 best-of-cluster: when top-2 composite within ε in a near-dup group, VLM picks, grounded in the
      **PhotoTriage/siamese** pairwise model; correct the pick forward.
- [ ] T2.2 soft-K: K is a target, not a hard wall — promote a standout just past the cutoff / drop a weak
      one within it (bounded band). Resolves the global-K ambiguity (K = target + band).
- [ ] T2.3 AC1 gate on the eval set; AC2 cost cap.

## P3 — Remaining checkpoints (each behind the same gate)
- [ ] T3.1 `verify_scenes` after `cluster_scenes` (borderline DINOv2 distances).
- [ ] T3.2 `verify_triage` after `triage_content` (borderline triage confidence).
- [ ] Each: step + helper + fake-VLM test + AC1 baseline-eval gate; enable only if it beats baseline.

## P4 — Harden + ship
- [ ] T4.1 Per-album cost/latency measured + capped (AC2); corrections surfaced in album.json (AC3).
- [ ] T4.2 Config: enable/disable + thresholds per checkpoint (like occlusion/tilt penalties).
- [ ] T4.3 CHANGES_LOG + `/code-review` → REVIEW.md.

## Out of scope
A meta-orchestrator that loops/re-sequences the pipeline (rejected) · re-reviewing confident decisions ·
per-image VLM on everything · silent overrides · shipping a checkpoint that doesn't beat baseline.
