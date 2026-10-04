# spec-111 — Tasks (AI Verification Layer)

**Status:** Draft — no implementation until spec.md approved. Rationale: `spec.md`.
**Shape (decided):** verifier *steps* after each decision (verify-early, correct-forward); thin steps +
`sim_bench/verification/` domain module; VLM via the spec-108 seam, grounded in our models; **no
meta-orchestrator, no loop**. Each checkpoint ships only if it beats the deterministic baseline.

## P0 — Shared infrastructure (the layer's spine)
- [ ] T0.1 `sim_bench/verification/` module: a `VerifyResult {corrections, escalated, reasons}` type +
      a shared escalation helper `borderline(items, margin) -> subset` (reads confidences/margins).
- [ ] T0.2 Correction-application contract: how a verifier writes forward into context (cluster
      reassignment, pick promote/demote, un-set-aside) + surfaces it (reason + reversible, spec-107).
- [ ] T0.3 **Baseline-eval harness**: given a checkpoint + the labeled/spec-102 eval set, measure
      agreement-with-human of {deterministic baseline} vs {+verifier}. This is the AC1 gate every
      checkpoint must pass before it's enabled. Reuse spec-102 metrics.

## P1 — Checkpoint #1: verify_face_clusters (fold in spec-105)
- [ ] T1.1 `verify_face_clusters` step after `cluster_face_components`: escalate cluster pairs whose
      distance is near the merge threshold; VLM adjudicates "same person?" grounded in embeddings/distance.
- [ ] T1.2 Apply merge/split corrections forward; surface them. Thin step + `verify_face_clusters()` helper + fake-VLM test.
- [ ] T1.3 **AC1 gate**: measure vs baseline on labeled clusters (spec-102/Budapest 340→15). Enable only if it wins.
- [ ] T1.4 Retire spec-105 as standalone → it is checkpoint #1 here.

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
