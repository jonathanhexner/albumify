# spec-111 — AI Verification Layer

**Status:** In Progress — P0 (spine) + P1 core (`verify_face_clusters`, pairwise gate) built 2026-10-05, 18 tests green. Built ahead of the roadmap's v1.5 placement; owner call pending (ROADMAP decision #4). Progress: [updates.html](updates.html)
**Owner:** Jonathan / Claude
**Priority:** P2 (the marquee "shine" feature; quality depth on top of the deterministic engine)
**Relation:** subsumes [[spec-105]] (VLM merge arbiter) as checkpoint #1; shares the VLM seam with
[[spec-108]]; its corrections honor the [[spec-107]] set_aside / show-everything principle.

## 1. Problem

The curation pipeline is deterministic and fast, but it makes **borderline calls** it sometimes gets
wrong — and those errors are the *visible* ones a user notices: two people merged into one cluster, a
genuinely-different moment collapsed as a near-dup, the wrong frame picked as "best" of a burst, a
standout photo dropped just past the K cutoff, a real photo set aside as junk. The deterministic
signals (cluster distances, composite scores, triage confidence) are *uncertain* exactly at these
boundaries.

We want a **second reviewer** — an AI that double-checks the pipeline's *borderline* decisions,
**grounded in our own trained models**, and corrects the likely mistakes. Not an orchestrator that
sequences the pipeline (that stays a deterministic DAG); a set of bounded **verifier steps**.

## 2. Goal / non-goals

**Goal:** a family of verifier steps, one per decision checkpoint, each of which (a) escalates only the
borderline subset, (b) adjudicates with a VLM grounded in our models' scores, (c) writes corrections
forward into the context, (d) is independently shippable and **proven to beat the deterministic baseline**
before it's enabled.

**Non-goals:** a meta-orchestrator that loops over / re-sequences the pipeline (rejected — nondeterminism,
untestable); re-reviewing *confident* decisions (cost); VLM per-image on everything; silent overrides
(all corrections are surfaced + reversible, per spec-107).

## 3. Architecture — verifier steps, "verify early, correct forward"

Each checkpoint is a **normal pipeline step placed immediately after the step it verifies**, so its
corrections propagate forward to every downstream consumer — no loop-back, no meta-orchestrator.

```
… cluster_face_components → [verify_face_clusters] → merge_face_clusters …
… cluster_scenes          → [verify_scenes]        → select_best …
… select_best             → [verify_selection]      (soft-K + best-of-cluster)
… triage_content          → [verify_triage]        → scoring …
```

- **Thin steps + domain module** (spec-053): a `verify_<x>` step wraps `verify_<x>(inputs, vlm) ->
  corrections` in a new `sim_bench/verification/`. Framework-agnostic helpers; step is the translator.
- **Escalation trigger** (bounds cost): each step reads confidences/margins from context (cluster
  distance near threshold, top-2 composite within ε, triage confidence in-band) and escalates **only
  that subset**. Confident decisions stand untouched.
- **Grounded in our models** (AI + tools, not AI instead of tools): the VLM adjudicates *with* our
  trained signals as evidence — the PhotoTriage/siamese pairwise model for best-of-cluster, cluster
  distances for merges, the triage classifier for junk. Cleanest composition: our model makes the call;
  the VLM reviews disagreements / low-confidence picks.
- **VLM via the seam**: uses the `VLMProvider` (spec-108) — in-container on the Modal GPU worker.
- **Corrections flow forward** into context (reassign clusters, promote/demote picks, un-set-aside);
  surfaced + reversible (spec-107 show-everything). Never silent.
- **Can stop** via the existing step-failure mechanism (rare; norm is correct-and-continue).

## 4. Checkpoints

| # | Step | Catches | Escalation trigger | Grounded in |
|---|---|---|---|---|
| 1 | `verify_face_clusters` | split one person / merged two | cluster distance near merge threshold | face embeddings + distances (= **spec-105**) |
| 2 | `verify_scenes` | collapsed real moments / missed dups | DINOv2 distance near dedup cutoff | scene embeddings + time/GPS |
| 3 | `verify_selection` (best-of-cluster) | wrong "best" of a burst | top-2 composite within ε | **PhotoTriage/siamese** + IQA/AVA |
| 4 | `verify_selection` (soft-K) | standout just past the cut | score near K cutoff | composite scores |
| 5 | `verify_triage` | wrongly set aside / kept junk | triage confidence in band | triage classifier (spec-107) |

## 5. The non-negotiable gate — each checkpoint must beat the baseline

AI double-checking *sounds* obviously good but can add cost + noise without adding quality. So each
checkpoint ships **only if measured to beat the deterministic baseline** on the human-labeled / spec-102
eval sets (does the VLM's correction agree with human judgment more than the pipeline alone?). Faith is
not evidence. A checkpoint that doesn't win on the eval stays disabled.

## 6. Acceptance criteria

- **AC1** Each enabled checkpoint improves agreement-with-human vs the deterministic baseline on the
  eval set (measured, recorded) — else it stays off.
- **AC2** Escalation is bounded: only the borderline subset hits the VLM; cost/album measured + capped.
- **AC3** Corrections are surfaced + reversible (never silent); the album shows "promoted/merged because…".
- **AC4** The pipeline stays deterministic outside the verifier steps; no loop / meta-orchestrator.
- **AC5** Each verifier is a thin step (spec-053) + a tested `verify()` helper (fake-VLM unit test).

## 7. Dependencies & risks

- VLM seam (spec-108); our trained models (PhotoTriage/siamese in `select_best`, triage classifier
  spec-107); the eval sets (spec-102 + labeled occlusion/trips).
- Risk: cost/latency if triggers are loose → tight borderline-only triggers + measured cap (AC2).
- Risk: a checkpoint adds noise not quality → the AC1 baseline gate blocks it.
- Risk: nondeterminism creep → confined to verifier steps on borderline subsets; the DAG is otherwise
  deterministic; SIGHTING-122-style tolerance applies.

## 8. Definition of done
≥1 checkpoint (start with spec-105's `verify_face_clusters`) implemented as a step + helper, measured
to beat baseline (AC1), bounded (AC2), surfaced (AC3); REVIEW.md; others follow the same gate. See `tasks.md`.
