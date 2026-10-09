# spec-111 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap tier: **v1.5 (COULD)** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Draft (spec) — P0 + P1 core built 2026-10-05 ahead of the roadmap's v1.5 placement. Needs an explicit owner call (ROADMAP decision #4).

## How it was verified so far
| Increment | Evidence |
|---|---|
| P0 spine: `Correction`/`VerifyResult` types, `borderline()` escalation trigger, `evaluate()` baseline gate | `tests/verification/test_verification_harness.py` (9 tests): band selection, empty inputs, helps/hurts/net-zero |
| P1 `verify_face_clusters`: merge candidates (just above threshold), split candidates (high spread), injected adjudicator, pairwise `evaluate_clustering()` gate | `tests/verification/test_face_clusters.py` (9 tests) with a fake adjudicator; 18 verification tests green. Pure, no VLM, no GPU |

## The non-negotiable gate
A checkpoint ships only if **measured** to beat the deterministic baseline on labeled data
(Budapest 340 faces → 15 clusters; spec-102 trips). Not yet run for any checkpoint.

## Open
- T1.3 real VLM adjudicator on face crops + the AC1 run on Budapest (needs the GPU worker).
- T1.4 thin pipeline step after `cluster_face_components` applying MERGE/SPLIT forward (unblocked).
- P2 `verify_selection`, P3 `verify_scenes` / `verify_triage`, P4 cost caps + config.

## Timeline
- 2026-10-05 — spec written (spec-105 folded in as checkpoint #1); P0 and P1 core built.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| Eval sets for the ship gate | Budapest reference run (`tests/face_clustering/e2e_budapest/README.md`: 340 faces → 15 clusters, truth for `evaluate_clustering`); spec-102 trips: [Budapest](../../reports/2026-07-17_albumify_vs_vlm_budapest/report.html) · [Austria](../../reports/2026-07-18_albumify_vs_vlm_austria/report.html) · [Germany](../../reports/2026-07-18_albumify_vs_vlm_germany/report.html) |
| [Agentic layer design note](../106-albumify-web-showcase/agentic_layer_design.html) | the escalation-not-orchestration pattern this spec formalizes |
| Tests | `tests/verification/test_verification_harness.py` (9) · `tests/verification/test_face_clusters.py` (9) |

No experiment report yet: no checkpoint has been measured against its baseline.
