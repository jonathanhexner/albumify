# PRD-080 — Scene selection by dominant identity (not face count)

Status: **Draft / parked** (return after spec-079 Stage 3)
Raised by: Jonathan Hexner, 2026-06-05
Related: spec-079 (Albumify shared core)

## The problem

`sim_bench/pipeline/steps/cluster_by_identity.py` currently sub-divides each
scene cluster **by the number of faces** in an image. That is a proxy, not the
goal.

> Observed during spec-079 clustering review: "you're grouping by the number of
> faces. That's not quite the goal. The goal is to group by the identities of
> the top-N dominant faces that pass all the criteria, and later choose the best
> image."

## Intended behavior

Within a scene cluster, the unit of grouping should be the **identity set of the
top-N dominant faces** that pass all quality/pose/size criteria — then pick the
single best image per identity-set.

1. For each image in a scene, take the **top-N dominant faces** (by size /
   det_score / centrality — TBD) that pass the gates.
2. Resolve each of those faces to its **identity** (`people_clusters` /
   `cluster_people` output).
3. Group images by their **dominant-identity signature** (the set of identities
   present among the top-N), not by raw face count.
4. Choose the **best image** per group for selection/export.

## Why it matters

Two photos with "3 faces" today land in the same sub-cluster even if they are of
different people. The goal is "the best photo of *these specific people*", which
requires identity-aware grouping. This directly affects `select_best` and the
exported album quality.

## Open questions (resolve when un-parking)

- N for "top-N dominant"? Dominance metric (area, det_score, centrality, a mix)?
- Identity source: `people_clusters` or `refined_people_clusters`?
- Tie-breaking for "best image" (IQA/AVA composite already exists).
- Does the identity-signature need ordering, or is it a set?
- Backward-compat: keep face-count path behind a flag during migration?

## Scope / modules (when implemented)

- `sim_bench/pipeline/steps/cluster_by_identity.py` (the rework)
- consumes `context.people_clusters` + per-face dominance metrics
- `select_best` downstream consumer
- tests: scene with same count / different identities must split

## Not now

Parked deliberately. spec-079 leaves `cluster_by_identity` untouched; this PRD
captures the intent so it isn't lost. Promote to a full `spec.md` + `tasks.md`
when picked up.
