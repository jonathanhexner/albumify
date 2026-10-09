# spec-103 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **B (optional)** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress since 2026-07-22 — `build_scene_distance` step wired, gated off by default. Labels (T1) and ARI metric (T4) open.

## How it was verified
- Three trips eyeballed: the additive time+visual blend over-merged (a 4-minute-apart pair wrongly glued) and a
  hand-rolled single-linkage clusterer chained a 49-photo "scene". Superseded by the one-sided short-range
  boost fed to the existing HDBSCAN. `ut_SceneDistance_never_pushes_apart` guards the shape.
- Report: `reports/2026-07-21_scene_clustering_fusion/`. LEARNINGS 2026-07-22.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [Scene clustering fusion report](../../reports/2026-07-21_scene_clustering_fusion/report.html) | Budapest / Austria / Germany: unplaceable-orphan bucket shrank 27→15, 104→69, 191→94 (−34…−51%); time does most of it, GPS refines; Germany (34% GPS) improved most. Correctness vs hand labels still open |
| [REVIEW](REVIEW.html) | code review of the gated-off step |
