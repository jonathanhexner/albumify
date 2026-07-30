# Tasks — Spec 103: Scene clustering — time+geo fusion (graceful degradation)

> Experiment that LANDS in the pipeline. Winner ships as a flag-gated `cluster_scenes` method.
> Budapest (geo-rich) + Germany (geo-poor) are the two labeled reference trips; Germany is the
> binding degradation stress test. Default behavior unchanged until the gates clear.

## T0 — Decisions [DRAFT — awaiting approval, see spec D1-D8]
- [ ] Confirm D1-D8. Key binding ones: **D2 no-imputation**, **D5 hand-label Budapest+Germany**,
      **D7 ship gated (default OFF)**. Open: is solo N=1 labeling acceptable for the reference (per
      spec-102 lineage), or do we want ≥2 labelers on the scene groups?

## T1 — Reference labels (gates everything; do first)
- [ ] T1.1 Labeling tool: group-the-thumbnails HTML (drag/assign each photo to a scene group), emits
      `scene_labels.json` per trip. Written rubric for "what is one scene" baked into the page.
- [ ] T1.2 Hand-label **Budapest** (122) and **Germany** (797 — may sample a labeled subset if full
      is too heavy; document the sample). Save under the spec dir.
- [ ] T1.3 Loader + validator for `scene_labels.json` (every labeled image maps to exactly one group).

## T2 — Fusion helper (spec-053 shape, framework-agnostic) — returns a DISTANCE
- [x] T2.1 `sim_bench/scene_cluster/geo_time_fusion.py`: `SceneDistanceBuilder` (Path A, SHIPPED) +
      `SceneDistanceFuser` (additive D3, superseded experiment). Both: typed `SceneDistanceInputs`,
      `calc() -> SceneDistanceResult` (matrix + image_ids + per_image_signal_used). Returns a distance.
- [x] T2.2 **Path A** distance (validated): `d = visual_cos · (1 - boost·exp(-dt_sec/tau_sec))` — a
      one-sided short-range time boost (pulls near-simultaneous shots, never pushes apart). Time from
      GeoMetadata.timestamp else filename stem. (Additive D3 blend kept in `SceneDistanceFuser` for the
      sweep but NOT shipped — it over-merged.)
- [x] T2.3 Unit tests `tests/scene_cluster/test_scene_distance_builder.py`: short-range pull shrinks
      distance; far-apart == pure visual; **never pushes apart**; missing-time == pure visual (no
      imputation); symmetric/zero-diag/aligned; determinism; result type.

## T3 — Pipeline integration (Option B — dedicated step, gated by presence) — DONE (Path A)
> NOTE: shipped **Path A** (`SceneDistanceBuilder`: visual × one-sided short-range time boost), NOT the
> D3 additive blend — the additive term over-merged (see report over-merge case). `requires` is
> `{scene_embeddings}` only (time from EXIF else filename; geo unused in Path A), so the step degrades
> without an `extract_geo_metadata` dependency.
- [x] T3.1 New step `build_scene_distance` (thin, ≤80 LOC): `sim_bench/pipeline/steps/build_scene_distance.py`.
- [x] T3.2 Two new `PipelineContext` fields `scene_distance` + `scene_distance_signal` (documented in
      context.py + classes.html + data_flow.html).
- [x] T3.3 `cluster_scenes`: ONE None-guarded branch — if `context.scene_distance` present, cluster it
      with `metric="precomputed"`; else unchanged.
- [x] T3.4 `configs/pipeline.yaml`: documented `build_scene_distance` block (boost + tau_sec); NOT in
      `default_pipeline`. Typed `BuildSceneDistanceConfig` registered.
- [x] T3.5 **Absent-step regression (binding A5)**: `ut_A5_boost_zero_equals_embedding_path` +
      `ut_cluster_scenes_unchanged_when_distance_absent` in `tests/pipeline/test_build_scene_distance.py`
      prove the precomputed path == embedding path (byte-identical when absent). Full face E2E unaffected
      (step not in any default pipeline).

## T4 — Experiment arms + scoring
- [ ] T4.1 Scoring: ARI + homogeneity/completeness vs reference, **stratified by geo availability**
      (report the geo-absent stratum separately — A4).
- [ ] T4.2 Run arms A0 (baseline) / A1 (time+visual) / A2 (time+geo+visual) / A3 (two-stage, reuse
      `geo_temporal_segment`) on Budapest + Germany.
- [ ] T4.3 **Degradation gate (binding A1)**: assert A2/A3 ARI ≥ A0 on geo-absent photos and on
      Germany overall. A regression here blocks the spec.

## T5 — Report + close-out
- [ ] T5.1 `reports/<date>_scene_clustering_fusion/report.html` + summary.md + EXPERIMENTS.md entry:
      inline before/after cluster galleries per trip, metric tables, the degradation-gate result,
      geo-availability caveats stated.
- [ ] T5.2 Docs (A7): `docs/architecture/data_flow.html` (geo→scene edge), `classes.html` (helper +
      Inputs/Result), pipeline.yaml comments, overview if the step contract shifts.
- [ ] T5.3 `/code-review` → REVIEW.md; resolve High findings. Flip status → Implemented only after
      A1/A5 (the binding gates) are green.
- [ ] T5.4 LEARNINGS.md entry; decide whether the VLM-scene arm (spec-099-adjacent) is worth a
      follow-up spec based on the ARI ceiling the metadata fusion hits.

## Deferred design ideas (parked — decide later)
- **DEFERRED (user, 2026-07-21): scene cluster-merge pass, mirroring face recognition.** After initial
  clustering, run a merge step like the face pipeline's cluster-merging (centroid/medoid similarity +
  a threshold, or a VLM/model arbiter as in spec-099) to fuse over-split scenes. Could subsume the
  flat-vs-two-stage tension: cluster conservatively, then merge. NOT scoped yet — revisit after the
  A0-A3 arm comparison shows where the residual over-split/over-merge errors actually are.

## Arms built during the exploration (ahead of formal T-order)
- **A2 flat fusion** helper `sim_bench/scene_cluster/geo_time_fusion.py` — built. Time-scale sweep in
  `scripts/exp_scene_sweep.py`. Eyeballing showed scale=30min over-merges (see report 2.2).
- **A3 two-stage** helper `sim_bench/scene_cluster/two_stage.py` (segment by time/geo, then visual
  connected-components within) — built + unit-tested (`tests/scene_cluster/test_two_stage.py`).
  Compared against A0/A2 in `scripts/exp_scene_sweep.py` on the two hand-flagged reference pairs.

## Notes / risks
- Do T1 FIRST — with no reference labels the whole thing degenerates to "methods differ" (the EXP-1
  trap). Labeling is the long pole.
- Germany is 4% geo: it is the degradation test, not an afterthought. Tune weights on Budapest, then
  Germany must not regress vs visual-only.
- Keep DINOv2 fixed across all arms (isolate the time/geo contribution; embedding choice is a separate
  spec).
