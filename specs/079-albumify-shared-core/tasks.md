# spec-079 — Tasks

**Refocused 2026-06-06** onto the *execution / config* divergence (Track A below):
Albumify and FC v2 share the pipeline executor but run different clustering steps
with differently-typed config, so identical input → different clusters
(SIGHTING-096). Goal: one typed contract (`FCParams`), one clustering step-set
(the 8 unified steps), one runner — so either app, given identical config, runs
identically. See `UNIFIED_CONFIG_PLAN.html` and `CONFIG_DIVERGENCE.html`.

Each stage's Definition of Done: listed tests green + equivalence anchor holds +
`/code-review` on changed modules shows no open High finding + committed. Mode:
autonomous within a stage, STOP at stage boundaries to report.

Anchor (binding): `D:\Budapest2025_Google` + `profile_5.json` →
**8 clusters, sizes [26,20,12,7,3,2,2], 72 assigned** (FC v2 reference
`a588521b…`). Constants in `tests/_budapest_baseline.py`.

---

# Track A — Unify pipeline execution + config

## Stage 0 — Reproduce & diagnose  `[x]` DONE
- [x] `scripts/run_profile.py` — headless FC v2 runner from a profile; Step 0 PASS (8 clusters)
- [x] `scripts/capture_albumify_baseline.py` (profile_5) — Albumify = 20 identities
- [x] `scripts/diff_fcconfig.py` — 3 differing gate fields (blur_min, min_face_area, cap)
- [x] Confirmation: nogates → FC v2 = 10 (not 20) ⇒ divergence is TWO layers (config + code path)
- [x] `tests/_budapest_baseline.py` repointed to profile_5 reference
- [x] SIGHTING-096 filed

## Stage 0b — Localize the 10-vs-20 gap  `[x]` DONE  (diagnostic)
- [x] `scripts/localize_gap.py` — run BOTH clustering recipes on identical face_records.
- [x] FINDING: bridge ≡ unified steps, sub-stage for sub-stage. Clustering code is
      NOT the divergence. FC v2 faces (340) → core 133 → 10/86 (both recipes);
      Albumify faces (337) → core 186 → 20/100 (both recipes). The gap is the FACE
      SET (producer chain), not the clustering step. `LOCALIZE_GAP.html`.
- [x] RE-SCOPE: old "route clustering onto unified steps" fixes nothing behaviorally;
      it becomes cleanup. The real fix is unifying the producer chain (Stage 3).

## Stage 0c — Pin the exact producer delta  `[ ]`  (diagnostic)
- [ ] Diff the two `_v4` face tables face-by-face: per-image counts, embedding
      presence, det_score, alignment/embedding values → which producer step drives
      core 133 vs 186 (suspects: `filter_faces`, `filter_quality`, alignment).
- [ ] GATE: written-up finding naming the divergent step(s); no commit.

## Stage 1 — Cross-app equivalence test (write it RED)  `[ ]`
- [ ] `tests/architecture/test_app_cluster_equivalence.py` — run a real profile
      through BOTH app entrypoints (FC v2 `run_profile`/`run_v2_pipeline` and
      Albumify `PipelineService`) on Budapest; assert IDENTICAL cluster sizes.
- [ ] Replace the false-confidence `tests/architecture/test_config_parity.py`
      hand-matched-dict cases (or supersede with a real-profile case).
- [ ] GATE: test is RED now (8 ≠ 20) and documents the bug; commit the RED test.

## Stage 2 — Blur-scoring step in the shared producer chain  `[ ]`
- [ ] New step `insightface_score_blur` (or equiv) populating `FaceRecord.blur_score`.
- [ ] Un-pin `blur_min` in the quality gate / remove the `build_fc_config` 0.0 pin.
- [ ] `tests/...` unit test on the blur step; both apps populate blur_score.
- [ ] GATE: Albumify gated count moves toward 72; anchor for FC v2 still holds; review; commit.

## Stage 3 — Unify the PRODUCER chain (the real behavioral fix)  `[ ]`
- [ ] Reconcile the face-producing sub-chain so both apps feed clustering the same
      faces: detection thresholds, `filter_faces` / `filter_quality`, orientation +
      `align_faces`, `extract_face_embeddings`. Target: core set converges (133↔186)
      and the Stage-1 equivalence test passes for one profile on Budapest.
- [ ] Driven by Stage 0c's named divergent step(s).
- [ ] (Cleanup, may fold into Stage 5) consolidate to one clustering step-set —
      behaviorally neutral since bridge ≡ unified.
- [ ] GATE: Stage-1 equivalence GREEN; core sets match; e2e_budapest green; review; commit.
- [ ] >>> STOP: report equivalence achieved before contract reshape <<<

## Stage 4 — FCParams as the single config contract  `[ ]`
- [ ] API `PipelineRequest` / `ConfigService` carry+validate an `FCParams`
      (or profile name) → `to_step_configs()`; stop merging untyped dicts for clustering.
- [ ] Retire `sim_bench/pipeline/steps/configs/cluster_people.py::ClusterPeopleConfig`.
- [ ] Generalize `run_v2_pipeline` → a shared step-list runner both apps call
      (drop hand-rolled `_discover_jpgs`, single executor path); producer tag differs.
- [ ] GATE: API endpoint tests vs golden + contract guard + equivalence green; review; commit.

## Stage 5 — Delete the bridge (finish spec-040 Phase 7)  `[ ]`
- [ ] Delete `sim_bench/pipeline/steps/face_cluster_bridge.py` + monolithic
      `cluster_people` step; grep shows no callers.
- [ ] GATE: full suite green; REVIEW.md; flip spec → Implemented.

---

# Track B — Read/serialization layer (DEFERRED)

Original spec-079 scope (PeopleRepository → PeopleService/view → API reads through
view → slim people table → unify read contracts). Independent of Track A; revisit
after execution is unified. Preserved here so it isn't lost:

- PeopleRepository (mirror `cluster_analysis_repo.py`) + `PersonRow`
- `face_cluster/views/people.py` (`PeopleService`, `PersonView`)
- API read methods delegate to the view (people/results/faces routers + services)
- Slim `people` table to authored-only (Alembic migration) — gated on PEOPLE_TABLE_REPORT.html
- Schemas generated from view dataclasses + `test_api_uses_view_layer.py`
