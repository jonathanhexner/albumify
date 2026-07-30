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

**NO-HARM GUARDS — green after EVERY stage (see spec.md §No-harm test strategy):**
- **Guard A**: `tests/face_clustering/e2e_budapest/` (`profile_4` → 15 clusters / 340 faces). Movement = harm.
- **Guard B**: `tests/pipeline/test_pipeline_spec.py` + `test_albumify_default_spec.py` (spec stays valid).
- **Guard C**: FC v2 `profile_5` anchor stays 8 (fixes change *Albumify*, not FC v2).
- **Hard rule**: do NOT commit `tests/_fixtures/budapest_golden/*.json` (WIP=13) until the equivalence target is GREEN at 8==8.

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

## Stage 0c — Pin the exact producer delta  `[x]` DONE  (diagnostic)
- [x] `scripts/diff_face_sets.py` — per-image core counts + gate rejection reasons.
- [x] FINDING: same images/faces; the delta is the **POSE gate**. FC v2 rejects
      53 off-angle faces (`pose_yaw` 46 + `pose_pitch` 7); Albumify rejects 0 because
      `FaceRecord.pose is None` (its producer never populates pose). 53 = 186−133 =
      the entire core gap → 20 vs 8. It's a missing face ATTRIBUTE, not a different set.

> **2026-06-20 CORRECTION — this diagnosis is WRONG.** Re-verified empirically:
> `insightface_detect_faces` DOES populate `FaceRecord.pose` (lines 108/194/212),
> for BOTH apps (same step). Pose is `None` only because the Budapest
> `insightface_detection` cache rows predate spec-070 (dated 2026-02-19, no `pose`
> key) and `cache_handler.load_from_cache` never invalidates on schema/model_version
> change — see **SIGHTING-099**. Both apps read the same stale cache, so pose cannot
> explain FC v2=8 vs Albumify=24. The real divergence is multi-confound (stale
> cache + config + HEIC discovery + Albumify-only `identity_refinement` redefining
> the count) — see **SIGHTING-100**. Stages 2 ("blur step") and 3 ("populate pose
> in the producer") are therefore moot as written. Number-matching is deferred to
> SIGHTING-100; the architecture unification below proceeds independently.

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

## Stage 3 — Populate pose in Albumify's FaceRecord (the behavioral fix)  `[ ]`
- [ ] Plumb pose (yaw/pitch/roll) into `FaceRecord.pose` on the Albumify path —
      either from `insightface_detect_faces` (as FC v2 gets it) or by wiring
      `insightface_score_pose`'s output into the record + persisting it.
- [ ] Verify: Albumify core set drops 186 → 133, identities 20 → 8 on Budapest+profile_5.
- [ ] Named by Stage 0c: pose gate rejects 53 off-angle faces FC v2 already drops.
- [ ] (Cleanup, may fold into Stage 5) consolidate to one clustering step-set —
      behaviorally neutral since bridge ≡ unified.
- [ ] GATE: Stage-1 equivalence GREEN; core sets match; e2e_budapest green; review; commit.
- [ ] >>> STOP: report equivalence achieved before contract reshape <<<

## Stage 4 — FCParams as the single, HIERARCHICAL config contract  `[ ]`
- [ ] Make `FCParams` hierarchical: `gating` / `graph` / `exemplars` / `split` /
      `merge` sub-models, composed of the existing `steps/configs/*.py` models
      (decision: ends the two-config-systems split — see spec.md §Design decisions).
- [ ] `to_step_configs()` PROJECTS each sub-model to its owning step (was: broadcast
      one flat blob to all 8). Enables `extra="forbid"` per step → typos caught.
- [ ] NO-HARM TEST `tests/.../test_fcparams_projection.py`: for the current default
      profile, the new projected `step_configs` is **byte-equal** to the pre-refactor
      flat broadcast; every step knob is reachable; no orphan knobs; round-trips.
- [ ] API `PipelineRequest` / `ConfigService` carry+validate an `FCParams`
      (or profile name) → `to_step_configs()`; stop merging untyped dicts for clustering.
- [ ] Retire `sim_bench/pipeline/steps/configs/cluster_people.py::ClusterPeopleConfig`.
- [ ] Generalize `run_v2_pipeline` → a shared step-list runner both apps call
      (drop hand-rolled `_discover_jpgs`, single executor path); producer tag differs.
- [ ] GATE: byte-equal projection test + API endpoint tests vs golden + Guards A/B/C
      + equivalence green; review; commit.

## Stage 5 — Delete the bridge (finish spec-040 Phase 7)  `[ ]`
- [ ] Delete `sim_bench/pipeline/steps/face_cluster_bridge.py` + monolithic
      `cluster_people` step; `grep` shows no callers; Albumify `default_pipeline`
      in `configs/pipeline.yaml` → the 8 unified clustering steps.
- [ ] GATE: full suite green; equivalence target still 8==8; Guard A unchanged;
      REVIEW.md; flip spec → Implemented.

## Refactor R — restore empty `__init__.py` convention  `[ ]`  (SIGHTING, anytime/parallel)
- [ ] `steps/configs/__init__.py` currently holds 16 re-export imports +
      `STEP_CONFIG_MODELS` + `__all__` (~3.7 KB) — violates CLAUDE.md "empty
      `__init__.py`". Move the registry + imports to `steps/configs/registry.py`;
      leave `__init__.py` empty; repoint the architecture test + introspection callers.
- [ ] File the sighting in `docs/project/SIGHTINGS.md` first (isolated refactor).
- [ ] GATE: import smoke + full suite green; zero behavior change.

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
