# spec-040 Concrete Plan — strangler-fig migration

**Companion to**: `spec.md` (what & why) and `tasks.md` (phase checklist).
**Pattern**: strangler fig. Build the new FC App alongside the old; prove equivalence on the same fixture; retire the old.
**Branch**: `unification/spec-040` (active, off main `740403d`).

---

## Locked decisions (2026-05-17)

1. **Legacy location**: `app/face_clustering_legacy/` + `face_cluster_legacy/` (rename existing).
2. **New FC App location**: `app/face_clustering/` (the original path — legacy moves out first).
3. **Global DB**: both apps share `~/.sim_bench/sim_bench.db`. `action_log` gains a `producer` column (`fc_app_legacy` vs `fc_app_v2`).
4. **Equivalence bar**: ≥95% cluster-assignment agreement on a labeled fixture, run as a deterministic CI test.
5. **NO bridge / adapter / translator classes in the final architecture.** Producers write Pydantic objects directly onto context (`List[FaceRecord]`, `List[ImageRecord]`, `List[SceneClusterRecord]`). Consumers read the same Pydantic objects. The only translation is Pydantic → DataFrame → SQL row at write time, validated by Pandera. No `face_cluster_bridge`, no `assemble_face_records` translator step, no `context.insightface_faces` dict-of-dicts. spec.md ↳ "Locked architectural constraints" for the full list.

---

## What moves vs. what stays

### Moves to `face_cluster_legacy/` (FC-App-specific runner)

| Source | Destination | Why legacy |
|---|---|---|
| `face_cluster/pipeline.py::FaceClusteringPipeline` | `face_cluster_legacy/pipeline.py` | The hand-written stage runner — replaced by unified framework |
| `face_cluster/pipeline.py::_RunContext` | same | Will be replaced by unified `PipelineContext` |
| `face_cluster/config.py::PipelineConfig` (dataclass) | `face_cluster_legacy/config.py` | Replaced by per-step Pydantic models |

### Moves to `app/face_clustering_legacy/`

| Source | Destination |
|---|---|
| `app/face_clustering/main.py` and every tab / helper | `app/face_clustering_legacy/*` |
| `app/face_clustering/state.py`, `_profile_bar.py`, `run_panels.py`, etc. | same |

### Stays in `face_cluster/` (shared infrastructure — both old and new use this)

| File | Why shared |
|---|---|
| `types.py` (FaceRecord, MergeDecisionRow, ClusterResult, GraphResult, ...) | Type contracts |
| `quality.py` (QualityGater, PoseEstimator) | Algorithm |
| `knn_graph.py`, `clustering.py`, `exemplars.py`, `merge.py`, `attach.py` | Algorithms |
| `embedding.py` | Face-crop reading |
| `run_exporter.py`, `run_store.py`, `db/`, `image_detail.py` | Storage |
| `filter_context.py`, `config_diff.py`, `cluster_diameter_cap.py` | Cross-cutting |
| `ml_trainer.py`, `analysis_views.py`, `loader.py` | Analysis layer |

### Stays in `sim_bench/pipeline/` (Albumify infrastructure)

- `sim_bench/pipeline/steps/face_cluster_bridge.py` — Albumify's adapter. Lives until Phase 3 replaces it with fine-grained steps. **Not** moved to legacy because Albumify (a NEW path) uses it.

---

## Phases (revised for strangler-fig)

### Phase 0 — DONE (2026-05-16)

- ✅ FR-033-1 E2E test on main (`tests/face_clustering/test_albumify_e2e.py`).
- ✅ Branch `unification/spec-040` off main `740403d`.

### Phase 1 — Rename to _legacy (Day 1)

Mechanical rename. The legacy app must still pass its tests after this phase.

**Files renamed**

```
face_cluster/pipeline.py        → face_cluster_legacy/pipeline.py
face_cluster/config.py          → face_cluster_legacy/config.py  (the dataclass only)
app/face_clustering/            → app/face_clustering_legacy/
```

**Import updates** (mechanical — grep + replace)

```
from face_cluster import FaceClusteringPipeline, PipelineConfig
  → from face_cluster_legacy import FaceClusteringPipeline, PipelineConfig

from face_cluster.pipeline import FaceClusteringPipeline
  → from face_cluster_legacy.pipeline import FaceClusteringPipeline

from face_cluster.config import PipelineConfig as FCConfig
  → from face_cluster_legacy.config import PipelineConfig as FCConfig
```

Affected files: bridge (`sim_bench/pipeline/steps/face_cluster_bridge.py`), tests, scripts, notebooks. Grep `from face_cluster import\|face_cluster\.pipeline\|face_cluster\.config` to find them.

**`face_cluster/__init__.py`**: remove the `FaceClusteringPipeline` and `PipelineConfig` exports. Keep `FaceRecord` etc.

**Streamlit menu**: leave only legacy entry for now. New entry added in Phase 5.

**Tests**

- All legacy face-clustering tests must still pass (full sweep).
- Architecture suite green.

**Rollback**: `git revert` the rename commit.

**Done when**: `pytest tests/` returns the same green/red as pre-rename; legacy FC App launches and clusters identically.

---

### Phase 2 — Unified PipelineContext + Pydantic configs (Day 2–6)

(Same as previous plan — now Phase 2 instead of Phases 1+2.)

- Extend `sim_bench/pipeline/context.py` with the fields that `face_cluster_legacy._RunContext` carries (face_records, core_indices, cluster_result, graph_result, merge_log, etc.).
- spec-034 row per new field.
- Per-step Pydantic models for every face-clustering step. `STEP_CONFIG_MODELS` covers all of them; specs/039 registry guard activates with empty allowlist.
- Closes FR-033-6.

**Done when**: 40 architecture tests pass with the new model coverage; spec-035 E2E still green.

---

### Phase 3 — Replace bridge with named pipeline steps (Day 7–11)

**No bridge / translator steps.** Producers write Pydantic objects directly; consumers read the same objects. Eight new steps (one fewer than before — `assemble_face_records` dropped per the "no bridges" locked constraint):

| New step | Replaces | Reads | Writes |
|---|---|---|---|
| `quality_gate_faces` | inline `QualityGater.select_core_set()` | `context.face_records` | mutates `face.is_core` / `face.rejection_reason` on each `FaceRecord` |
| `build_face_knn_graph` | inline `KNNGraphBuilder.build_graph()` | `context.face_records` | `context.graph_result` |
| `cluster_face_components` | inline `ConnectedComponentsClusterer.cluster()` | `context.graph_result` | `context.cluster_result` |
| `select_face_exemplars` | inline `D10ExemplarSelector.select_exemplars()` | `context.cluster_result` | mutates `face.d10_score`; sets `cluster_result.exemplars` |
| `merge_face_clusters` | inline `ConservativeMerger.merge_clusters_with_logging()` | `context.cluster_result` | `context.merged_cluster_result`, `context.merge_log`, `context.merge_metadata` |
| `apply_diameter_cap` | inline spec-031 cap | `context.merged_cluster_result` | `context.cap_decisions`, `context.cap_summary` |
| `attach_holdout_faces` | inline `HoldoutAttacher.attach_holdouts()` | `context.cluster_result` + holdout indices | mutates `cluster_result.clusters` |
| `assign_people_clusters` | labels-loop tail of `run_face_cluster_knn` | final `cluster_result` | `context.people_clusters` |

**Also in Phase 3** — modify existing producer steps to write Pydantic directly (kills the dict-of-dicts):

| Step | Today | Phase 3 |
|---|---|---|
| `insightface_detect_faces` | writes `context.insightface_faces[path]["faces"]` (list of dicts) | constructs `FaceRecord` objects and appends to `context.face_records: List[FaceRecord]` |
| `align_faces` | mutates the dict, sets `face["aligned_face"]` | mutates `face.aligned_face` on the Pydantic object |
| `insightface_score_pose / eyes / expression` | writes nested dicts under `face["scores"]` | sets `face.pose_score`, `face.eyes_score`, `face.expression_score` directly |
| `extract_face_embeddings` | writes a separate `context.face_embeddings[path]` dict | sets `face.embedding` / `face.embedding_normalized` on the existing `FaceRecord` |
| `filter_faces` | reads / writes dict | reads / writes Pydantic attrs |

After Phase 3: `context.insightface_faces` and `context.face_embeddings` are **dead state**. They can be deprecated and removed in Phase 7. Every step from detect through clustering uses `context.face_records: List[FaceRecord]` as the single source of truth.

`configs/pipeline.yaml::default_pipeline` updates to invoke the new clustering steps in order. `cluster_people` reduces to a thin dispatcher or is removed.

**Bridge file (`sim_bench/pipeline/steps/face_cluster_bridge.py`) stays alive** for one more phase because legacy FC App (in `face_cluster_legacy/`) still imports from it. Deleted in Phase 7 alongside legacy retirement.

**Tests**: spec-035 E2E still green; unit tests per new step. New test `test_no_intermediate_face_dicts.py` asserts `context.insightface_faces` is not read by any step after Phase 3 (architecture test — fails on regression).

---

### Phase 4 — Schema v5 + blur step + scene-side persistence + global DB producer tag (Day 12–15)

**Schema v5** (in `face_cluster/db/schema.py`):
- `SCHEMA_VERSION = 5`
- New `images` table (image_path PK, image_id, n_faces, iqa, ava, sharpness, composite_score, created_at)
- New `scene_clusters` table (scene_cluster_id PK, iteration, size, method, exemplar_image_path, avg_intra_distance, created_at) — parallel to face `clusters` table
- New `scene_cluster_assignments` table (image_path FK→images, scene_cluster_id, iteration, distance_to_centroid) — parallel to face `cluster_assignments`
- New `scene_embeddings.npy` + `scene_embedding_image_paths.npy` (parallel to faces side bulk storage)
- `faces` adds `area_ratio` + `bbox_*_ratio` (all ∈ [0,1]); drops the 4 denormalized image columns (`iqa_score`, `ava_score`, `sharpness_score`, `scene_cluster_id`)
- Both legacy and new write v5. Legacy must be updated to populate `area_ratio` / `bbox_*_ratio` (small change in `RunExporter._write_faces_and_scores`). Scene-side tables populated by a new `_write_scene_clusters` / `_write_images` in `RunExporter`.

**Closes**: SIGHTING-060 (unit drift), SIGHTING-061 (blur producer via FR-033-3), SIGHTING-064 (area_ratio), SIGHTING-065 (images table), **SIGHTING-066 (scene-side structural symmetry)**.

**Global DB**: `action_log` gains `producer TEXT` column (`fc_app_legacy` | `fc_app_v2` | `albumify`). Idempotent migration in `face_cluster.run_history_db._migrate_*`.

**Blur step**: new `sim_bench/pipeline/steps/insightface_score_blur.py` writes `insightface_faces[path]["scores"]["blur_score"]`. Bridge's `blur_min=0` pin lifted (legacy still benefits because the legacy bridge reads from the same shared insightface_faces dict).

**Migration script**: `scripts/migrate_v4_to_v5.py` reads v4 DB, writes v5. Idempotent.

**Closes**: SIGHTING-060/-061/-064/-065, specs/037.

**Tests**: `test_schema_v5_migration.py`; `test_gate_has_producer.py` passes with empty waiver list (closes specs/036).

---

### Phase 5 — Build NEW FC App at `app/face_clustering/` (Day 15–21)

The strangler fig. New app lives where the old one used to be (legacy already moved out in Phase 1).

**New files**

| File | Purpose | LOC |
|---|---|---|
| `face_cluster/fc_app_runner.py` | Thin wrapper. Builds a `PipelineContext` + step list and runs `PipelineExecutor`. Same code path Albumify uses. | ≤200 |
| `app/face_clustering/main.py` | Streamlit entry. Tabs mirror legacy: Run, Recluster, Clusters Base, Merge Analysis, etc. | similar to legacy |
| `app/face_clustering/tabs/run_tab.py` | Builds per-step config dicts (per the Pydantic models from Phase 2). Calls `fc_app_runner.run()`. | smaller than legacy |
| `app/face_clustering/tabs/recluster_tab.py` | Same |
| Other tabs as needed | Same | — |

**Reuses shared infrastructure**: `face_cluster.{types, quality, knn_graph, clustering, exemplars, merge, run_exporter, run_store, db}` — all of it.

**Streamlit menu**: now has BOTH entries: "Face Clustering" (new) and "Face Clustering (legacy)" — separately runnable.

**Profile compatibility**: new app reads the same `~/.sim_bench/profiles/*.json` via `migrate_fc_profiles.py` (re-shapes flat-dataclass-shape → per-step-dict-shape at load time). Legacy app reads the original shape. Both apps write back in their native shape.

**Tests**

- `tests/face_clustering/test_fc_app_v2_e2e.py` — runs the NEW FC App through `fc_app_runner` on the 5-image fixture; same assertions as the Albumify E2E test.
- spec-035 Albumify E2E still passes.
- Legacy FC App tests still pass.

**Rollback**: delete `app/face_clustering/` (the new one) and `face_cluster/fc_app_runner.py`. Legacy untouched.

**Done when**: both apps launch from the Streamlit menu; both produce a v5 face_clustering.db; both write to action_log with their producer tag.

---

### Phase 6 — Equivalence test (Day 22–23 + ongoing)

**The acceptance gate.** Same input + same canonical config → ≥95% cluster-assignment agreement.

**New test**: `tests/face_clustering/test_legacy_vs_v2_equivalence.py`

```
def test_v2_matches_legacy_on_fixture():
    fixture = test_data/face_clustering_100/  # 100-image labeled set
    profile = test_data/profiles/canonical_profile.json
    legacy_out = tmp_path / "legacy"
    v2_out    = tmp_path / "v2"

    legacy_app.run(fixture, profile, legacy_out)
    v2_app.run(fixture, profile, v2_out)

    agreement = compute_assignment_agreement(
        RunStore(legacy_out).faces(),
        RunStore(v2_out).faces(),
    )
    assert agreement >= 0.95, (
        f"v2 disagrees with legacy on {(1-agreement)*100:.1f}% of faces. "
        f"Investigate: {find_disagreeing_pairs(...)}"
    )
```

**`compute_assignment_agreement` helper**: pairs faces by `(image_path, face_index)` across the two DBs; for each pair, asks "do these two clustering algorithms put them in the same person?" Uses Adjusted Rand Index or simple pairwise agreement.

**Runs**: every PR on the branch.

**Done when**: equivalence test green for 3 consecutive runs.

---

### Phase 7 — Sunset legacy (after equivalence holds, ≥2 weeks)

**Sunset gate**: equivalence test green every day for 2 weeks. Then:

**Files deleted**

| File | Why |
|---|---|
| `face_cluster_legacy/` (entire package) | Replaced by unified framework |
| `app/face_clustering_legacy/` (entire dir) | Replaced by new FC App |
| `sim_bench/pipeline/steps/face_cluster_bridge.py` | Bridge dead since Phase 3 |
| `face_cluster/result_db.py` if unused | Verify with grep |

**Architecture tests**

- `tests/architecture/test_no_legacy.py` — asserts the legacy packages don't exist; CI catches accidental re-introduction.
- `tests/architecture/test_no_bridge.py` — already exists per old plan.

**Closes**: SIGHTING-062 (filter_decisions dedup — only one writer now), SIGHTING-063 (bridge pose-lookup — bridge gone), FR-033-2/3/4/5/6/7 (all absorbed).

**Done when**: legacy dirs gone; new FC App is the only FC App; CI is the gate.

---

### Phase 8 — Documentation collapse (Day after Phase 7)

Same as before. All architecture HTMLs collapse the two-producer columns; `config_diff.py` simplifies; spec-033 status → "Implemented + superseded by spec-040".

---

## Critical path

```
Day  1     Phase 1 — rename to _legacy
Day  2-6   Phase 2 — context + typed configs
Day  7-11  Phase 3 — bridge → named steps (Albumify migrates)
Day  12-14 Phase 4 — schema v5 + blur step + producer tag
Day  15-21 Phase 5 — build new FC App
Day  22-23 Phase 6 — equivalence test green
Day  24+   Phase 7 wait — equivalence holds for 2 weeks
Day  ~38   Phase 7 — delete legacy + bridge
Day  39    Phase 8 — docs collapse
Day  40    Merge to main
```

**Total**: ~6 weeks. Rollback at any phase = use the legacy app. No "burn-in metric" anxiety — equivalence is a deterministic CI test.

---

## Risk register (revised)

| Risk | Mitigation under strangler |
|---|---|
| New FC App doesn't match legacy | Equivalence test catches it on every PR. Don't merge Phase 5 until ≥95%. |
| Legacy breaks during Phase 1 rename | Mechanical rename; full test sweep gates the commit. |
| Schema v5 breaks legacy writes | Legacy updated to populate `*_ratio` in same Phase 4 commit. Otherwise no schema change visible to legacy. |
| Two apps confuse users | Streamlit menu labels both clearly. Legacy carries a deprecation banner from Phase 5 onward. |
| Equivalence < 95% on a fixture | Don't merge. Investigate. Possible causes: algorithm drift, config translation bug, non-determinism in the new path. |
| Legacy never retires (zombie code) | Sunset gate is calendar-based + CI green. After 2 weeks equivalence green, sunset is automatic — file an issue if it slips. |

---

## What's better about this plan vs. the previous one

| Concern | Previous plan | This plan |
|---|---|---|
| Acceptance | Run on labeled album; compare to pre-`pre-spec-040` tag; chase numbers | Deterministic CI test; ≥95% or red |
| Rollback | `git reset` to a tag | Use legacy app — always runnable |
| Bus factor | One engineer holding the migration | Anyone can run the equivalence test |
| Sunset signal | "2-week burn-in" — fuzzy | Equivalence green daily for 2 weeks — auditable |
| Bridge fate | Deleted in Phase 6 mid-migration | Deleted only after legacy retires (Phase 7) |

---

## Open questions (4 → 2)

Decisions you've made (logged above): legacy location, new-app location, DB sharing, equivalence bar.

Remaining:

1. **Which "canonical profile"** for the equivalence test? Suggest: the FC App's default profile saved as `test_data/profiles/canonical_profile.json`. Locked at the start of Phase 6 and not changed during the burn-in.
2. **What metric for "cluster-assignment agreement"** — Adjusted Rand Index, pairwise agreement, or per-face same-cluster-as-baseline? Suggest: pairwise agreement (simplest, interpretable; ARI is too forgiving).

Want me to lock these now, or wait?
