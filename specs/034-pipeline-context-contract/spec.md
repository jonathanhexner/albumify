# spec-034 — Pipeline Context Contract

**Status**: In Progress
**Date**: 2026-05-15
**Triggered by**: spec-033 P-B (master plan: `specs/033-data-integrity/MASTER_PLAN.md`)
**Related**: spec-012 (StepDecision), spec-030 (Storage Ownership), spec-032 (FilterContext)

## 1. What this spec is

The two pipeline frameworks (Albumify `sim_bench.pipeline.context.PipelineContext` and Face Clustering App `face_cluster.pipeline._RunContext`) carry **shared mutable state** between steps. Today there is no document declaring: who produces each field, who consumes it, what its unit is, whether it is persisted to disk/DB, when it is valid.

This spec is that document. It is checked into the repo and CI-enforced: a new field in either dataclass with no spec entry fails `tests/architecture/test_pipeline_context_contract.py`.

The intent is bidirectional drift protection — code-and-spec stay aligned, or CI breaks.

## 2. Conventions

- **Producer step**: the canonical step that writes this field. Multiple writers are flagged in the Notes column.
- **Consumer step(s)**: every downstream step that *reads* this field. `(many)` is allowed for fields read by most steps (e.g. `image_paths`).
- **Unit**:
  - `path` — `pathlib.Path` or `str` path. Absolute, forward-slashes per CLAUDE.md.
  - `float[0,1]` — dimensionless score normalized to `[0, 1]`.
  - `px` — pixels (absolute).
  - `frac` — fraction of image dimension `[0, 1]`.
  - `frac²` — fraction-of-image-area.
  - `cos` — cosine distance / similarity (range depends on field).
  - `°` — degrees.
  - `id` — integer cluster/face id.
  - `n/a` — categorical / no unit (e.g. a boolean, a dict-of-dicts, a registry object).
- **Persisted**: `yes (table.column)` if written to DB / disk, `ephemeral` otherwise. `ephemeral` is the *deliberate* answer when the field is reconstructable or only valid mid-run.
- **Lifecycle**: when the field becomes valid → when it is consumed for the last time. Used by debuggers reading partial pipeline state.

## 3. `PipelineContext` (Albumify — `sim_bench/pipeline/context.py`)

### 3.1 Inputs & discovery

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `source_directory` | `Path` | path | pipeline runner | (many) | yes (`run_metadata.source_path`) | full run | input only, never mutated |
| `image_paths` | `list[Path]` | path | `discover_images` | (many) | indirectly (`faces.image_path` rows) | full run | every face/score row keys off these |

### 3.2 Image-level scores

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `iqa_scores` | `dict[str, float]` | float[0,1] | `score_iqa` | `filter_quality`, `select_best`, `cluster_by_identity` | NOT YET — spec-033 P-C adds `faces.iqa_score` | from `score_iqa` → end | currently dropped at export (SIGHTING-059) |
| `ava_scores` | `dict[str, float]` | float[0,1] | `score_ava` | `select_best`, `filter_quality` | NOT YET — spec-033 P-C adds `faces.ava_score` | from `score_ava` → end | currently dropped at export |
| `sharpness_scores` | `dict[str, float]` | float[0,1] | (legacy MediaPipe path) | `filter_quality` | NOT YET — spec-033 P-C adds `faces.sharpness_score` | computed once, used at filter time | NULL on InsightFace pipeline runs today |
| `method_scores` | `dict[str, dict[str, float]]` | float (higher=better) | `score_quality` (spec-093) | Image Analysis Studio (spec-094) | `universal_cache` per (image, `quality_<method>`) | from `score_quality` → end | path → {method: score}; in-run hand-off, cache is the store |
| `geo_metadata` | `dict[str, GeoMetadata]` | lat/lon/time | `extract_geo_metadata` (spec-022) | `geo_temporal_segment`, studio geo view | `universal_cache` (`geo_exif`) | producer → end | absent fields None (graceful) |
| `geo_segments` | `list` | — | `geo_temporal_segment` | trip detection | not persisted | run-scoped | winning-axis segments (empty if FLAT) |
| `geo_home` | `Optional[tuple]` | (lat, lon) | `geo_temporal_segment` | trip detection | not persisted | run-scoped | auto-detected home anchor or None |
| `geo_clip_predictions` | `dict[str, list]` | label+score | `infer_geo_clip` (StreetCLIP) | studio geo view | `universal_cache` (`geo_streetclip`) | producer → end | top-k city guesses |
| `geo_coord_predictions` | `dict[str, list]` | lat/lon+prob | `infer_geo_coords` (GeoCLIP) | studio geo view | `universal_cache` (`geo_geoclip`) | producer → end | top-k coord guesses |
| `image_captions` | `dict[str, str]` | text | `caption_images` (BLIP) | studio geo view | `universal_cache` (`blip_caption`) | producer → end | scene caption per image |

### 3.3 Face-specific (MediaPipe legacy path)

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `face_records` | `list[FaceRecord]` | n/a (typed object) | spec-040 producer steps (insightface_detect_faces, align_faces, score_*, extract_face_embeddings, filter_faces) | spec-040 clustering chain (quality_gate_faces → assign_people_clusters); RunExporter at write time | yes (`faces` + `face_scores` tables) | full run from detect → end | spec-040 Phase 3 canonical face state; replaces the dict-based `faces` / `insightface_faces` / `face_embeddings` fields below as those are phased out |
| `faces` | `dict[str, list]` | n/a | `detect_faces` (MediaPipe) | `score_face_*`, `cluster_by_identity` | indirectly via `insightface_faces` on the active pipeline | full run | LEGACY — InsightFace pipeline uses `insightface_faces`; spec-040 Phase 7 deletes |
| `face_pose_scores` | `dict[str, list[float]]` | float[0,1] | `score_face_pose` | `select_best` | ephemeral (legacy) | scoring → end | legacy path only |
| `face_eyes_scores` | `dict[str, list[float]]` | float[0,1] | `score_face_eyes` | `select_best` | ephemeral (legacy) | scoring → end | legacy path only |
| `face_smile_scores` | `dict[str, list[float]]` | float[0,1] | `score_face_smile` | `select_best` | ephemeral (legacy) | scoring → end | legacy path only |
| `is_face_dominant` | `dict[str, bool]` | bool | `score_face_pose` (legacy) | `select_best` | ephemeral | scoring → end | legacy path only |

### 3.4 InsightFace pipeline data

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `persons` | `dict[str, dict]` | n/a | `detect_persons` (YOLOv8) | `insightface_detect_faces` (for face↔person association) | ephemeral | detect → end | bbox in px |
| `insightface_faces` | `dict[str, dict]` | mixed (px + landmarks + scores) | `insightface_detect_faces` | (many — `filter_faces`, `align_faces`, `score_face_*`, `cluster_people`) | indirectly (`faces` table after `cluster_people`) | detect → end | the canonical "list of faces per image" for the active pipeline |

### 3.5 Embeddings

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `scene_embeddings` | `dict[str, np.ndarray]` | n/a (vector) | `extract_scene_embedding` | `cluster_scenes` | yes (`embeddings` table, `kind='scene'`) | extract → cluster | DINOv2 / CLIP feature vectors |
| `face_embeddings` | `dict[str, list[np.ndarray]]` | n/a (vector) | `extract_face_embeddings` | `cluster_people`, `cluster_by_identity` | yes (`embeddings` table, `kind='face'`) | extract → cluster | ArcFace 512-d vectors |

### 3.6 Filtering results

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `quality_passed` | `set[str]` | path | `filter_quality` | (advisory only) | superseded by `filters` (spec-032) | filter → end | LEGACY advisory — to be removed in spec-032 P7 |
| `portrait_passed` | `set[str]` | path | (legacy `filter_portraits`) | advisory only | ephemeral | filter → end | legacy MediaPipe path only |
| `active_images` | `set[str]` | path | (sometimes set by tests / by spec-032 migration) | `get_active_image_paths()` | ephemeral | varies | grandfathered — see spec-032 |
| `filters` | `FilterContext` | n/a (typed object) | every filter step (`filter_quality`, `filter_faces`, etc.) | (many — query via `filters.active(item_type)`) | yes (`filter_decisions` table) | full run | **canonical** filter state — spec-032 |
| `step_decisions` | `list[StepDecision]` | n/a | every step that emits decisions | API serialization | yes (DB) | full run | spec-012 — coexists with `filters` |

### 3.7 Scene + identity clustering

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `scene_clusters` | `dict[int, list[str]]` | id → paths | `cluster_scenes` | `cluster_by_identity`, `select_best` | NOT YET — spec-033 P-C adds `faces.scene_cluster_id` | cluster → end | image-level scene assignment |
| `scene_cluster_labels` | `dict[str, int]` | path → id | `cluster_scenes` | `cluster_by_identity`, `select_best` | NOT YET — spec-033 P-C adds `faces.scene_cluster_id` | cluster → end | inverse view of `scene_clusters` |
| `face_clusters` | `dict[int, dict[int, list[str]]]` | scene_id → cluster_id → paths | `cluster_by_identity` | `select_best` | yes (`cluster_assignments` table) | cluster → end | per-scene identity subclusters |

### 3.8 Global face clustering (People feature)

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `all_faces` | `list` | n/a (FaceForClustering) | `cluster_people` (assembly stage) | `cluster_people` (clustering stage) | yes (`faces` table — after `cluster_people` writes it) | within `cluster_people` | the canonical list passed into `face_cluster_bridge` |
| `all_face_embeddings` | `np.ndarray` | n/a (matrix) | `cluster_people` | `cluster_people` (internal) | yes (`embeddings` table) | within `cluster_people` | normalized in-place |
| `people_clusters` | `dict[int, list]` | cluster_id → faces | `cluster_people` | `identity_refinement`, API | yes (`cluster_assignments` table) | cluster → end | global identity clusters |
| `people_thumbnails` | `dict[int, Any]` | id → PIL.Image | `cluster_people` (post-process) | API | indirectly (`crops/`) | post-cluster → end | display only |
| `people_best_images` | `dict[int, dict]` | id → metadata | `cluster_people` (post-process) | API | ephemeral | post-cluster → end | "best face per cluster" UI helper |

### 3.9 Identity refinement

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `refined_people_clusters` | `dict[int, list]` | id → faces | `identity_refinement` | `cluster_by_identity`, API | yes (`cluster_assignments` table) | refinement → end | replaces `people_clusters` if step ran |
| `unassigned_faces` | `list` | n/a | `identity_refinement` | API | ephemeral | refinement → end | for "noise" review UI |
| `cluster_exemplars` | `dict[int, list]` | id → faces | `identity_refinement` | `cluster_by_identity` | indirectly (`face_scores.is_exemplar`) | refinement → end | exemplar face per cluster |
| `cluster_centroids` | `dict[int, np.ndarray]` | id → embedding | `identity_refinement` | `cluster_by_identity` | ephemeral | refinement → end | recomputed per run |
| `attachment_decisions` | `dict[str, dict]` | path → decision | `identity_refinement` | API | yes (`filter_decisions` via spec-032) | refinement → end | per-face attach outcome |

### 3.10 Output

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `fc_export_dir` | `Optional[str]` | path | `cluster_people` (if `export_for_analysis=True`) | `face_cluster_export` | yes (filesystem path) | cluster → end | FC App artifact dir |
| `user_overrides` | `list` | n/a | pipeline runner (DB pre-load) | `identity_refinement` | yes (`user_overrides` table) | full run | user corrections from prior runs |
| `composite_scores` | `dict[str, float]` | float[0,1] | `select_best` | API | NOT YET — spec-033 P-C adds `faces.composite_score` (image-level via face row) | select → end | final score |
| `quality_scores` | `dict[str, float]` | float[0,1] | `select_best` | API (Results table) | yes (`image_metrics.quality_score`) | select → end | spec-084: quality half of composite (`composite = quality + penalty`) |
| `person_penalties` | `dict[str, float]` | float (≤0) | `select_best` | API (Results table) | yes (`image_metrics.person_penalty`) | select → end | spec-084: penalty half of composite |
| `siamese_comparisons` | `list[dict]` | n/a | `select_best` (siamese refinement) | API (debug) | ephemeral | select → end | debug log |
| `selected_images` | `list[str]` | path | `select_best` | API | yes (`run_metadata.selected_images` JSON) | select → end | final selection |

### 3.11 Plumbing

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `on_progress` | `Callable` | n/a | pipeline runner | (many — via `report_progress`) | ephemeral | full run | UI progress callback |
| `step_configs` | `dict[str, dict]` | n/a | pipeline runner | (every step) | yes (`pipeline_run.json:config`) | full run | spec-033 P-G replaces with typed Pydantic models |
| `cache_handler` | `Optional[UniversalCacheHandler]` | n/a | pipeline runner | every step with cacheable output | n/a (registry object) | full run | universal feature cache wrapper |

## 4. `_RunContext` (Face Clustering App — `face_cluster/pipeline.py`)

### 4.1 Config + plumbing

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `config` | `PipelineConfig` | n/a | pipeline runner | (every stage) | yes (`pipeline_run.json`) | full run | spec-033 P-G migrates to Pydantic |
| `image_dir` | `Path` | path | pipeline runner | discover stage | yes (`run_metadata.source_path`) | full run | input |
| `output_dir` | `Path` | path | pipeline runner | export stage | yes (the run dir itself) | full run | run target |
| `on_progress` | `ProgressCallback` | n/a | pipeline runner | (many) | ephemeral | full run | UI callback |
| `run_record` | `Dict` | n/a | pipeline runner | written by `write_run_record()` | yes (`pipeline_run.json`) | full run | run metadata accumulator |
| `file_handler` | `logging.FileHandler` | n/a | pipeline runner | logging | n/a | full run | log file handle |
| `log_path` | `Path` | path | pipeline runner | logging, export | indirectly (log file on disk) | full run | log file location |
| `action_id` | `Optional[int]` | id | run_history_db | export | yes (`run_history_db`) | full run | row id for this run |

### 4.2 Discovery + faces

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `image_paths` | `list[Path]` | path | discover stage | embed, quality, crops | indirectly (FaceRecord rows) | discover → end | source images |
| `faces` | `list[FaceRecord]` | n/a | embed stage | (every later stage) | yes (`faces`, `face_scores`, `embeddings` tables) | embed → end | spec-033 P-C C-2 makes this Pydantic |
| `core_indices` | `list[int]` | id (into `faces`) | quality stage | cluster, exemplars, merge | indirectly (`face_scores.in_core_set`) | quality → end | indices that passed quality gating |
| `holdout_indices` | `list[int]` | id (into `faces`) | quality stage | attach | indirectly (inverse of in_core_set) | quality → end | indices that failed quality gating |
| `crop_manifest` | `Dict` | n/a | crops stage | export | yes (`crop_manifest.json`) | crops → end | face_id → crop path mapping |

### 4.3 Clustering outputs

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `cluster_result` | `Optional[ClusterResult]` | n/a | cluster stage | exemplars, merge, export | yes (`clusters`, `cluster_assignments`) | cluster → end | post-cluster snapshot |
| `graph_result` | `object` | n/a | cluster stage | merge, attach | ephemeral | cluster → end | mutual kNN graph + distances |
| `merged_cluster_result` | `Optional[ClusterResult]` | n/a | merge stage | export, cap | yes (`clusters` after merge) | merge → end | post-merge snapshot |
| `merge_log` | `Optional[List[Dict]]` | n/a | merge stage | export, FC App merge tab | yes (`merge_log.json` + `merge_decisions` table) | merge → end | per-pair gate results |
| `merge_metadata` | `Optional[Dict]` | n/a | merge stage | export | yes (`merge_metadata.json`) | merge → end | run-level merge statistics |
| `cap_decisions` | `Optional[List[Dict]]` | n/a | diameter_cap stage (spec-031) | export | yes (`cap_decisions.json`) | cap → end | which merges were reverted |
| `cap_summary` | `Optional[Dict]` | n/a | diameter_cap stage | export | yes (`cap_summary.json`) | cap → end | per-run cap statistics |

### 4.4 Filter context (spec-032)

| Field | Type | Unit | Producer | Consumer(s) | Persisted | Lifecycle | Notes |
|---|---|---|---|---|---|---|---|
| `filters` | `FilterContext` | n/a | quality stage, crops stage | downstream stages, export | yes (`filter_decisions` table) | full run | shared with Albumify — same class |

## 5. Drift-prevention rule

A new field on either dataclass that lacks a row in this spec FAILS `tests/architecture/test_pipeline_context_contract.py`. Likewise, a spec row that names a field absent from the dataclass FAILS.

This is the bidirectional drift guard. Fields can graduate (`NOT YET → yes (table.column)`) as their persistence lands; fields can deprecate (`legacy …` notes) but cannot vanish silently.

## 6. Locked decisions

1. **Per-field rows** — every field gets its own row. No collapsing groups, even when many fields share producer/consumer. The verbosity buys debuggability.
2. **NOT YET vs ephemeral** — `NOT YET` is the deliberate marker for fields that *should* be persisted but currently aren't (a load-bearing distinction for spec-033 P-C). `ephemeral` is the deliberate "this field is mid-run only" answer.
3. **`(many)` is allowed** but only when the field is read by ≥4 steps OR by all-downstream-of-X. Otherwise enumerate.
4. **One table per logical grouping** — readers scan headers, not full tables.
5. **No code in the spec** — type names only. Field bodies live in the source.

## 7. References

- spec-012 — StepDecision (the per-item decision record, coexists with `filters`)
- spec-030 — Storage Ownership (RunExporter/RunStore, schema columns this spec references)
- spec-032 — FilterContext (the canonical filter-decision primitive on `filters`)
- spec-033 — Data Integrity master plan (this spec is its P-B)
- `sim_bench/pipeline/context.py` — PipelineContext source
- `face_cluster/pipeline.py` — _RunContext source
