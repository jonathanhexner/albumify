# Class Diagram - Album Organization Pipeline

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SYSTEM OVERVIEW                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐                       │
│   │   NiceGUI   │ ──────▶ │   FastAPI   │ ──────▶ │  Pipeline   │                       │
│   │  Frontend   │  HTTP   │   Backend   │         │   Engine    │                       │
│   └─────────────┘   WS    └─────────────┘         └─────────────┘                       │
│                                  │                       │                               │
│                                  ▼                       ▼                               │
│                           ┌─────────────┐         ┌─────────────┐                       │
│                           │   SQLite    │         │   ML Models │                       │
│                           │  Database   │         │  (ModelHub) │                       │
│                           └─────────────┘         └─────────────┘                       │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Engine Classes

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  PIPELINE ENGINE                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐
│      <<Protocol>>               │
│      PipelineStep               │
├─────────────────────────────────┤
│ + metadata: StepMetadata        │
├─────────────────────────────────┤
│ + process(context, config)      │
│ + validate(context): list[str]  │
└─────────────────────────────────┘
              △
              │ implements
    ┌─────────┴─────────┬─────────────────┬─────────────────┬─────────────────┐
    │                   │                 │                 │                 │
┌───┴───────────┐ ┌─────┴─────────┐ ┌─────┴─────────┐ ┌─────┴─────────┐ ┌─────┴─────────┐
│ DiscoverStep  │ │ ScoreIQAStep  │ │ DetectFaces   │ │ ClusterScene  │ │ ClusterPeople │
│               │ │               │ │ Step          │ │ Step          │ │ Step          │
├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
│ _extensions   │ │ _iqa_model    │ │ _crop_service │ │ _clusterer    │ │ _embedder     │
├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
│ process()     │ │ process()     │ │ process()     │ │ process()     │ │ process()     │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                        ... (18 total steps)


┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│         StepMetadata            │       │        PipelineContext          │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ + name: str                     │       │ + source_directory: Path        │
│ + display_name: str             │       │ + image_paths: list[Path]       │
│ + description: str              │       │ + iqa_scores: dict[str, float]  │
│ + category: str                 │       │ + ava_scores: dict[str, float]  │
│ + requires: set[str]            │       │ + faces: dict[str, list]        │
│ + produces: set[str]            │       │ + face_pose_scores: dict        │
│ + depends_on: list[str]         │       │ + face_eyes_scores: dict        │
│ + config_schema: dict           │       │ + scene_embeddings: dict        │
└─────────────────────────────────┘       │ + face_embeddings: dict         │
                                          │ + scene_clusters: dict          │
                                          │ + people_clusters: dict         │
┌─────────────────────────────────┐       │ + selected_images: list[str]    │
│         StepRegistry            │       │ + all_faces: list[FaceInstance] │
├─────────────────────────────────┤       │ + people_thumbnails: dict       │
│ - _steps: dict[str, Step]       │       │ + on_progress: Callable         │
├─────────────────────────────────┤       │ + step_configs: dict            │
│ + register(step)                │       │ + cache_handler: CacheHandler   │  ◄── Injected by PipelineService
│ + get(name): PipelineStep       │       └─────────────────────────────────┘
│ + get(name): PipelineStep       │
│ + list_steps(): list[StepMeta]  │
│ + find_by_category(cat): list   │       ┌─────────────────────────────────┐
└─────────────────────────────────┘       │       PipelineBuilder           │
                                          ├─────────────────────────────────┤
                                          │ - _registry: StepRegistry       │
┌─────────────────────────────────┐       │ - _dependencies: dict           │
│       PipelineExecutor          │       ├─────────────────────────────────┤
├─────────────────────────────────┤       │ + build(step_names): list[Step] │
│ - _registry: StepRegistry       │       │ + resolve_deps(name): list[str] │
│ - _builder: PipelineBuilder     │       │ + validate_order(steps): bool   │
├─────────────────────────────────┤       └─────────────────────────────────┘
│ + execute(context, steps, cfg)  │
│ + execute_step(ctx, step, cfg)  │
│ - _notify_progress(step, pct)   │
└─────────────────────────────────┘


┌─────────────────────────────────┐
│       PipelineConfig            │
├─────────────────────────────────┤
│ + fail_fast: bool = True        │
│ + step_configs: dict            │
│ + progress_callback: Callable   │
└─────────────────────────────────┘
```

---

## Data Structures

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  DATA STRUCTURES                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       <<dataclass>>             │       │       <<dataclass>>             │
│        CroppedFace              │       │        FaceInstance             │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ + image: PIL.Image              │       │ + face_id: str                  │
│ + original_path: str            │       │ + image_path: str               │
│ + bbox: dict                    │       │ + face_index: int               │
│ + bbox_pixels: dict             │       │ + cropped_face: CroppedFace     │
│ + padding: float                │       │ + pose_score: float             │
│ + face_ratio: float             │       │ + eyes_open_score: float        │
│ + detection_confidence: float   │       │ + smile_score: float            │
└─────────────────────────────────┘       │ + sharpness_score: float        │
              △                           │ + overall_score: float          │
              │ contains                  │ + person_id: int = -1           │
              │                           └─────────────────────────────────┘
┌─────────────┴───────────────────┐                     △
│       <<dataclass>>             │                     │ contains
│        PoseEstimate             │                     │
├─────────────────────────────────┤       ┌─────────────┴───────────────────┐
│ + yaw: float                    │       │       <<dataclass>>             │
│ + pitch: float                  │       │          Person                 │
│ + roll: float                   │       ├─────────────────────────────────┤
│ + frontal_score: float          │       │ + person_id: int                │
└─────────────────────────────────┘       │ + face_instances: list          │
                                          │ + thumbnail: FaceInstance       │
                                          │ + image_count: int              │
┌─────────────────────────────────┐       │ + name: str (optional)          │
│       <<dataclass>>             │       └─────────────────────────────────┘
│       ImageMetrics              │
├─────────────────────────────────┤
│ + image_path: str               │       ┌─────────────────────────────────┐
│ + iqa_score: float              │       │       <<dataclass>>             │
│ + ava_score: float              │       │      WorkflowResult             │
│ + sharpness: float              │       ├─────────────────────────────────┤
│ + has_face: bool                │       │ + source_directory: Path        │
│ + num_faces: int                │       │ + total_images: int             │
│ + is_face_dominant: bool        │       │ + filtered_images: int          │
│ + scene_embedding: ndarray      │       │ + scene_clusters: dict          │
│ + cluster_id: int               │       │ + face_clusters: dict           │
├─────────────────────────────────┤       │ + people: list[Person]          │
│ + get_composite_score(): float  │       │ + selected_images: list[str]    │
└─────────────────────────────────┘       │ + metrics: dict                 │
                                          │ + telemetry: dict               │
                                          └─────────────────────────────────┘
```

---

## FastAPI Layer

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FASTAPI LAYER                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     FastAPI App                                          │
│                                     (main.py)                                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  Middleware: CORS, Authentication (future)                                              │
│  Startup: init_db(), register_steps()                                                   │
│  Routers: albums, pipeline, steps, results, people, websocket                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│      AlbumsRouter         │ │     PipelineRouter        │ │      StepsRouter          │
│   /api/v1/albums          │ │   /api/v1/pipeline        │ │   /api/v1/steps           │
├───────────────────────────┤ ├───────────────────────────┤ ├───────────────────────────┤
│                           │ │                           │ │                           │
│ POST   /                  │ │ POST   /run               │ │ GET    /                  │
│   → create_album()        │ │   → run_pipeline()        │ │   → list_steps()          │
│   body: {name, path}      │ │   body: {album_id,        │ │   query: ?category=       │
│   returns: Album          │ │          steps?, config?} │ │   returns: list[StepInfo] │
│                           │ │   returns: {job_id}       │ │                           │
│ GET    /                  │ │                           │ │ GET    /{name}            │
│   → list_albums()         │ │ GET    /{job_id}          │ │   → get_step()            │
│   returns: list[Album]    │ │   → get_status()          │ │   returns: StepInfo       │
│                           │ │   returns: PipelineStatus │ │                           │
│ GET    /{id}              │ │                           │ │ GET    /{name}/schema     │
│   → get_album()           │ │ DELETE /{job_id}          │ │   → get_config_schema()   │
│   returns: Album          │ │   → cancel_pipeline()     │ │   returns: JSONSchema     │
│                           │ │   returns: {ok: true}     │ │                           │
│ DELETE /{id}              │ │                           │ │                           │
│   → delete_album()        │ │                           │ │                           │
│   returns: {ok: true}     │ │                           │ │                           │
│                           │ │                           │ │                           │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘

┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│     ResultsRouter         │ │      PeopleRouter         │ │    WebSocketRouter        │
│   /api/v1/results         │ │   /api/v1/people          │ │   /ws                     │
├───────────────────────────┤ ├───────────────────────────┤ ├───────────────────────────┤
│                           │ │                           │ │                           │
│ GET    /{job_id}          │ │ GET    /{album_id}        │ │ WS /progress/{job_id}     │
│   → get_result()          │ │   → list_people()         │ │   → pipeline_progress()   │
│   returns: PipelineResult │ │   returns: list[Person]   │ │                           │
│                           │ │                           │ │ Messages (server→client): │
│ GET    /{job_id}/images   │ │ GET    /{album_id}/{pid}  │ │  {type: "progress",       │
│   → get_images()          │ │   → get_person()          │ │   step: str,              │
│   query: ?cluster_id=     │ │   returns: Person         │ │   progress: float,        │
│   returns: list[ImageInfo]│ │                           │ │   message: str}           │
│                           │ │ GET    /{aid}/{pid}/imgs  │ │                           │
│ POST   /{job_id}/export   │ │   → get_person_images()   │ │  {type: "step_complete",  │
│   → export_result()       │ │   returns: list[ImageInfo]│ │   step: str,              │
│   body: {path, format}    │ │                           │ │   duration_ms: int}       │
│   returns: {ok: true}     │ │ PATCH  /{aid}/{pid}       │ │                           │
│                           │ │   → rename_person()       │ │  {type: "complete",       │
│                           │ │   body: {name: "Mom"}     │ │   result_id: str}         │
│                           │ │   returns: Person         │ │                           │
│                           │ │                           │ │  {type: "error",          │
│                           │ │ POST   /{aid}/merge       │ │   step: str,              │
│                           │ │   → merge_people()        │ │   message: str}           │
│                           │ │   body: {person_ids: []}  │ │                           │
│                           │ │   returns: Person         │ │                           │
│                           │ │                           │ │                           │
│                           │ │ POST   /{aid}/split       │ │                           │
│                           │ │   → split_person()        │ │                           │
│                           │ │   body: {pid, face_ids}   │ │                           │
│                           │ │   returns: list[Person]   │ │                           │
│                           │ │                           │ │                           │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```

---

## Services Layer

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SERVICES LAYER                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│        AlbumService             │       │       PipelineService           │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ - _session: Session             │       │ - _session: Session             │
│ - _logger: Logger               │       │ - _registry: StepRegistry       │
├─────────────────────────────────┤       │ - _executor: PipelineExecutor   │
│ + create(name, path): Album     │       │ - _cache: FeatureCacheService   │
│ + get(id): Album                │       │ - _jobs: dict[str, Job]         │
│ + list(): list[Album]           │       │ - _logger: Logger               │
│ + delete(id): bool              │       ├─────────────────────────────────┤
│ + validate_path(path): bool     │       │ + start(album_id, steps, cfg)   │
└─────────────────────────────────┘       │   → str (job_id)                │
                                          │ + get_status(job_id): Status    │
                                          │ + cancel(job_id): bool          │
┌─────────────────────────────────┐       │ + subscribe(job_id): AsyncGen   │
│        ResultService            │       └─────────────────────────────────┘
├─────────────────────────────────┤
│ - _session: Session             │
├─────────────────────────────────┤       ┌─────────────────────────────────┐
│ + get(job_id): PipelineResult   │       │        PeopleService            │
│ + get_images(job_id, cluster)   │       ├─────────────────────────────────┤
│   → list[ImageInfo]             │       │ - _session: Session             │
│ + export(job_id, path, format)  │       ├─────────────────────────────────┤
│   → bool                        │       │ + list(album_id): list[Person]  │
│ + save(run_id, result): str     │       │ + get(album_id, pid): Person    │
└─────────────────────────────────┘       │ + get_images(aid, pid): list    │
                                          │ + rename(aid, pid, name): Person│
                                          │ + merge(aid, pids): Person      │
                                          │ + split(aid, pid, fids): list   │
                                          └─────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               CACHING SYSTEM                                             │
│                                                                                          │
│   ✅ COMPLETE: Template method pattern with opaque byte storage                         │
│   - UniversalCache table stores opaque bytes (steps handle serialization)               │
│   - UniversalCacheHandler provides 3 simple methods                                     │
│   - BaseStep template method handles cache flow automatically                           │
│   - Expected speedup: 10-100x for repeated runs                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       <<dataclass>>             │       │     UniversalCacheHandler       │
│         CacheKey                │       │  (pipeline/cache_handler.py)    │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ + image_path: str               │       │ - _session: Session             │
│ + feature_type: str             │       ├─────────────────────────────────┤
│ + model_name: str               │       │ + store_to_cache(key, data,     │
├─────────────────────────────────┤       │     metadata) → None            │
│ + to_string() → str             │       │ + load_from_cache(keys)         │
└─────────────────────────────────┘       │     → dict[str, (bytes, dict)]  │
                                          │ + search_keys(filter)           │
                                          │     → list[CacheKey]            │
┌─────────────────────────────────┐       └─────────────────────────────────┘
│         Serializers             │
│   (pipeline/serializers.py)     │
├─────────────────────────────────┤
│ + json_serialize(value) → bytes │       ┌─────────────────────────────────┐
│ + json_deserialize(data) → Any  │       │         BaseStep                │
│ + numpy_serialize(arr) → bytes  │       │     (pipeline/base.py)          │
│ + numpy_deserialize(data) → arr │       ├─────────────────────────────────┤
│ + pickle_serialize(obj) → bytes │       │ # Template method (in process): │
│ + pickle_deserialize(data) → obj│       │ 1. _get_cache_config()          │
└─────────────────────────────────┘       │ 2. load_from_cache()            │
                                          │ 3. _deserialize_from_cache()    │
                                          │ 4. _process_uncached()          │
                                          │ 5. _serialize_for_cache()       │
                                          │ 6. store_to_cache()             │
                                          │ 7. _store_results()             │
                                          └─────────────────────────────────┘

                              CACHE INTEGRATION
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│   PipelineService.start_pipeline(album_id, steps, ...):                                 │
│       cache_handler = UniversalCacheHandler(self._session)                              │
│       context = PipelineContext(cache_handler=cache_handler)                            │
│                                                                                          │
│   BaseStep.process(context, config):                                                    │
│       cache_config = self._get_cache_config(context, config)                            │
│       cached = context.cache_handler.load_from_cache(keys)                              │
│       uncached_items = [item for item in items if item not in cached]                   │
│       new_results = self._process_uncached(uncached_items, context, config)             │
│       for item, result in new_results.items():                                          │
│           data = self._serialize_for_cache(result, item)                                │
│           context.cache_handler.store_to_cache(key, data, metadata)                     │
│       self._store_results(context, all_results, config)                                 │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Models (SQLAlchemy)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  DATABASE MODELS                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│           Album                 │       │        PipelineRun              │
│         (albums)                │       │      (pipeline_runs)            │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ PK id: str (UUID)               │       │ PK id: str (UUID)               │
│    name: str                    │◄──────│ FK album_id: str                │
│    source_path: str             │   1:N │    pipeline_name: str           │
│    image_count: int             │       │    steps: JSON                  │
│    created_at: datetime         │       │    step_configs: JSON           │
└─────────────────────────────────┘       │    fail_fast: bool              │
              │                           │    status: str                  │
              │ 1:N                       │    current_step: str            │
              ▼                           │    progress: float              │
┌─────────────────────────────────┐       │    error_message: str           │
│           Person                │       │    created_at: datetime         │
│         (people)                │       │    started_at: datetime         │
├─────────────────────────────────┤       │    completed_at: datetime       │
│ PK id: str (UUID)               │       └─────────────────────────────────┘
│ FK album_id: str                │                     │
│ FK run_id: str                  │                     │ 1:1
│    person_index: int            │                     ▼
│    name: str (nullable)         │       ┌─────────────────────────────────┐
│    thumbnail_image_path: str    │       │       PipelineResult            │
│    thumbnail_face_index: int    │       │     (pipeline_results)          │
│    thumbnail_bbox: JSON         │       ├─────────────────────────────────┤
│    face_count: int              │       │ PK id: str (UUID)               │
│    image_count: int             │       │ FK run_id: str                  │
│    face_instances: JSON         │       │    total_images: int            │
│    created_at: datetime         │       │    filtered_images: int         │
└─────────────────────────────────┘       │    num_scene_clusters: int      │
                                          │    num_face_clusters: int       │
                                          │    num_people: int              │
                                          │    num_selected: int            │
                                          │    scene_clusters: JSON         │
                                          │    face_clusters: JSON          │
                                          │    selected_images: JSON        │
                                          │    image_metrics: JSON          │
                                          │    people_summary: JSON         │
                                          │    step_timings: JSON           │
                                          │    total_duration_ms: int       │
                                          └─────────────────────────────────┘


                              RELATIONSHIPS
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Album ──1:N──▶ PipelineRun ──1:1──▶ PipelineResult                       │
│     │                                                                       │
│     └────1:N──▶ Person                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Universal Cache (Database)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   UNIVERSAL CACHE                                        │
│                                                                                          │
│   Stores opaque bytes - steps handle their own serialization using Serializers class    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

Cache key: (image_path, feature_type, model_name) - unique together

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           UniversalCache (universal_cache)                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ PK  id: int (auto)                                                                       │
│                                                                                          │
│ --- Cache Key (unique together) ---                                                      │
│     image_path: str              # Full path to image file                               │
│     feature_type: str            # 'scene_embedding', 'iqa_scores', 'face_detection'... │
│     model_name: str              # 'dinov2', 'rule_based', 'mediapipe', 'arcface'...    │
│     model_version: str?          # Optional version for cache invalidation              │
│                                                                                          │
│ --- Cached Value (opaque bytes - step handles serialization) ---                        │
│     data_blob: blob              # Opaque bytes (JSON, numpy, or pickle)                │
│                                                                                          │
│ --- Metadata ---                                                                         │
│     image_mtime: float           # File modification time (for invalidation)            │
│     created_at: datetime         # When cache entry was created                         │
│     last_accessed: datetime      # Last access time                                      │
│                                                                                          │
│ UNIQUE(image_path, feature_type, model_name)                                            │
│ INDEX(image_path, feature_type, model_name)                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘


                    CONCEPTUAL CACHE STRUCTURE (per image)
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│   Image: "D:/photos/vacation/IMG_001.jpg" (mtime: 1706540400.0)                         │
│       │                                                                                  │
│       ├── (scene_embedding, dinov2)    → bytes (numpy serialized)                       │
│       ├── (scene_embedding, openclip)  → bytes (numpy serialized)                       │
│       ├── (iqa_scores, rule_based)     → bytes (JSON: {iqa: 0.85, sharpness: 0.92})    │
│       ├── (face_detection, mediapipe)  → bytes (JSON: [{bbox, confidence}, ...])       │
│       └── (face_embedding, arcface)    → bytes (numpy serialized)                       │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘


                          FEATURE TYPES REFERENCE
┌──────────────────┬─────────────────────────┬──────────────────────────────────────────────┐
│ feature_type     │ model_name examples     │ serialization                                │
├──────────────────┼─────────────────────────┼──────────────────────────────────────────────┤
│ scene_embedding  │ dinov2, openclip,       │ Serializers.numpy_serialize()                │
│                  │ resnet50                │                                              │
├──────────────────┼─────────────────────────┼──────────────────────────────────────────────┤
│ iqa_scores       │ rule_based              │ Serializers.json_serialize()                 │
├──────────────────┼─────────────────────────┼──────────────────────────────────────────────┤
│ face_detection   │ mediapipe, retinaface   │ Serializers.json_serialize()                 │
├──────────────────┼─────────────────────────┼──────────────────────────────────────────────┤
│ face_embedding   │ arcface, facenet        │ Serializers.numpy_serialize()                │
├──────────────────┼─────────────────────────┼──────────────────────────────────────────────┤
│ face_pose        │ sixdrepnet              │ Serializers.json_serialize()                 │
├──────────────────┼─────────────────────────┼──────────────────────────────────────────────┤
│ face_eyes        │ mediapipe               │ Serializers.json_serialize()                 │
├──────────────────┼─────────────────────────┼──────────────────────────────────────────────┤
│ face_smile       │ mediapipe               │ Serializers.json_serialize()                 │
└──────────────────┴─────────────────────────┴──────────────────────────────────────────────┘
```

---

## API Schemas (Pydantic)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   API SCHEMAS                                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘

REQUEST SCHEMAS                              RESPONSE SCHEMAS
─────────────────                            ────────────────

┌─────────────────────────┐                 ┌─────────────────────────┐
│     AlbumCreate         │                 │      AlbumResponse      │
├─────────────────────────┤                 ├─────────────────────────┤
│ + name: str             │                 │ + id: str               │
│ + source_path: str      │                 │ + name: str             │
└─────────────────────────┘                 │ + source_path: str      │
                                            │ + image_count: int      │
┌─────────────────────────┐                 │ + created_at: datetime  │
│    PipelineRequest      │                 └─────────────────────────┘
├─────────────────────────┤
│ + album_id: str         │                 ┌─────────────────────────┐
│ + steps: list[str]?     │                 │    PipelineStatus       │
│ + config: dict?         │                 ├─────────────────────────┤
└─────────────────────────┘                 │ + job_id: str           │
                                            │ + status: str           │
┌─────────────────────────┐                 │ + current_step: str?    │
│     ExportRequest       │                 │ + progress: float       │
├─────────────────────────┤                 │ + message: str?         │
│ + output_path: str      │                 └─────────────────────────┘
│ + format: str           │
│ + organize_by_cluster:  │                 ┌─────────────────────────┐
│   bool = True           │                 │   PipelineResultResp    │
└─────────────────────────┘                 ├─────────────────────────┤
                                            │ + job_id: str           │
┌─────────────────────────┐                 │ + album_id: str         │
│    PersonRename         │                 │ + total_images: int     │
├─────────────────────────┤                 │ + filtered_images: int  │
│ + name: str             │                 │ + scene_clusters: dict  │
└─────────────────────────┘                 │ + people: list[Person]  │
                                            │ + selected_images: list │
┌─────────────────────────┐                 │ + telemetry: dict       │
│     PeopleMerge         │                 └─────────────────────────┘
├─────────────────────────┤
│ + person_ids: list[int] │                 ┌─────────────────────────┐
└─────────────────────────┘                 │      StepInfo           │
                                            ├─────────────────────────┤
┌─────────────────────────┐                 │ + name: str             │
│     PeopleSplit         │                 │ + display_name: str     │
├─────────────────────────┤                 │ + description: str      │
│ + person_id: int        │                 │ + category: str         │
│ + face_ids: list[str]   │                 │ + requires: list[str]   │
└─────────────────────────┘                 │ + produces: list[str]   │
                                            │ + depends_on: list[str] │
                                            │ + config_schema: dict   │
                                            └─────────────────────────┘

                                            ┌─────────────────────────┐
                                            │     PersonResponse      │
                                            ├─────────────────────────┤
                                            │ + person_id: int        │
                                            │ + name: str?            │
                                            │ + face_count: int       │
                                            │ + image_count: int      │
                                            │ + thumbnail_url: str    │
                                            │ + face_instances: list  │
                                            └─────────────────────────┘

                                            ┌─────────────────────────┐
                                            │      ImageInfo          │
                                            ├─────────────────────────┤
                                            │ + path: str             │
                                            │ + thumbnail_url: str    │
                                            │ + iqa_score: float      │
                                            │ + ava_score: float      │
                                            │ + cluster_id: int       │
                                            │ + faces: list[FaceInfo] │
                                            │ + is_selected: bool     │
                                            └─────────────────────────┘
```

---

## NiceGUI Frontend

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  NICEGUI FRONTEND                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐
│          APIClient              │       PAGES
├─────────────────────────────────┤       ─────
│ - _base_url: str                │
│ - _ws: WebSocket                │       ┌─────────────────────────┐
├─────────────────────────────────┤       │       HomePage          │
│ + create_album(name, path)      │       ├─────────────────────────┤
│ + list_albums(): list[Album]    │       │ - _client: APIClient    │
│ + get_steps(): list[StepInfo]   │       ├─────────────────────────┤
│ + run_pipeline(album_id, ...)   │       │ + render()              │
│ + get_status(job_id): Status    │       │   - Album list          │
│ + get_result(job_id): Result    │       │   - Upload button       │
│ + list_people(album_id): list   │       │   - Quick actions       │
│ + connect_ws(job_id): AsyncGen  │       └─────────────────────────┘
└─────────────────────────────────┘
                                          ┌─────────────────────────┐
                                          │     ConfigurePage       │
┌─────────────────────────────────┐       ├─────────────────────────┤
│         AppState                │       │ - _album: Album         │
├─────────────────────────────────┤       │ - _steps: list[Step]    │
│ + current_album: Album?         │       ├─────────────────────────┤
│ + current_job: str?             │       │ + render()              │
│ + pipeline_status: Status?      │       │   - Pipeline selector   │
│ + result: Result?               │       │   - Step checkboxes     │
│ + selected_steps: list[str]     │       │   - Config panels       │
│ + step_configs: dict            │       │   - Run button          │
└─────────────────────────────────┘       └─────────────────────────┘

                                          ┌─────────────────────────┐
COMPONENTS                                │     ProgressPage        │
──────────                                ├─────────────────────────┤
                                          │ - _job_id: str          │
┌─────────────────────────┐               │ - _ws: WebSocket        │
│     UploadComponent     │               ├─────────────────────────┤
├─────────────────────────┤               │ + render()              │
│ + on_upload: Callable   │               │   - Progress bar        │
│ + render()              │               │   - Step timeline       │
└─────────────────────────┘               │   - Live logs           │
                                          └─────────────────────────┘
┌─────────────────────────┐
│   PipelineBuilder       │               ┌─────────────────────────┐
├─────────────────────────┤               │     ResultsPage         │
│ + available_steps: list │               ├─────────────────────────┤
│ + selected: list[str]   │               │ - _result: Result       │
│ + on_change: Callable   │               ├─────────────────────────┤
│ + render()              │               │ + render()              │
└─────────────────────────┘               │   - Tabs:               │
                                          │     - Gallery           │
┌─────────────────────────┐               │     - Clusters          │
│    StepConfigPanel      │               │     - People            │
├─────────────────────────┤               │     - Metrics           │
│ + step: StepInfo        │               │     - Export            │
│ + config: dict          │               └─────────────────────────┘
│ + on_change: Callable   │
│ + render()              │               ┌─────────────────────────┐
└─────────────────────────┘               │      PeoplePage         │
                                          ├─────────────────────────┤
┌─────────────────────────┐               │ - _people: list[Person] │
│    ImageGallery         │               ├─────────────────────────┤
├─────────────────────────┤               │ + render()              │
│ + images: list[Image]   │               │   - Person grid         │
│ + on_select: Callable   │               │   - Rename dialog       │
│ + show_metrics: bool    │               │   - Merge/Split actions │
│ + render()              │               │   - Face details        │
└─────────────────────────┘               └─────────────────────────┘

┌─────────────────────────┐
│    ClusterView          │
├─────────────────────────┤
│ + clusters: dict        │
│ + on_select: Callable   │
│ + render()              │
└─────────────────────────┘

┌─────────────────────────┐
│     FaceGrid            │
├─────────────────────────┤
│ + faces: list[Face]     │
│ + show_scores: bool     │
│ + on_select: Callable   │
│ + render()              │
└─────────────────────────┘
```

---

## Full System Interaction

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              FULL SYSTEM INTERACTION                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

User clicks "Run Pipeline"
           │
           ▼
┌─────────────────────┐     POST /pipeline/run      ┌─────────────────────┐
│   NiceGUI Frontend  │ ──────────────────────────▶ │   PipelineRouter    │
│   (ConfigurePage)   │     {album_id, steps}       │                     │
└─────────────────────┘                             └──────────┬──────────┘
           │                                                   │
           │ WS /progress/{job_id}                             ▼
           │                                        ┌─────────────────────┐
           │                                        │   PipelineService   │
           │                                        │   .start()          │
           │                                        └──────────┬──────────┘
           │                                                   │
           │                                                   ▼
           │                                        ┌─────────────────────┐
           │                                        │  BackgroundTask:    │
           │                                        │  PipelineExecutor   │
           │                                        │  .execute()         │
           │                                        └──────────┬──────────┘
           │                                                   │
           │                                    ┌──────────────┼──────────────┐
           │                                    ▼              ▼              ▼
           │                              ┌──────────┐  ┌──────────┐  ┌──────────┐
           │                              │ Step 1   │  │ Step 2   │  │ Step N   │
           │                              │ process()│  │ process()│  │ process()│
           │                              └────┬─────┘  └────┬─────┘  └────┬─────┘
           │                                   │             │             │
           │         {"type": "progress"}      │             │             │
           │◀──────────────────────────────────┴─────────────┴─────────────┘
           │
           ▼
┌─────────────────────┐     {"type": "complete"}    ┌─────────────────────┐
│   NiceGUI Frontend  │ ◀─────────────────────────  │   ResultService     │
│   (ProgressPage)    │                             │   .save()           │
└─────────────────────┘                             └─────────────────────┘
           │
           │ GET /results/{job_id}
           ▼
┌─────────────────────┐                             ┌─────────────────────┐
│   NiceGUI Frontend  │ ◀─────────────────────────  │   SQLite Database   │
│   (ResultsPage)     │     PipelineResult          │                     │
└─────────────────────┘                             └─────────────────────┘
```
