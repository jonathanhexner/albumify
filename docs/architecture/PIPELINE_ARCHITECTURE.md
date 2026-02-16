# Pipeline Architecture

## Overview

The sim-bench pipeline is a modular image processing system that analyzes photos, detects faces, clusters similar images, and selects the best ones from each cluster.

## Key Files

| File | Role |
|------|------|
| `sim_bench/pipeline/context.py` | **PipelineContext** - Shared state container (dataclass) |
| `sim_bench/pipeline/base.py` | BaseStep, StepMetadata, PipelineStep protocol |
| `sim_bench/pipeline/registry.py` | StepRegistry singleton, @register_step decorator |
| `sim_bench/pipeline/executor.py` | PipelineExecutor - orchestrates step execution |
| `sim_bench/pipeline/builder.py` | PipelineBuilder - dependency resolution, topological sort |
| `sim_bench/pipeline/cache_handler.py` | UniversalCacheHandler - mtime-based feature caching |
| `sim_bench/pipeline/steps/all_steps.py` | Imports all steps to populate registry |

---

## Component Diagram

```
                    ┌─────────────────────────────┐
                    │     API / Streamlit         │
                    │   (user triggers pipeline)  │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PipelineExecutor                              │
│  • execute(context, steps) - runs steps sequentially                │
│  • Uses PipelineBuilder for dependency ordering                     │
│  • Returns PipelineResult with step timings and errors              │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐
│   StepRegistry    │  │  PipelineBuilder  │  │   PipelineContext     │
│   (singleton)     │  │                   │  │   (shared state)      │
│                   │  │  • build()        │  │                       │
│  • get(name)      │  │  • topo_sort()    │  │  source_directory     │
│  • list_steps()   │  │                   │  │  image_paths          │
│  • has_step()     │  │  Resolves         │  │  iqa_scores           │
│                   │  │  depends_on →     │  │  ava_scores           │
│  @register_step   │  │  execution order  │  │  faces                │
│  auto-registers   │  │                   │  │  embeddings           │
│  at import time   │  │                   │  │  clusters             │
│                   │  │                   │  │  selected_images      │
└───────────────────┘  └───────────────────┘  │                       │
                                              │  cache_handler ──────►│ SQLite
                                              │  on_progress ────────►│ Callback
                                              └───────────────────────┘
                                                        ▲
                                                        │
                    ┌───────────────────────────────────┤
                    │                                   │
         ┌──────────┴─────────┐           ┌────────────┴────────────┐
         │      Step 1        │           │        Step N           │
         │   process(ctx)     │    ...    │     process(ctx)        │
         │   reads/writes     │           │     reads/writes        │
         │   context fields   │           │     context fields      │
         └────────────────────┘           └─────────────────────────┘
```

---

## PipelineContext (Shared State)

**File: `sim_bench/pipeline/context.py`**

The `PipelineContext` is a Python dataclass that acts as a shared mutable container. Every step receives the same context object and reads/writes its fields.

```python
@dataclass
class PipelineContext:
    # Input
    source_directory: Path = None

    # Discovery
    image_paths: list[Path] = field(default_factory=list)

    # Analysis scores (keyed by image path string)
    iqa_scores: dict[str, float] = field(default_factory=dict)
    ava_scores: dict[str, float] = field(default_factory=dict)
    sharpness_scores: dict[str, float] = field(default_factory=dict)

    # Face-specific (keyed by image path string)
    faces: dict[str, list] = field(default_factory=dict)
    face_pose_scores: dict[str, list[float]] = field(default_factory=dict)
    face_eyes_scores: dict[str, list[float]] = field(default_factory=dict)
    face_smile_scores: dict[str, list[float]] = field(default_factory=dict)

    # Embeddings
    scene_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    face_embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict)

    # Clustering
    scene_clusters: dict[int, list[str]] = field(default_factory=dict)
    face_clusters: dict[int, dict[int, list[str]]] = field(default_factory=dict)

    # Selection
    composite_scores: dict[str, float] = field(default_factory=dict)
    selected_images: list[str] = field(default_factory=list)

    # Infrastructure
    cache_handler: Optional[UniversalCacheHandler] = None
    on_progress: Callable[[str, float, str], None] = None
```

---

## Data Flow Through Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              discover_images                                 │
│                                                                             │
│  Input: source_directory                                                    │
│  Output: image_paths                                                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│    score_iqa     │     │    score_ava     │     │    detect_persons     │
│                  │     │                  │     │     (InsightFace)     │
│ reads: paths     │     │ reads: paths     │     │                       │
│ writes:          │     │ writes:          │     │ reads: paths          │
│ iqa_scores       │     │ ava_scores       │     │ writes: persons       │
└────────┬─────────┘     └────────┬─────────┘     └───────────┬───────────┘
         │                        │                           │
         ▼                        ▼                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            PipelineContext                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  image_paths: [Path(...), Path(...), ...]                            │  │
│  │  iqa_scores: {"path1": 0.85, "path2": 0.72, ...}                     │  │
│  │  ava_scores: {"path1": 7.2, "path2": 6.8, ...}                       │  │
│  │  persons: {"path1": {...}, ...}  ← InsightFace data                  │  │
│  │  insightface_faces: {"path1": {...}, ...}  ← InsightFace faces       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│insightface_detect    │  │insightface_score     │  │insightface_score     │
│_faces                │  │_expression           │  │_eyes / _pose         │
│                      │  │                      │  │                      │
│writes:               │  │reads:                │  │reads:                │
│insightface_faces     │  │insightface_faces     │  │insightface_faces     │
│                      │  │writes: expression    │  │writes: eyes/pose     │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │
           ▼                         ▼                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      extract_face_embeddings                                │
│                                                                             │
│  reads: insightface_faces OR faces                                         │
│  crops face regions from original images                                   │
│  writes: face_embeddings                                                   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      extract_scene_embedding                                │
│                                                                             │
│  reads: image_paths                                                        │
│  uses DINOv2 to extract scene-level embeddings                             │
│  writes: scene_embeddings                                                  │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          cluster_scenes                                     │
│                                                                             │
│  reads: scene_embeddings                                                   │
│  uses HDBSCAN to cluster similar scenes                                    │
│  writes: scene_clusters, scene_cluster_labels                              │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            select_best                                      │
│                                                                             │
│  reads: scene_clusters, face_clusters, scores, embeddings                  │
│  selects best images per cluster using smart branching logic               │
│  writes: selected_images, composite_scores                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## select_best.py Flowchart

The `select_best` step uses branching logic to select the best images from each cluster.

```
                            ┌─────────────────────────────────┐
                            │          select_best            │
                            │         process(ctx)            │
                            └───────────────┬─────────────────┘
                                            │
                                            ▼
                            ┌─────────────────────────────────┐
                            │   use_face_subclusters &&       │
                            │   context.face_clusters exists? │
                            └───────────────┬─────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │ YES                                     NO    │
                    ▼                                               ▼
        ┌───────────────────────┐                   ┌───────────────────────┐
        │ Iterate face_clusters │                   │ Iterate scene_clusters│
        │ (scene → subclusters) │                   │ (cluster_id → images) │
        └───────────┬───────────┘                   └───────────┬───────────┘
                    │                                           │
                    └─────────────────┬─────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────┐
                    │         For each cluster/subcluster:        │
                    │         _select_from_cluster()              │
                    └───────────────────┬─────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │            has_faces?                        │
                    └───────────────────┬─────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │ YES                             NO    │
                    ▼                                       ▼
        ┌───────────────────────────┐       ┌───────────────────────────┐
        │  _score_face_cluster()    │       │  _score_non_face_cluster()│
        │                           │       │                           │
        │  composite =              │       │  score = 0.7 * AVA        │
        │    w_eyes * eyes_avg +    │       │        + 0.3 * IQA        │
        │    w_pose * pose_avg +    │       │                           │
        │    w_smile * smile_avg +  │       └─────────────┬─────────────┘
        │    w_ava * ava_score      │                     │
        └─────────────┬─────────────┘                     │
                      │                                   │
                      └───────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────┐
                    │         Sort by score descending            │
                    │         scored_images.sort(reverse=True)    │
                    └───────────────────┬─────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │   siamese_model && scores close?            │
                    │   (within tiebreaker_threshold)             │
                    └───────────────────┬─────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │ YES                             NO    │
                    ▼                                       │
        ┌───────────────────────────┐                       │
        │ _apply_siamese_tiebreaker │                       │
        │                           │                       │
        │ Compare top candidates    │                       │
        │ with Siamese CNN to       │                       │
        │ determine true winner     │                       │
        │                           │                       │
        │ Rerank if needed          │                       │
        └─────────────┬─────────────┘                       │
                      │                                     │
                      └───────────────┬─────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────┐
                    │              SELECTION LOGIC                │
                    └───────────────────┬─────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │  Take #1 if score >= min_score_threshold    │
                    │  or only 1 image in cluster                 │
                    └───────────────────┬─────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │  Consider #2 if max_per_cluster > 1         │
                    └───────────────────┬─────────────────────────┘
                                        │
                                        ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                    SMART RULES FOR #2                                │
        │                                                                      │
        │  ┌─────────────────────────────────────────────────────────────┐    │
        │  │  Rule 1: Score above min_score_threshold?                   │    │
        │  │          NO → reject                                         │    │
        │  └─────────────────────────────────────────────────────────────┘    │
        │                              │                                       │
        │                              ▼                                       │
        │  ┌─────────────────────────────────────────────────────────────┐    │
        │  │  Rule 2: Score gap <= max_score_gap?                        │    │
        │  │          gap = (best_score - second_score) / best_score     │    │
        │  │          NO → reject                                         │    │
        │  └─────────────────────────────────────────────────────────────┘    │
        │                              │                                       │
        │                              ▼                                       │
        │  ┌─────────────────────────────────────────────────────────────┐    │
        │  │  Rule 3: Not a near-duplicate?                              │    │
        │  │          _check_near_duplicate() uses embedding similarity  │    │
        │  │          similarity > duplicate_threshold → reject          │    │
        │  └─────────────────────────────────────────────────────────────┘    │
        │                                                                      │
        └──────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │  All rules passed?                          │
                    │  YES → add to selected                      │
                    │  NO  → skip #2                              │
                    └───────────────────┬─────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │  context.selected_images = selected         │
                    │  context.composite_scores updated           │
                    └─────────────────────────────────────────────┘
```

### Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_images_per_cluster` | 2 | Maximum images to select per cluster |
| `min_score_threshold` | 0.4 | Minimum score to keep an image |
| `max_score_gap` | 0.25 | Maximum gap between #1 and #2 scores |
| `duplicate_similarity_threshold` | 0.85 | Embedding similarity threshold for duplicates |
| `tiebreaker_threshold` | 0.05 | Use Siamese CNN if scores within this range |
| `face_weights.eyes_open` | 0.30 | Weight for eyes-open score |
| `face_weights.pose` | 0.30 | Weight for head pose score |
| `face_weights.smile` | 0.20 | Weight for smile score |
| `face_weights.ava` | 0.20 | Weight for AVA aesthetic score |

### Note on Duplicate Detection

The step uses **embedding cosine similarity** for duplicate detection, NOT Siamese CNN. This is because:
- Siamese CNN compares image **quality** (which is better)
- Embedding similarity compares image **content** (are they the same scene)

Low Siamese confidence means "can't tell which is better quality", not "they are duplicates".

---

## Step Registration

Steps are registered via the `@register_step` decorator:

```python
# In sim_bench/pipeline/steps/select_best.py
from sim_bench.pipeline.registry import register_step

@register_step
class SelectBestStep(BaseStep):
    def __init__(self):
        self._metadata = StepMetadata(
            name="select_best",
            display_name="Select Best Images",
            requires={"scene_clusters"},
            produces={"selected_images"},
            depends_on=["cluster_scenes"],
            ...
        )
```

When `sim_bench/pipeline/steps/all_steps.py` is imported:
1. All step classes are imported
2. `@register_step` decorator calls `registry.register(step_class)`
3. Registry instantiates step and stores by name

---

## Caching (UniversalCacheHandler)

Steps can opt into automatic caching by implementing template methods:

```python
class MyStep(BaseStep):
    def _get_cache_config(self, context, config):
        return {
            "items": ["image1.jpg", "image2.jpg"],
            "feature_type": "my_feature",
            "model_name": "my_model",
        }

    def _process_uncached(self, items, context, config):
        # Only process items not in cache
        return {item: compute_feature(item) for item in items}

    def _serialize_for_cache(self, result, item):
        return Serializers.numpy_serialize(result)

    def _deserialize_from_cache(self, data, item):
        return Serializers.numpy_deserialize(data)
```

Cache invalidation is **mtime-based**: if the image file is modified, cached features are discarded.
