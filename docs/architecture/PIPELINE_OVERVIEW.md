# Pipeline System Overview

A simple explanation of how the photo processing system works.

## The Big Picture

```
User clicks "Run" in browser
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│   Pipeline      │────▶│   Pipeline      │
│   Router        │     │   Service       │     │   Executor      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Database      │     │   Steps         │
                        │   (SQLite)      │     │   (18 steps)    │
                        └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │   Cache         │
                                                │   Handler       │
                                                └─────────────────┘
```

## Layer by Layer

### 1. Router (`sim_bench/api/routers/pipeline.py`)

**What it does:** Receives HTTP requests from the browser.

```python
# When user clicks "Run Pipeline":
POST /api/v1/pipeline/run  →  starts processing in background
GET  /api/v1/pipeline/{id} →  check progress
GET  /api/v1/pipeline/{id}/result  →  get results
```

### 2. Service (`sim_bench/api/services/pipeline_service.py`)

**What it does:** Orchestrates the work.

1. Creates a job record in database
2. Creates a `PipelineContext` (shared data bag)
3. Creates a `UniversalCacheHandler` (for caching)
4. Calls the Executor to run the steps

```python
# Key line - cache handler is created and passed to context:
cache_handler = UniversalCacheHandler(self._session)
context = PipelineContext(cache_handler=cache_handler)
```

### 3. Executor (`sim_bench/pipeline/executor.py`)

**What it does:** Runs each step in order.

```
discover_images → score_iqa → filter_quality → extract_embedding → cluster → select
```

Each step reads from `context`, does its work, writes results back to `context`.

### 4. Steps (`sim_bench/pipeline/steps/*.py`)

**What they do:** Actual image processing.

Each step can use caching by implementing 4 methods:

| Method | Purpose |
|--------|---------|
| `_get_cache_config()` | What to cache (image paths, feature type, model name) |
| `_process_uncached()` | Do the actual work for uncached items |
| `_serialize_for_cache()` | Convert result → bytes |
| `_deserialize_from_cache()` | Convert bytes → result |

The `BaseStep.process()` handles the caching logic automatically.

### 5. Cache Handler (`sim_bench/pipeline/cache_handler.py`)

**What it does:** Stores computed results so we don't recompute them.

Three simple methods:
```python
store_to_cache(key, data_bytes, metadata)  # Save
load_from_cache(keys) → dict               # Load (checks if file changed)
search_keys(filter) → list                 # Find cached items
```

**Cache Key:** `(image_path, feature_type, model_name)`

Example: `("D:/photos/img.jpg", "scene_embedding", "dinov2")`

### 6. Database (`sim_bench/api/database/models.py`)

**What it stores:**

| Table | Purpose |
|-------|---------|
| `albums` | Photo folders |
| `pipeline_runs` | Job status (pending/running/completed) |
| `pipeline_results` | Final results (clusters, selected images) |
| `universal_cache` | Cached features (embeddings, scores) |

## Data Flow Example

**First Run (no cache):**
```
User: "Process my photos"
  → Router receives POST
  → Service creates job + context + cache_handler
  → Executor runs steps:
      1. discover_images: finds 100 photos
      2. score_iqa:
         - checks cache → 0 hits
         - computes scores for 100 images (slow)
         - saves to cache
      3. extract_embedding:
         - checks cache → 0 hits
         - extracts embeddings (slow)
         - saves to cache
  → Results saved to database
```

**Second Run (cached):**
```
User: "Process same photos again"
  → Same flow, but:
      2. score_iqa:
         - checks cache → 100 hits!
         - returns cached scores instantly
      3. extract_embedding:
         - checks cache → 100 hits!
         - returns cached embeddings instantly
  → 10-100x faster!
```

## Key Files

| File | Purpose |
|------|---------|
| `api/routers/pipeline.py` | HTTP endpoints |
| `api/services/pipeline_service.py` | Job orchestration |
| `pipeline/executor.py` | Runs steps in order |
| `pipeline/base.py` | BaseStep with caching template |
| `pipeline/cache_handler.py` | Cache storage/retrieval |
| `pipeline/context.py` | Shared data between steps |
| `pipeline/serializers.py` | Convert data ↔ bytes |
| `api/database/models.py` | Database tables |

## The Caching Pattern

Steps that want caching inherit from `BaseStep` and implement:

```python
class ScoreIQAStep(BaseStep):

    def _get_cache_config(self, context, config):
        return {
            "items": [str(p) for p in context.image_paths],
            "feature_type": "iqa_scores",
            "model_name": "rule_based"
        }

    def _process_uncached(self, items, context, config):
        # Only called for items NOT in cache
        return {path: compute_score(path) for path in items}

    def _serialize_for_cache(self, result, item):
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data, item):
        return Serializers.json_deserialize(data)

    def _store_results(self, context, results, config):
        context.iqa_scores = results
```

The `BaseStep.process()` method handles:
- Loading cached items
- Calling `_process_uncached()` only for misses
- Saving new results to cache
- Merging cached + new results

## Summary

1. **Router** = HTTP interface (receives requests)
2. **Service** = Orchestrator (creates jobs, context, cache)
3. **Executor** = Step runner (runs steps in order)
4. **Steps** = Workers (process images, use cache)
5. **Cache Handler** = Storage (save/load computed features)
6. **Database** = Persistence (jobs, results, cache)
