# Caching Architecture

## Overview

The caching system prevents recomputation of expensive features (embeddings, quality scores, face detections) when processing the same images multiple times.

## Design Principles

1. **Opaque bytes storage** - Cache handler stores raw bytes, steps handle serialization
2. **Template method pattern** - BaseStep provides caching logic, steps implement specifics
3. **mtime-based invalidation** - Cache entries invalidated when source file changes

## Cache Key

Each cached item is identified by three components:

```
(image_path, feature_type, model_name)
```

Examples:
- `("D:/photos/img.jpg", "scene_embedding", "dinov2")`
- `("D:/photos/img.jpg", "iqa_scores", "rule_based")`
- `("D:/photos/img.jpg", "face_detection", "mediapipe")`

## Components

### 1. UniversalCacheHandler (`sim_bench/pipeline/cache_handler.py`)

Simple 3-method interface for cache storage:

```python
class UniversalCacheHandler:
    def store_to_cache(self, key: CacheKey, data: bytes, metadata: dict) -> None
    def load_from_cache(self, keys: List[CacheKey]) -> Dict[str, Tuple[bytes, dict]]
    def search_keys(self, filter: dict) -> List[CacheKey]
```

- Stores opaque bytes (steps handle serialization)
- Validates mtime on load (auto-invalidates stale entries)
- Uses SQLite `universal_cache` table

### 2. CacheKey (`sim_bench/pipeline/cache_handler.py`)

Immutable dataclass identifying a cached feature:

```python
@dataclass(frozen=True)
class CacheKey:
    image_path: str
    feature_type: str
    model_name: str
```

### 3. Serializers (`sim_bench/pipeline/serializers.py`)

Helper class for common serialization patterns:

```python
class Serializers:
    @staticmethod
    def json_serialize(value) -> bytes      # For scores, metadata
    def json_deserialize(data) -> Any

    @staticmethod
    def numpy_serialize(array) -> bytes     # For embeddings
    def numpy_deserialize(data) -> ndarray

    @staticmethod
    def pickle_serialize(obj) -> bytes      # For complex objects
    def pickle_deserialize(data) -> Any
```

### 4. BaseStep Template (`sim_bench/pipeline/base.py`)

Steps inherit from `BaseStep` and implement caching hooks:

```python
class MyStep(BaseStep):
    def _get_cache_config(self, context, config) -> dict:
        """Return cache configuration or None to skip caching."""
        return {
            "items": [...],           # List of items to process
            "feature_type": "...",    # Cache category
            "model_name": "...",      # Model identifier
            "metadata": {}            # Optional metadata
        }

    def _process_uncached(self, items, context, config) -> dict:
        """Process only uncached items. Return {item: result}."""
        ...

    def _serialize_for_cache(self, result, item) -> bytes:
        """Convert result to bytes for storage."""
        ...

    def _deserialize_from_cache(self, data, item) -> Any:
        """Convert cached bytes back to result."""
        ...

    def _store_results(self, context, results, config) -> None:
        """Store all results (cached + new) in context."""
        ...
```

The `BaseStep.process()` method handles the caching flow automatically.

## Database Schema

```sql
CREATE TABLE universal_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Cache key (unique together)
    image_path TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,

    -- Opaque data storage
    data_blob BLOB NOT NULL,

    -- Invalidation metadata
    image_mtime REAL NOT NULL,
    created_at DATETIME,
    last_accessed DATETIME,

    UNIQUE(image_path, feature_type, model_name)
);
```

## Caching Flow

```
BaseStep.process()
    │
    ├─► _get_cache_config()
    │       │
    │       └─► Returns {items, feature_type, model_name}
    │
    ├─► cache_handler.load_from_cache(keys)
    │       │
    │       ├─► Check mtime for each entry
    │       ├─► Delete stale entries
    │       └─► Return valid cached data
    │
    ├─► _deserialize_from_cache() for each hit
    │
    ├─► _process_uncached() for misses only
    │
    ├─► _serialize_for_cache() for new results
    │
    ├─► cache_handler.store_to_cache() for new results
    │
    └─► _store_results() with all results
```

## Example: IQA Scoring Step

```python
@register_step
class ScoreIQAStep(BaseStep):

    def _get_cache_config(self, context, config):
        return {
            "items": [str(p) for p in context.image_paths],
            "feature_type": "iqa_scores",
            "model_name": "rule_based",
            "metadata": {}
        }

    def _process_uncached(self, items, context, config):
        model = self._get_model()
        results = {}
        for path in items:
            scores = model.get_detailed_scores(path)
            results[path] = {
                "iqa": scores["overall"],
                "sharpness": scores["sharpness_normalized"]
            }
        return results

    def _serialize_for_cache(self, result, item):
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data, item):
        return Serializers.json_deserialize(data)

    def _store_results(self, context, results, config):
        context.iqa_scores = {p: r["iqa"] for p, r in results.items()}
        context.sharpness_scores = {p: r["sharpness"] for p, r in results.items()}
```

## Performance

| Scenario | Time |
|----------|------|
| First run (100 images) | ~225 seconds |
| Second run (100 images, all cached) | ~2-5 seconds |
| Adding 10 new images | ~25 seconds (90 cached, 10 computed) |

**Speedup: 10-100x for repeated runs**

## Cache Invalidation

Cache entries are automatically invalidated when:

1. **File modified** - mtime differs from cached mtime
2. **File deleted** - file not found during load

Manual invalidation:
```python
# Clear all cache for an image
cache_handler.search_keys({"image_path": "/path/to/img.jpg"})
# Then delete entries

# Clear all cache for a feature type
cache_handler.search_keys({"feature_type": "scene_embedding"})
```

## Integration

Cache handler is injected into pipeline context by `PipelineService`:

```python
# In PipelineService.start_pipeline():
cache_handler = UniversalCacheHandler(self._session)
context = PipelineContext(cache_handler=cache_handler)
```

Steps access it via `context.cache_handler` (handled automatically by `BaseStep`).
