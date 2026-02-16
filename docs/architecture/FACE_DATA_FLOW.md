# Face Detection Data Flow Architecture

## Overview

This document describes how face detection data flows from the InsightFace pipeline step through the cache to the FaceService API.

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│ InsightFaceDetect   │────▶│ UniversalCache   │────▶│ FaceService     │────▶│ REST API     │
│ FacesStep           │     │ (SQLite)         │     │                 │     │ /faces       │
└─────────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
        │                           │                        │
        │ Serializes faces          │ data_blob              │ Deserializes
        │ to JSON bytes             │ (LargeBinary)          │ JSON bytes
        ▼                           ▼                        ▼
   Dict → bytes                bytes stored             bytes → Dict
```

## 1. InsightFace Detection Step

**File:** `sim_bench/pipeline/steps/insightface_detect_faces.py`

### Detection Flow

```python
# Step processes each image
faces = analyzer.detect_faces(Path(image_path), person_data)

# Serializes to JSON-compatible dict
result = {
    'faces': [
        {
            'face_index': 0,
            'bbox': {
                'x': 0.495,      # Relative coordinates (0-1)
                'y': 0.157,
                'w': 0.147,
                'h': 0.167,
                'x_px': 1719,   # Pixel coordinates
                'y_px': 730,
                'w_px': 511,
                'h_px': 776
            },
            'confidence': 0.92,
            'landmarks': [[x1,y1], [x2,y2], ...],  # 5-point landmarks
            'person_bbox': {...} or None,
            'face_occluded': False
        },
        ...
    ]
}
```

### Cache Configuration

```python
def _get_cache_config(self, context, config):
    return {
        "items": image_paths,                    # List of image paths
        "feature_type": "insightface_detection", # Cache key component
        "model_name": "buffalo_l",               # Cache key component
        "metadata": {"device": "cpu"}
    }
```

### Serialization

```python
def _serialize_for_cache(self, result, item):
    return Serializers.json_serialize(result)  # Dict → JSON bytes
```

## 2. BaseStep Template Method (Caching)

**File:** `sim_bench/pipeline/base.py`

The `BaseStep` class provides automatic caching via template method pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    process(context, config)                  │
├─────────────────────────────────────────────────────────────┤
│  1. _get_cache_config() → Get items, feature_type, model    │
│  2. Build CacheKey for each item                            │
│  3. cache_handler.load_from_cache(keys)                     │
│  4. For uncached items: _process_uncached()                 │
│  5. _serialize_for_cache() → bytes                          │
│  6. cache_handler.store_to_cache(key, bytes)                │
│  7. _store_results() → Write to context                     │
└─────────────────────────────────────────────────────────────┘
```

### Cache Key Structure

```python
CacheKey(
    image_path="D:/photos/image.jpg",
    feature_type="insightface_detection",
    model_name="buffalo_l"
)
```

## 3. UniversalCache Table

**File:** `sim_bench/api/database/models.py`

```sql
CREATE TABLE universal_cache (
    id              INTEGER PRIMARY KEY,
    image_path      TEXT NOT NULL,        -- "D:/photos/image.jpg"
    feature_type    TEXT NOT NULL,        -- "insightface_detection"
    model_name      TEXT NOT NULL,        -- "buffalo_l"
    model_version   TEXT,
    data_blob       BLOB NOT NULL,        -- JSON bytes (face data)
    image_mtime     REAL NOT NULL,        -- For cache invalidation
    created_at      DATETIME,
    last_accessed   DATETIME,

    UNIQUE(image_path, feature_type, model_name)
);
```

### What's Stored in data_blob

The `data_blob` column contains **JSON bytes** (not raw binary). To read:

```python
import json
face_data = json.loads(entry.data_blob.decode('utf-8'))
# Returns: {'faces': [{face_index, bbox, confidence, landmarks, ...}, ...]}
```

## 4. FaceService

**File:** `sim_bench/api/services/face_service.py`

### Reading Faces from Cache

```python
def get_all_faces(self, album_id, run_id, status_filter=None):
    # Query all face detections from cache
    cache_entries = (
        self._session.query(UniversalCache)
        .filter(UniversalCache.feature_type == "insightface_detection")
        .all()
    )

    faces = []
    for entry in cache_entries:
        # Deserialize JSON bytes → dict
        face_data = json.loads(entry.data_blob.decode('utf-8'))

        for face in face_data.get('faces', []):
            # Build FaceInfo with status from Person records
            face_info = FaceInfo(
                face_key=f"{entry.image_path}:face_{face['face_index']}",
                bbox=face['bbox'],
                thumbnail_base64=self._generate_face_thumbnail(
                    entry.image_path,
                    face['bbox']
                ),
                ...
            )
            faces.append(face_info)

    return faces
```

### Thumbnail Generation

Thumbnails are generated on-the-fly by cropping from the original image:

```python
def _generate_face_thumbnail(self, image_path, bbox, size=80):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # Handle rotation

    # Use pixel coordinates from bbox
    x, y = bbox['x_px'], bbox['y_px']
    w, h = bbox['w_px'], bbox['h_px']

    # Crop and resize
    face_crop = img.crop((x, y, x + w, y + h))
    face_crop.thumbnail((size, size))

    # Encode to base64
    buffer = io.BytesIO()
    face_crop.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
```

## 5. Face Key Convention

Faces are identified by a composite key:

```
{image_path}:face_{index}
```

Examples:
- `D:/photos/image.jpg:face_0`
- `D:/photos/image.jpg:face_1`

This key is used:
- In `Person.face_instances` to track which faces belong to which person
- In `FaceOverride` to track user corrections
- In face embedding cache (`feature_type="face_embedding"`)

## 6. Related Cache Feature Types

| feature_type | model_name | Content |
|--------------|------------|---------|
| `insightface_detection` | `buffalo_l` | Face detections with bbox, landmarks |
| `face_embedding` | `arcface_insightface` | 512-dim embedding vector per face |
| `insightface_person` | `yolov8s-pose` | Person detection with keypoints |

## Common Pitfalls

1. **`data_blob` is bytes, not JSON** - Must decode: `json.loads(entry.data_blob.decode('utf-8'))`

2. **Path normalization** - Cache stores forward slashes, Windows uses backslashes. Always normalize:
   ```python
   path = path.replace('\\', '/')
   ```

3. **Face crops are NOT stored** - Unlike MediaPipe pipeline, InsightFace does NOT save cropped face images to disk. Thumbnails are generated on-demand from original images + bbox.

4. **image_metrics vs cache** - `PipelineResult.image_metrics` stores aggregated scores (face_count, frontal_scores). Raw face detection data is only in `UniversalCache`.
