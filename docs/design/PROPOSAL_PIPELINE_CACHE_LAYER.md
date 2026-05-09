# Proposal: Pipeline Cache Layer (LMDB + Parquet)

**Status**: Pending Approval
**Author**: Claude (for Senior SW Engineer review)
**Date**: 2026-02-20

---

## Problem Statement

Current caching has several issues:

1. **SQLite blob storage** - Not optimal for either images or numerical features
2. **Re-computation on every run** - Aligned faces, crops recalculated each time
3. **No intermediate artifacts** - Can't inspect cached data for debugging
4. **Mixed concerns** - Cache logic scattered across steps
5. **No generic pattern** - Each step implements caching differently

## Objective

Create a unified, generic cache layer that:
- Uses appropriate storage for each data type (LMDB for binary, Parquet for features)
- Provides simple getter/setter API for all pipeline steps
- Supports cache invalidation based on source file changes
- Enables easy debugging and inspection of cached data

## Proposed Architecture

### Core Design: `PipelineCache` Class

```python
from typing import TypeVar, Generic, Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

T = TypeVar('T')

class CacheBackend(ABC, Generic[T]):
    """Abstract backend for cache storage."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get item by key."""
        pass

    @abstractmethod
    def put(self, key: str, value: T, metadata: Optional[Dict] = None) -> None:
        """Store item with optional metadata."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete item by key."""
        pass

    @abstractmethod
    def keys(self, prefix: Optional[str] = None) -> List[str]:
        """List keys with optional prefix filter."""
        pass


class PipelineCache:
    """Unified cache for pipeline artifacts.

    Usage:
        cache = PipelineCache(album_path="/path/to/album")

        # Store aligned face
        cache.images.put("img001:face_0:aligned", aligned_crop)

        # Get aligned face
        crop = cache.images.get("img001:face_0:aligned")

        # Store embeddings (batch)
        cache.features.put_dataframe("face_embeddings", embeddings_df)

        # Query features
        df = cache.features.get_dataframe("face_embeddings")
    """

    def __init__(self, album_path: Path, cache_root: Optional[Path] = None):
        self.album_path = Path(album_path)
        self.cache_root = cache_root or Path.home() / ".sim_bench" / "cache"
        self.album_id = self._compute_album_id()

        # Initialize backends
        self._images = LMDBBackend(self.cache_root / self.album_id / "images.lmdb")
        self._features = ParquetBackend(self.cache_root / self.album_id / "features")
        self._metadata = JSONBackend(self.cache_root / self.album_id / "metadata.json")

    @property
    def images(self) -> "LMDBBackend":
        """LMDB backend for images and binary data."""
        return self._images

    @property
    def features(self) -> "ParquetBackend":
        """Parquet backend for features and embeddings."""
        return self._features

    @property
    def metadata(self) -> "JSONBackend":
        """JSON backend for small metadata."""
        return self._metadata

    def _compute_album_id(self) -> str:
        """Compute stable ID for album (hash of path)."""
        return hashlib.md5(str(self.album_path).encode()).hexdigest()[:12]

    def invalidate_image(self, image_path: str) -> None:
        """Invalidate all cached data for an image."""
        prefix = self._image_key_prefix(image_path)
        for key in self.images.keys(prefix):
            self.images.delete(key)
        # Also invalidate features
        self.features.invalidate_by_image(image_path)
```

### Backend Implementations

#### 1. LMDBBackend (Images)

```python
import lmdb
import cv2
import numpy as np

class LMDBBackend(CacheBackend[np.ndarray]):
    """LMDB backend for image storage.

    Key format: "{image_hash}:{artifact_type}"
    Artifact types:
        - original: EXIF-normalized full image
        - face_{idx}:raw: Bbox crop without alignment
        - face_{idx}:aligned: Orientation-corrected + 5-point aligned
        - face_{idx}:thumbnail: Small preview (64x64)
    """

    def __init__(self, path: Path, map_size: int = 10 * 1024**3):  # 10GB default
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            map_size=map_size,
            readonly=False,
            create=True
        )

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get image by key."""
        with self.env.begin() as txn:
            data = txn.get(key.encode())
            if data is None:
                return None
            # Decode JPEG bytes to numpy array
            return cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

    def put(self, key: str, image: np.ndarray, quality: int = 95) -> None:
        """Store image (JPEG compressed)."""
        # Encode as JPEG
        _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), encoded.tobytes())

    def put_png(self, key: str, image: np.ndarray) -> None:
        """Store image (PNG lossless)."""
        _, encoded = cv2.imencode('.png', image)
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), encoded.tobytes())

    def exists(self, key: str) -> bool:
        with self.env.begin() as txn:
            return txn.get(key.encode()) is not None

    def delete(self, key: str) -> None:
        with self.env.begin(write=True) as txn:
            txn.delete(key.encode())

    def keys(self, prefix: Optional[str] = None) -> List[str]:
        result = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                key_str = key.decode()
                if prefix is None or key_str.startswith(prefix):
                    result.append(key_str)
        return result

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stat = self.env.stat()
        info = self.env.info()
        return {
            "entries": stat["entries"],
            "size_mb": info["map_size"] / 1024**2,
            "used_mb": stat["psize"] * stat["leaf_pages"] / 1024**2,
        }
```

#### 2. ParquetBackend (Features)

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

class ParquetBackend(CacheBackend[pd.DataFrame]):
    """Parquet backend for structured feature storage.

    File structure:
        features/
            face_metadata.parquet      # bbox, landmarks, orientation, confidence
            face_embeddings.parquet    # 512-dim vectors
            image_scores.parquet       # IQA, AVA, sharpness
            clustering_labels.parquet  # cluster assignments

    Each parquet file has:
        - image_path: str (index)
        - face_index: int (for face-level data)
        - mtime: float (source file modification time)
        - ... feature columns
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """Get entire feature table."""
        file_path = self.path / f"{name}.parquet"
        if not file_path.exists():
            return None
        return pd.read_parquet(file_path)

    def put_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """Store entire feature table."""
        file_path = self.path / f"{name}.parquet"
        df.to_parquet(file_path, index=False)

    def get_rows(self, name: str, image_path: str) -> Optional[pd.DataFrame]:
        """Get rows for specific image."""
        df = self.get_dataframe(name)
        if df is None:
            return None
        return df[df["image_path"] == image_path]

    def put_rows(self, name: str, image_path: str, rows: pd.DataFrame) -> None:
        """Update/insert rows for specific image."""
        df = self.get_dataframe(name)
        if df is None:
            df = rows
        else:
            # Remove existing rows for this image
            df = df[df["image_path"] != image_path]
            # Append new rows
            df = pd.concat([df, rows], ignore_index=True)
        self.put_dataframe(name, df)

    def invalidate_by_image(self, image_path: str) -> None:
        """Remove all feature rows for an image."""
        for file_path in self.path.glob("*.parquet"):
            name = file_path.stem
            df = self.get_dataframe(name)
            if df is not None and "image_path" in df.columns:
                df = df[df["image_path"] != image_path]
                self.put_dataframe(name, df)

    def get_embeddings(self, name: str = "face_embeddings") -> Optional[np.ndarray]:
        """Get embeddings as numpy array (optimized for clustering)."""
        df = self.get_dataframe(name)
        if df is None:
            return None
        # Assume embedding columns are named emb_0, emb_1, ..., emb_511
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            return None
        return df[sorted(emb_cols)].values

    def put_embeddings(
        self,
        keys: List[str],
        embeddings: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        name: str = "face_embeddings"
    ) -> None:
        """Store embeddings with keys."""
        # Parse keys into image_path and face_index
        data = []
        for i, key in enumerate(keys):
            parts = key.rsplit(":face_", 1)
            image_path = parts[0]
            face_index = int(parts[1]) if len(parts) > 1 else 0
            row = {
                "image_path": image_path,
                "face_index": face_index,
                "key": key,
            }
            # Add embedding dimensions
            for j, val in enumerate(embeddings[i]):
                row[f"emb_{j}"] = val
            data.append(row)

        df = pd.DataFrame(data)
        if metadata is not None:
            df = df.merge(metadata, on=["image_path", "face_index"], how="left")

        self.put_dataframe(name, df)
```

### Integration with Pipeline Steps

#### CacheMixin for Steps

```python
class CacheAwareStep(BaseStep):
    """Mixin for steps that use the cache layer."""

    def get_cache(self, context: PipelineContext) -> PipelineCache:
        """Get or create cache for current album."""
        if not hasattr(context, '_pipeline_cache'):
            context._pipeline_cache = PipelineCache(context.source_directory)
        return context._pipeline_cache


# Example: Updated align_faces step
@register_step
class AlignFacesStep(CacheAwareStep):
    """Align faces using orientation correction and 5-point affine transform."""

    def process(self, context: PipelineContext, config: dict) -> None:
        cache = self.get_cache(context)
        target_size = config.get("target_size", 256)

        aligned_faces = {}

        for image_path, face_data in context.insightface_faces.items():
            image_key = self._image_key(image_path)

            # Check cache first
            for face_info in face_data.get('faces', []):
                face_idx = face_info.get('face_index', 0)
                cache_key = f"{image_key}:face_{face_idx}:aligned"

                # Try cache
                cached = cache.images.get(cache_key)
                if cached is not None:
                    aligned_faces[f"{image_path}:face_{face_idx}"] = cached
                    continue

                # Compute alignment
                aligned = self._align_face(image_path, face_info, target_size)
                if aligned is not None:
                    # Store in cache
                    cache.images.put(cache_key, aligned)
                    aligned_faces[f"{image_path}:face_{face_idx}"] = aligned

        context.aligned_faces = aligned_faces
```

### Key Generation Strategy

```python
class KeyGenerator:
    """Consistent key generation for cache entries."""

    @staticmethod
    def image_key(image_path: str) -> str:
        """Generate stable key for an image.

        Uses content hash (EXIF datetime + file size) for stability
        across path changes.
        """
        path = Path(image_path)
        stat = path.stat()

        # Try to get EXIF datetime
        exif_date = get_exif_datetime(path)
        if exif_date:
            content_id = f"{exif_date}_{stat.st_size}"
        else:
            # Fallback to path + mtime
            content_id = f"{path.name}_{stat.st_mtime}_{stat.st_size}"

        return hashlib.md5(content_id.encode()).hexdigest()[:16]

    @staticmethod
    def face_key(image_key: str, face_index: int, artifact: str) -> str:
        """Generate key for face artifact.

        Artifacts: 'raw', 'aligned', 'thumbnail'
        """
        return f"{image_key}:face_{face_index}:{artifact}"

    @staticmethod
    def feature_key(image_path: str, feature_type: str) -> str:
        """Generate key for image-level feature."""
        image_key = KeyGenerator.image_key(image_path)
        return f"{image_key}:{feature_type}"
```

### Cache Invalidation

```python
class CacheInvalidator:
    """Handle cache invalidation based on source changes."""

    def __init__(self, cache: PipelineCache):
        self.cache = cache

    def check_and_invalidate(self, image_path: str) -> bool:
        """Check if image has changed and invalidate if needed.

        Returns True if cache was invalidated.
        """
        path = Path(image_path)
        if not path.exists():
            return False

        current_mtime = path.stat().st_mtime
        cached_mtime = self.cache.metadata.get(f"mtime:{image_path}")

        if cached_mtime is None or current_mtime > cached_mtime:
            # Image changed, invalidate all artifacts
            self.cache.invalidate_image(image_path)
            self.cache.metadata.put(f"mtime:{image_path}", current_mtime)
            return True

        return False

    def invalidate_album(self) -> None:
        """Invalidate entire album cache."""
        # Remove LMDB and Parquet files
        shutil.rmtree(self.cache.cache_root / self.cache.album_id)
```

## Data Schema

### face_metadata.parquet

| Column | Type | Description |
|--------|------|-------------|
| image_path | str | Full path to source image |
| face_index | int | Face index within image |
| bbox_x | float | Normalized bbox x |
| bbox_y | float | Normalized bbox y |
| bbox_w | float | Normalized bbox width |
| bbox_h | float | Normalized bbox height |
| bbox_x_px | int | Pixel bbox x |
| bbox_y_px | int | Pixel bbox y |
| bbox_w_px | int | Pixel bbox width |
| bbox_h_px | int | Pixel bbox height |
| confidence | float | Detection confidence |
| orientation_angle | int | 0, 90, 180, 270 |
| roll_angle | float | Eye-line tilt angle |
| frontal_score | float | How frontal the face is |
| filter_passed | bool | Passed size/confidence filter |
| is_clusterable | bool | Suitable for clustering |
| landmark_0_x | float | Left eye x |
| landmark_0_y | float | Left eye y |
| ... | ... | Other landmarks |

### face_embeddings.parquet

| Column | Type | Description |
|--------|------|-------------|
| image_path | str | Full path to source image |
| face_index | int | Face index within image |
| key | str | Cache key |
| emb_0 | float | Embedding dimension 0 |
| emb_1 | float | Embedding dimension 1 |
| ... | ... | ... |
| emb_511 | float | Embedding dimension 511 |

### image_scores.parquet

| Column | Type | Description |
|--------|------|-------------|
| image_path | str | Full path to source image |
| iqa_score | float | Technical quality |
| ava_score | float | Aesthetic score |
| sharpness | float | Sharpness score |
| mtime | float | Source file mtime |

### clustering_labels.parquet

| Column | Type | Description |
|--------|------|-------------|
| image_path | str | Full path to source image |
| face_index | int | Face index |
| cluster_label | int | Cluster assignment (-1 = noise) |
| method | str | Clustering method used |
| run_timestamp | str | When clustering was run |

## Migration Plan

1. **Phase 1**: Implement core backends (LMDB, Parquet, JSON)
2. **Phase 2**: Implement `PipelineCache` class with unified API
3. **Phase 3**: Update `align_faces` step to use cache
4. **Phase 4**: Update `extract_face_embeddings` to use cache
5. **Phase 5**: Update remaining steps (IQA, AVA, scene embeddings)
6. **Phase 6**: Add CLI commands for cache inspection/management
7. **Phase 7**: Remove old SQLite `UniversalCache`

## CLI Commands

```bash
# Inspect cache
python -m sim_bench.cache info /path/to/album
# Output:
#   Album: /path/to/album
#   Cache ID: a1b2c3d4e5f6
#   Images: 1,234 (456 MB)
#   Features: 4 tables
#     - face_metadata: 5,678 rows
#     - face_embeddings: 5,678 rows (512 dims)
#     - image_scores: 1,234 rows
#     - clustering_labels: 5,678 rows

# List cached faces for image
python -m sim_bench.cache faces /path/to/album/image.jpg

# Export embeddings to numpy
python -m sim_bench.cache export embeddings /path/to/album -o embeddings.npy

# Clear cache
python -m sim_bench.cache clear /path/to/album

# Clear specific artifact type
python -m sim_bench.cache clear /path/to/album --type aligned
```

## Benefits

1. **Speed**: No recomputation of aligned faces, embeddings on repeat runs
2. **Debugging**: Inspect cached artifacts directly (LMDB viewer, Parquet in pandas)
3. **Generic**: Same API for all steps, consistent key generation
4. **Appropriate storage**: LMDB for images, Parquet for features
5. **Batch operations**: Parquet enables efficient batch reads for clustering
6. **Portable**: Cache can be copied/shared between machines

## Risks and Trade-offs

1. **Disk space**: LMDB + Parquet will use more space than SQLite blobs
2. **Dependencies**: Adds `lmdb` and `pyarrow` dependencies
3. **Migration**: Need to handle transition from old cache
4. **Complexity**: More moving parts than single SQLite file

## Acceptance Criteria

- [ ] `PipelineCache` provides unified getter/setter API
- [ ] LMDB stores images efficiently (JPEG compressed)
- [ ] Parquet stores features with proper schema
- [ ] Cache invalidation works on file mtime changes
- [ ] `align_faces` step uses cache (no recomputation)
- [ ] `extract_face_embeddings` step uses cache
- [ ] CLI commands for cache inspection work
- [ ] Old pipeline tests still pass
- [ ] New cache tests cover edge cases
