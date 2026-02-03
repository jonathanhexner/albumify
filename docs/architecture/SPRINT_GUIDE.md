# Sprint Implementation Guide

## ⚠️ IMPLEMENTATION STATUS (February 2026)

**Most features described in this guide are now IMPLEMENTED:**

- ✅ **Sprint 1 (Feature Caching)**: UniversalCache + UniversalCacheHandler exist
- ✅ **Sprint 2 (Pipeline Steps)**: All 18 steps implemented (score_ava, detect_faces, etc.)
- ⚠️ **Sprints 3-5 (UI)**: Status unclear - needs verification

**This document is kept as a reference guide for the implementation approach used.**

---

## Overview

This guide provides step-by-step instructions for implementing the Phase 2 work across 5 sprints.

---

## 🚀 Sprint 1: Feature Caching (Priority 1 - CRITICAL)

**Duration:** 4-5 hours  
**Goal:** Enable feature caching for 10-100x speedup on repeated runs

### Step 1.1: Add FeatureCache Table (30 min)

**File:** `sim_bench/api/database/models.py`

```python
from sqlalchemy import Column, Integer, String, Float, LargeBinary, DateTime, JSON, Index, UniqueConstraint

class FeatureCache(Base):
    """Cached features for images (embeddings, scores, detections)."""
    __tablename__ = "feature_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Cache key (unique together)
    image_path = Column(String, nullable=False, index=True)
    feature_type = Column(String, nullable=False)  # 'scene_embedding', 'iqa_score', etc.
    model_name = Column(String, nullable=False)    # 'dinov2', 'pyiqa', etc.
    model_version = Column(String, nullable=True)
    
    # Cached value (only one populated based on feature_type)
    value_float = Column(Float, nullable=True)         # For scores
    value_vector = Column(LargeBinary, nullable=True)  # For embeddings (numpy bytes)
    value_json = Column(JSON, nullable=True)           # For structured data
    
    # Metadata
    vector_dim = Column(Integer, nullable=True)
    image_mtime = Column(Float, nullable=False)        # File modification time
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feature_lookup', 'image_path', 'feature_type', 'model_name'),
        UniqueConstraint('image_path', 'feature_type', 'model_name', name='uq_cache_key'),
    )
```

**Test:**
```python
# In Python shell
from sim_bench.api.database.session import create_db_engine, init_db
engine = create_db_engine()
init_db(engine)  # Creates new table
```

---

### Step 1.2: Implement CacheService (2 hours)

**File:** `sim_bench/api/services/cache_service.py` (NEW)

```python
"""Cache service for storing computed features."""

import io
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
from sqlalchemy.orm import Session

from sim_bench.api.database.models import FeatureCache

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching computed features with mtime validation."""
    
    def __init__(self, session: Session):
        self._session = session
    
    # ============ Float Operations ============
    
    def get_float(
        self,
        image_path: str,
        feature_type: str,
        model_name: str
    ) -> Optional[float]:
        """Get cached float value (scores)."""
        entry = self._get_valid_entry(image_path, feature_type, model_name)
        return entry.value_float if entry else None
    
    def set_float(
        self,
        image_path: str,
        feature_type: str,
        model_name: str,
        value: float,
        model_version: str = None
    ) -> None:
        """Cache a float value."""
        self._upsert_entry(
            image_path, feature_type, model_name, model_version,
            value_float=value
        )
    
    # ============ Vector Operations ============
    
    def get_vector(
        self,
        image_path: str,
        feature_type: str,
        model_name: str
    ) -> Optional[np.ndarray]:
        """Get cached vector (embeddings)."""
        entry = self._get_valid_entry(image_path, feature_type, model_name)
        if entry and entry.value_vector:
            return self._bytes_to_vector(entry.value_vector)
        return None
    
    def set_vector(
        self,
        image_path: str,
        feature_type: str,
        model_name: str,
        value: np.ndarray,
        model_version: str = None
    ) -> None:
        """Cache a vector value."""
        vector_bytes = self._vector_to_bytes(value)
        self._upsert_entry(
            image_path, feature_type, model_name, model_version,
            value_vector=vector_bytes,
            vector_dim=value.shape[0]
        )
    
    # ============ JSON Operations ============
    
    def get_json(
        self,
        image_path: str,
        feature_type: str,
        model_name: str
    ) -> Optional[dict]:
        """Get cached JSON (structured data)."""
        entry = self._get_valid_entry(image_path, feature_type, model_name)
        return entry.value_json if entry else None
    
    def set_json(
        self,
        image_path: str,
        feature_type: str,
        model_name: str,
        value: dict,
        model_version: str = None
    ) -> None:
        """Cache a JSON value."""
        self._upsert_entry(
            image_path, feature_type, model_name, model_version,
            value_json=value
        )
    
    # ============ Batch Operations ============
    
    def get_vectors_batch(
        self,
        image_paths: List[str],
        feature_type: str,
        model_name: str
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Get cached vectors for multiple images.
        
        Returns dict mapping image_path -> vector (or None if not cached/invalid).
        """
        result = {}
        
        for path in image_paths:
            result[path] = self.get_vector(path, feature_type, model_name)
        
        return result
    
    def set_vectors_batch(
        self,
        features: Dict[str, np.ndarray],
        feature_type: str,
        model_name: str,
        model_version: str = None
    ) -> None:
        """Cache vectors for multiple images in one transaction."""
        for image_path, vector in features.items():
            self.set_vector(image_path, feature_type, model_name, vector, model_version)
        
        self._session.commit()
    
    def get_floats_batch(
        self,
        image_paths: List[str],
        feature_type: str,
        model_name: str
    ) -> Dict[str, Optional[float]]:
        """Get cached float values for multiple images."""
        result = {}
        
        for path in image_paths:
            result[path] = self.get_float(path, feature_type, model_name)
        
        return result
    
    def set_floats_batch(
        self,
        features: Dict[str, float],
        feature_type: str,
        model_name: str,
        model_version: str = None
    ) -> None:
        """Cache float values for multiple images in one transaction."""
        for image_path, value in features.items():
            self.set_float(image_path, feature_type, model_name, value, model_version)
        
        self._session.commit()
    
    # ============ Utilities ============
    
    def invalidate_image(self, image_path: str) -> int:
        """Remove all cached features for an image. Returns count deleted."""
        count = self._session.query(FeatureCache).filter(
            FeatureCache.image_path == image_path
        ).delete()
        self._session.commit()
        
        logger.info(f"Invalidated {count} cache entries for {image_path}")
        return count
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._session.query(FeatureCache).count()
        
        # Count by feature type
        from sqlalchemy import func
        by_type = self._session.query(
            FeatureCache.feature_type,
            func.count(FeatureCache.id)
        ).group_by(FeatureCache.feature_type).all()
        
        return {
            'total_entries': total,
            'by_feature_type': dict(by_type)
        }
    
    # ============ Internal Methods ============
    
    def _get_valid_entry(
        self,
        image_path: str,
        feature_type: str,
        model_name: str
    ) -> Optional[FeatureCache]:
        """Get cache entry if valid (mtime matches)."""
        entry = self._session.query(FeatureCache).filter(
            FeatureCache.image_path == image_path,
            FeatureCache.feature_type == feature_type,
            FeatureCache.model_name == model_name
        ).first()
        
        if not entry:
            return None
        
        # Check if file has been modified
        current_mtime = self._get_mtime(image_path)
        if current_mtime != entry.image_mtime:
            logger.debug(f"Cache invalid (mtime changed): {image_path}")
            self._session.delete(entry)
            self._session.commit()
            return None
        
        return entry
    
    def _upsert_entry(
        self,
        image_path: str,
        feature_type: str,
        model_name: str,
        model_version: Optional[str],
        **values
    ) -> None:
        """Insert or update cache entry."""
        mtime = self._get_mtime(image_path)
        
        entry = self._session.query(FeatureCache).filter(
            FeatureCache.image_path == image_path,
            FeatureCache.feature_type == feature_type,
            FeatureCache.model_name == model_name
        ).first()
        
        if entry:
            # Update
            for key, value in values.items():
                setattr(entry, key, value)
            entry.image_mtime = mtime
            entry.model_version = model_version
        else:
            # Insert
            entry = FeatureCache(
                image_path=image_path,
                feature_type=feature_type,
                model_name=model_name,
                model_version=model_version,
                image_mtime=mtime,
                **values
            )
            self._session.add(entry)
        
        self._session.commit()
    
    @staticmethod
    def _get_mtime(image_path: str) -> float:
        """Get file modification time."""
        return os.path.getmtime(image_path)
    
    @staticmethod
    def _vector_to_bytes(arr: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage."""
        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        return buffer.getvalue()
    
    @staticmethod
    def _bytes_to_vector(data: bytes) -> np.ndarray:
        """Convert stored bytes back to numpy array."""
        buffer = io.BytesIO(data)
        return np.load(buffer, allow_pickle=False)
```

**Test:**
```python
from sim_bench.api.services.cache_service import CacheService
from sim_bench.api.database.session import get_session

with next(get_session()) as session:
    cache = CacheService(session)
    
    # Test float
    cache.set_float("test.jpg", "iqa_score", "pyiqa", 0.85)
    assert cache.get_float("test.jpg", "iqa_score", "pyiqa") == 0.85
    
    # Test vector
    import numpy as np
    vec = np.random.randn(768)
    cache.set_vector("test.jpg", "scene_embedding", "dinov2", vec)
    cached_vec = cache.get_vector("test.jpg", "scene_embedding", "dinov2")
    assert np.allclose(vec, cached_vec)
    
    # Test stats
    print(cache.get_stats())
```

---

### Step 1.3: Integrate Cache into Pipeline (15 min)

**File:** `sim_bench/pipeline/context.py`

```python
from typing import Optional

@dataclass
class PipelineContext:
    # ... existing fields ...
    
    # NEW: Cache service
    cache_service: Optional['CacheService'] = None
```

**File:** `sim_bench/api/services/pipeline_service.py`

```python
from sim_bench.api.services.cache_service import CacheService

class PipelineService:
    def _create_context(self, album: Album, config: dict) -> PipelineContext:
        """Create pipeline context with cache service."""
        cache_service = CacheService(self._session)
        
        context = PipelineContext(
            source_directory=Path(album.source_path),
            cache_service=cache_service,
            # ... other fields ...
        )
        
        return context
```

---

### Step 1.4: Update Steps to Use Cache (1 hour)

**File:** `sim_bench/pipeline/steps/score_iqa.py`

```python
def process(self, context: PipelineContext, config: dict) -> None:
    """Score IQA with caching."""
    logger = logging.getLogger(__name__)
    
    image_paths = list(context.image_paths)
    
    # Try cache first
    if context.cache_service:
        cached_scores = context.cache_service.get_floats_batch(
            [str(p) for p in image_paths],
            feature_type="iqa_score",
            model_name="rule_based"
        )
        
        uncached_paths = [p for p in image_paths if cached_scores.get(str(p)) is None]
        
        logger.info(f"IQA cache: {len(cached_scores) - len(uncached_paths)} hits, {len(uncached_paths)} misses")
    else:
        uncached_paths = image_paths
        cached_scores = {}
    
    # Compute uncached
    if uncached_paths:
        new_scores = {}
        for path in uncached_paths:
            score = self._compute_iqa(path, config)
            new_scores[str(path)] = score
        
        # Save to cache
        if context.cache_service:
            context.cache_service.set_floats_batch(
                new_scores,
                feature_type="iqa_score",
                model_name="rule_based"
            )
        
        cached_scores.update(new_scores)
    
    # Store in context
    context.iqa_scores = cached_scores
```

**Similar changes for:**
- `extract_scene_embedding.py` - Cache embeddings as vectors
- Any future step that computes expensive features

---

### Sprint 1 Testing

```bash
# Run pipeline twice on same album
python -m uvicorn sim_bench.api.main:app --reload --port 8000

# In another terminal
curl -X POST http://localhost:8000/api/v1/albums/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Test", "source_path": "D:/photos/test"}'

# Run pipeline
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"album_id": "...", "steps": null}'

# Check logs for cache hits
tail -f logs/*/api.log | grep "cache"
```

**Success Criteria:**
- First run takes normal time
- Second run is 10x+ faster
- Cache hit rate shown in logs

---

## 🔧 Sprint 2: Core Pipeline Steps (Day 2)

**Duration:** 4 hours  
**Goal:** Implement missing analysis steps (AVA, faces)

### Implementation Order

1. **score_ava.py** (1 hour) - Aesthetic scoring
2. **detect_faces.py** (1 hour) - MediaPipe wrapper
3. **score_face_pose.py** (30 min) - SixDRepNet wrapper
4. **score_face_eyes.py** (30 min) - MediaPipe EAR
5. **score_face_smile.py** (30 min) - MediaPipe smile
6. **filter_portrait.py** (30 min) - Filter by face quality

See `docs/architecture/PIPELINE_ARCHITECTURE_PLAN.md` for detailed step specifications.

---

## 🎨 Sprint 3-5: UI Enhancements

See `IMPLEMENTATION_PLAN.md` for detailed UI component specifications.

---

## Quick Commands

```bash
# Start backend
.venv\Scripts\python -m uvicorn sim_bench.api.main:app --reload --port 8000

# Start frontend
.venv\Scripts\python -m app.nicegui.main

# Run tests
.venv\Scripts\pytest tests/

# Check cache stats
python -c "from sim_bench.api.services.cache_service import CacheService; from sim_bench.api.database.session import get_session; print(CacheService(next(get_session())).get_stats())"
```
