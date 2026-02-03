# Quick Start - Phase 2 Implementation

## ⚠️ STATUS UPDATE (February 2026)

**Most of the issues described below have been RESOLVED:**

### ✅ Problem 1: Pipeline Steps - **SOLVED**
- ✅ ALL 18 pipeline steps are now implemented
- ✅ AVA scoring, face detection, face analysis, clustering, people feature all exist
- **Status:** Feature-complete pipeline

### ✅ Problem 2: Feature Caching - **IMPLEMENTED**
- ✅ UniversalCache database table exists
- ✅ UniversalCacheHandler provides caching API
- ✅ File-based FeatureCache also available
- **Status:** Needs verification that 10-100x speedup is achieved

### ⚠️ Problem 3: UI - **NEEDS VERIFICATION**
- Status of NiceGUI frontend richness unclear
- May still need config sidebar enhancements
- May still need real-time metrics
- **Status:** Requires testing to verify vs Streamlit parity

---

## 🚀 START HERE: Sprint 1 - Feature Caching

**Why first?** Without caching, implementing other features is too slow to iterate.

**Duration:** 4-5 hours  
**Result:** 10-100x speedup on repeated runs

### Step 1: Add Database Table (30 min)

**File:** `sim_bench/api/database/models.py`

Add at the end of the file:

```python
class FeatureCache(Base):
    """Cached features (embeddings, scores, detections)."""
    __tablename__ = "feature_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False, index=True)
    feature_type = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=True)
    
    value_float = Column(Float, nullable=True)
    value_vector = Column(LargeBinary, nullable=True)
    value_json = Column(JSON, nullable=True)
    
    vector_dim = Column(Integer, nullable=True)
    image_mtime = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feature_lookup', 'image_path', 'feature_type', 'model_name'),
        UniqueConstraint('image_path', 'feature_type', 'model_name'),
    )
```

Test by starting the backend - it will create the table.

---

### Step 2: Create CacheService (2 hours)

**File:** `sim_bench/api/services/cache_service.py` (NEW FILE)

Copy the complete implementation from `docs/architecture/SPRINT_GUIDE.md` section 1.2.

Key methods:
- `get_float()` / `set_float()` - For scores
- `get_vector()` / `set_vector()` - For embeddings
- `get_vectors_batch()` / `set_vectors_batch()` - Batch operations
- `invalidate_image()` - Clear cache for modified images

---

### Step 3: Add Cache to Pipeline (15 min)

**File:** `sim_bench/pipeline/context.py`

Add this field to `PipelineContext`:

```python
from typing import Optional

@dataclass
class PipelineContext:
    # ... existing fields ...
    cache_service: Optional['CacheService'] = None
```

**File:** `sim_bench/api/services/pipeline_service.py`

In the method that creates context, add:

```python
from sim_bench.api.services.cache_service import CacheService

# When creating context
cache_service = CacheService(self._session)
context = PipelineContext(
    source_directory=...,
    cache_service=cache_service,  # Add this
    ...
)
```

---

### Step 4: Update Steps to Use Cache (1 hour)

**File:** `sim_bench/pipeline/steps/score_iqa.py`

Replace the `process` method:

```python
def process(self, context: PipelineContext, config: dict) -> None:
    """Score IQA with caching."""
    image_paths = [str(p) for p in context.image_paths]
    
    # Check cache
    cached_scores = {}
    if context.cache_service:
        cached_scores = context.cache_service.get_floats_batch(
            image_paths,
            feature_type="iqa_score",
            model_name="rule_based"
        )
        uncached = [p for p in image_paths if cached_scores.get(p) is None]
        print(f"Cache: {len(cached_scores)-len(uncached)} hits, {len(uncached)} misses")
    else:
        uncached = image_paths
    
    # Compute uncached only
    if uncached:
        new_scores = {}
        for path in uncached:
            score = self._compute_iqa(path, config)
            new_scores[path] = score
        
        # Save to cache
        if context.cache_service:
            context.cache_service.set_floats_batch(
                new_scores, "iqa_score", "rule_based"
            )
        
        cached_scores.update(new_scores)
    
    context.iqa_scores = cached_scores
```

**Do the same for:**
- `extract_scene_embedding.py` - Use `get_vectors_batch()` / `set_vectors_batch()`

---

### Step 5: Test It

```bash
# Start backend
.venv\Scripts\python -m uvicorn sim_bench.api.main:app --reload --port 8000

# Run pipeline twice and compare times
# First run: Full computation
# Second run: Should be 10-50x faster
```

---

## 📊 Expected Results

### Before (Current):
```
Run 1: 225 seconds (compute everything)
Run 2: 225 seconds (recompute everything!) ❌
Run 3: 225 seconds (still recomputing!) ❌
```

### After (With Caching):
```
Run 1: 225 seconds (compute everything)
Run 2: 2-5 seconds (cache hit!) ✅
Run 3: 2-5 seconds (cache hit!) ✅
```

**Speedup: 45-110x for repeated runs**

---

## 🔄 Next Steps After Caching Works

### Sprint 2: Missing Pipeline Steps (4 hours)
1. `score_ava.py` - Aesthetic scoring
2. `detect_faces.py` - Face detection
3. Face analysis steps (pose, eyes, smile)

### Sprint 3-5: UI Improvements (8 hours)
1. Config panel with sliders
2. Real-time progress metrics
3. Gallery with cluster views

---

## 📁 Documentation References

- **Full Plan:** `docs/architecture/IMPLEMENTATION_PLAN.md` (detailed)
- **Visual Comparison:** `docs/architecture/CURRENT_VS_TARGET.md`
- **Step-by-Step Guide:** `docs/architecture/SPRINT_GUIDE.md`

---

## 🆘 Troubleshooting

**Problem:** Table not created
- **Solution:** Restart FastAPI backend, check logs

**Problem:** Cache always misses
- **Solution:** Check `image_mtime` logic in `CacheService`

**Problem:** No speedup
- **Solution:** Verify cache hits in logs, check that steps are using cache

---

## 📞 Need Help?

The full implementation with all code is in:
- `docs/architecture/SPRINT_GUIDE.md` - Complete code for CacheService
- `docs/architecture/IMPLEMENTATION_PLAN.md` - Full 20-hour plan
