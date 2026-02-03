# Implementation Plan - Phase 2

## ⚠️ STATUS UPDATE (February 2026)

**This document was originally written as a plan. Most features described below are now IMPLEMENTED.**

See [CURRENT_IMPLEMENTATION_STATUS.md](../CURRENT_IMPLEMENTATION_STATUS.md) for current state.

---

## Current State Analysis

### ✅ What's Implemented

**Backend (FastAPI)**:
- ✅ API routers (albums, pipeline, steps, websocket)
- ✅ Database models (Album, PipelineRun, PipelineResult, UniversalCache)
- ✅ Services (AlbumService, PipelineService, ResultService)
- ✅ Logging infrastructure

**Pipeline Engine**:
- ✅ Core infrastructure (base, context, registry, builder, executor)
- ✅ **ALL 18 pipeline steps implemented** (as of Feb 2026):
  - `discover_images` - Find image files
  - `score_iqa` - Rule-based quality (sharpness, exposure)
  - `score_ava` - AVA aesthetic scoring with trained model ✅
  - `score_face_quality` - Overall face quality assessment ✅
  - `score_face_pose` - Head pose scoring ✅
  - `score_face_eyes` - Eye openness scoring ✅
  - `score_face_smile` - Smile detection scoring ✅
  - `detect_faces` - MediaPipe face detection ✅
  - `filter_quality` - Filter by IQA threshold
  - `filter_portraits` - Portrait quality filtering ✅
  - `filter_best_faces` - Filter best faces ✅
  - `extract_scene_embedding` - DINOv2 scene features
  - `extract_face_embeddings` - Face embeddings ✅
  - `cluster_scenes` - HDBSCAN clustering
  - `cluster_people` - Global face clustering ✅
  - `cluster_by_identity` - Cluster by person identity ✅
  - `select_best` - Select best per cluster
  - `select_best_per_person` - People thumbnails ✅

**Feature Caching**:
- ✅ UniversalCache database table ✅
- ✅ UniversalCacheHandler implementation (sim_bench/pipeline/cache_handler.py) ✅
- ✅ File-based FeatureCache (sim_bench/feature_cache.py) ✅
- ✅ Integration with pipeline steps ✅
- ✅ Batch cache operations ✅
- ✅ Cache invalidation logic (mtime-based) ✅

**Frontend (NiceGUI)**:
- ✅ Basic UI (home, results pages)
- ✅ API client with WebSocket
- ⚠️ UI richness may still be below Streamlit level (needs verification)

**UI/UX Enhancements**:
- ❌ Rich configuration panel (like Streamlit sidebar)
- ❌ Real-time metrics (rate, ETA, processed count)
- ❌ Multi-tab results view (Gallery, Metrics, Performance, Export)
- ❌ Image galleries with thumbnails
- ❌ Cluster visualization
- ❌ Performance charts
- ❌ Export options

---

## Priority 1: Feature Caching (CRITICAL)

**Why First?** Without caching, every pipeline run recomputes everything. This makes development painfully slow.

### Task 1.1: Database Schema Update

**File:** `sim_bench/api/database/models.py`

Add `FeatureCache` table:

```python
class FeatureCache(Base):
    """Cached features for images."""
    __tablename__ = "feature_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Cache key (unique together)
    image_path = Column(String, nullable=False, index=True)
    feature_type = Column(String, nullable=False)  # 'scene_embedding', 'iqa_score', etc.
    model_name = Column(String, nullable=False)    # 'dinov2', 'pyiqa', etc.
    model_version = Column(String, nullable=True)
    
    # Cached value (only one populated)
    value_float = Column(Float, nullable=True)     # For scores
    value_vector = Column(LargeBinary, nullable=True)  # For embeddings (numpy bytes)
    value_json = Column(JSON, nullable=True)       # For structured data
    
    # Metadata
    vector_dim = Column(Integer, nullable=True)
    image_mtime = Column(Float, nullable=False)    # File modification time
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feature_lookup', 'image_path', 'feature_type', 'model_name'),
        UniqueConstraint('image_path', 'feature_type', 'model_name', name='uq_cache_key'),
    )
```

**Estimate:** 30 minutes

### Task 1.2: Cache Service Implementation

**File:** `sim_bench/api/services/cache_service.py` (NEW)

```python
class CacheService:
    """Service for caching computed features."""
    
    def get_float(self, image_path: str, feature_type: str, model_name: str) -> Optional[float]
    def get_vector(self, image_path: str, feature_type: str, model_name: str) -> Optional[np.ndarray]
    def get_json(self, image_path: str, feature_type: str, model_name: str) -> Optional[dict]
    
    def set_float(self, image_path: str, feature_type: str, model_name: str, value: float) -> None
    def set_vector(self, image_path: str, feature_type: str, model_name: str, value: np.ndarray) -> None
    def set_json(self, image_path: str, feature_type: str, model_name: str, value: dict) -> None
    
    # Batch operations (critical for performance)
    def get_vectors_batch(self, image_paths: List[str], feature_type: str, model_name: str) -> Dict[str, Optional[np.ndarray]]
    def set_vectors_batch(self, features: Dict[str, np.ndarray], feature_type: str, model_name: str) -> None
    
    # Utilities
    def invalidate_image(self, image_path: str) -> int
    def get_stats(self) -> dict
```

**Key Implementation Details:**
- Check `mtime` on every get - invalidate if file changed
- Use numpy's efficient binary serialization for vectors
- Batch operations use single transaction for speed

**Estimate:** 2 hours

### Task 1.3: Integrate Cache into PipelineContext

**File:** `sim_bench/pipeline/context.py`

Add to `PipelineContext`:

```python
@dataclass
class PipelineContext:
    # ... existing fields ...
    
    # NEW: Cache service
    cache_service: Optional['CacheService'] = None
```

**File:** `sim_bench/api/services/pipeline_service.py`

Inject cache service when creating context:

```python
def start_pipeline(self, album_id: str, ...):
    cache_service = CacheService(self._session)
    context = PipelineContext(
        source_directory=album.source_path,
        cache_service=cache_service,
        ...
    )
```

**Estimate:** 15 minutes

### Task 1.4: Update Existing Steps to Use Cache

Update these 3 steps to use caching:

**File:** `sim_bench/pipeline/steps/score_iqa.py`

```python
def process(self, context: PipelineContext, config: dict) -> None:
    if not context.cache_service:
        # No cache - fallback to old behavior
        ...
        return
    
    # Check cache
    cached_scores = context.cache_service.get_floats_batch(
        list(context.image_paths),
        feature_type="iqa_score",
        model_name="rule_based"
    )
    
    # Identify uncached
    uncached = [p for p in context.image_paths if cached_scores.get(str(p)) is None]
    
    # Compute only uncached
    if uncached:
        new_scores = self._compute_scores(uncached)
        context.cache_service.set_floats_batch(new_scores, "iqa_score", "rule_based")
        cached_scores.update(new_scores)
    
    # Store in context
    context.iqa_scores = cached_scores
```

**Similar changes for:**
- `extract_scene_embedding.py` - Cache embeddings
- `detect_faces.py` (when implemented) - Cache face detections

**Estimate:** 1 hour

---

## Priority 2: Complete Pipeline Steps

### Task 2.1: AVA Aesthetic Scoring

**File:** `sim_bench/pipeline/steps/score_ava.py` (NEW)

```python
class ScoreAVAStep(BaseStep):
    """Score aesthetic quality using trained AVA model."""
    
    metadata = StepMetadata(
        name="score_ava",
        display_name="Score Aesthetics (AVA)",
        category="analysis",
        requires={"image_paths"},
        produces={"ava_scores"},
        depends_on=["discover_images"]
    )
    
    def process(self, context: PipelineContext, config: dict) -> None:
        # Use existing AVAQualityModel from sim_bench/image_quality_models/
        ...
```

**Estimate:** 1 hour (wrapper around existing model)

### Task 2.2: Face Detection & Analysis Steps

**Files to create:**
- `sim_bench/pipeline/steps/detect_faces.py` - Wrap MediaPipe
- `sim_bench/pipeline/steps/score_face_pose.py` - Wrap SixDRepNet
- `sim_bench/pipeline/steps/score_face_eyes.py` - Wrap MediaPipe EAR
- `sim_bench/pipeline/steps/score_face_smile.py` - Wrap MediaPipe smile

**Estimate:** 3 hours (wrapping existing services)

### Task 2.3: Face Filtering Steps

**Files to create:**
- `sim_bench/pipeline/steps/filter_portrait.py` - Filter by eyes/smile
- `sim_bench/pipeline/steps/filter_pose.py` - Filter by head pose

**Estimate:** 1 hour

### Task 2.4: Face Clustering Steps

**Files to create:**
- `sim_bench/pipeline/steps/extract_face_embedding.py` - ArcFace embeddings
- `sim_bench/pipeline/steps/cluster_faces.py` - Within-scene clustering
- `sim_bench/pipeline/steps/cluster_people.py` - Global clustering

**Estimate:** 2 hours

### Task 2.5: Selection Steps

**Files to create:**
- `sim_bench/pipeline/steps/select_best_face_per_identity.py`
- `sim_bench/pipeline/steps/select_best_per_person.py`

**Estimate:** 1 hour

**Total for Priority 2:** ~8 hours

---

## Priority 3: NiceGUI UI Enhancements

### Task 3.1: Rich Configuration Panel

**File:** `app/nicegui/components/config_panel.py` (NEW)

Recreate Streamlit sidebar experience:

```python
class ConfigPanel:
    """Rich configuration panel with collapsible sections."""
    
    def __init__(self, container):
        with container:
            self._render_quality_thresholds()
            self._render_portrait_preferences()
            self._render_selection_weights()
            self._render_clustering()
            self._render_performance()
            self._render_export()
    
    def _render_quality_thresholds(self):
        with ui.expansion('📊 Quality Thresholds', value=True):
            with ui.row():
                self.min_iqa = ui.slider('Min IQA', min=0, max=1, value=0.3, step=0.05)
                self.min_ava = ui.slider('Min AVA', min=1, max=10, value=4.0, step=0.5)
            # ... more sliders
    
    def get_config(self) -> dict:
        """Return current configuration."""
        ...
```

**Features to match Streamlit:**
- Collapsible sections with expansion panels
- Two-column layouts for compact display
- Validation warnings (e.g., weights sum to 1.0)
- Default values matching Streamlit

**Estimate:** 2 hours

### Task 3.2: Enhanced Progress Display

**File:** `app/nicegui/components/progress_display.py` (NEW)

```python
class ProgressDisplay:
    """Rich progress display with real-time metrics."""
    
    def __init__(self, container):
        with container:
            self.progress_bar = ui.linear_progress()
            self.status_text = ui.label()
            self.detail_text = ui.label()
            
            with ui.row():
                self.processed_card = ui.card().tight()  # Processed count
                self.rate_card = ui.card().tight()       # Images/second
                self.elapsed_card = ui.card().tight()    # Elapsed time
                self.eta_card = ui.card().tight()        # ETA
    
    def update(self, stage: str, progress: float, detail: str = None):
        """Update progress display."""
        self.progress_bar.value = progress
        self.status_text.text = STAGE_DESCRIPTIONS[stage]
        # ... calculate and update metrics
```

**Features:**
- Real-time rate calculation (images/second)
- ETA estimation based on current rate
- Visual stage descriptions with emojis
- Detail text for current image

**Estimate:** 1.5 hours

### Task 3.3: Multi-Tab Results View

**File:** `app/nicegui/pages/results.py` (ENHANCE)

```python
def render_results_tabs(result: PipelineResult):
    """Render results in tabs like Streamlit."""
    
    with ui.tabs() as tabs:
        gallery_tab = ui.tab('🖼️ Gallery')
        metrics_tab = ui.tab('📊 Metrics')
        performance_tab = ui.tab('⚡ Performance')
        export_tab = ui.tab('📤 Export')
    
    with ui.tab_panels(tabs, value=gallery_tab):
        with ui.tab_panel(gallery_tab):
            render_gallery(result)
        
        with ui.tab_panel(metrics_tab):
            render_metrics(result)
        
        with ui.tab_panel(performance_tab):
            render_performance_charts(result)
        
        with ui.tab_panel(export_tab):
            render_export_options(result)
```

**Estimate:** 1 hour

### Task 3.4: Image Gallery Component

**File:** `app/nicegui/components/gallery.py` (NEW)

```python
class ImageGallery:
    """Display images in a responsive grid."""
    
    def __init__(self, images: List[str], columns: int = 4):
        with ui.grid(columns=columns):
            for img_path in images:
                with ui.card().tight():
                    ui.image(img_path).classes('w-full h-48 object-cover')
                    ui.label(Path(img_path).name).classes('text-xs')
    
    def render_clustered(self, clusters: Dict[int, List[str]]):
        """Render images grouped by cluster."""
        for cluster_id, images in clusters.items():
            cluster_name = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
            with ui.expansion(f"{cluster_name} ({len(images)} images)"):
                ImageGallery(images[:20])  # Limit to 20 per cluster
```

**Features:**
- Responsive grid layout
- Lazy loading for large galleries
- Cluster grouping with expansion panels
- Thumbnail display with filename

**Estimate:** 2 hours

### Task 3.5: Metrics & Performance Charts

**File:** `app/nicegui/components/charts.py` (NEW)

```python
class PerformanceCharts:
    """Display performance metrics and charts."""
    
    def render_timing_chart(self, telemetry: dict):
        """Bar chart of step timings."""
        # Use plotly or matplotlib
        ...
    
    def render_metrics_table(self, metrics: dict):
        """Table of image metrics."""
        # Use ag-grid or native table
        ...
```

**Estimate:** 1.5 hours

**Total for Priority 3:** ~8 hours

---

## Summary of Work

| Priority | Task | Estimate | Status |
|----------|------|----------|--------|
| **1. Feature Caching** | Database schema | 30 min | ⏳ TODO |
| | Cache service | 2 hours | ⏳ TODO |
| | Context integration | 15 min | ⏳ TODO |
| | Update existing steps | 1 hour | ⏳ TODO |
| **2. Pipeline Steps** | AVA scoring | 1 hour | ⏳ TODO |
| | Face detection & analysis | 3 hours | ⏳ TODO |
| | Face filtering | 1 hour | ⏳ TODO |
| | Face clustering | 2 hours | ⏳ TODO |
| | Selection steps | 1 hour | ⏳ TODO |
| **3. UI Enhancements** | Config panel | 2 hours | ⏳ TODO |
| | Progress display | 1.5 hours | ⏳ TODO |
| | Results tabs | 1 hour | ⏳ TODO |
| | Image gallery | 2 hours | ⏳ TODO |
| | Charts & metrics | 1.5 hours | ⏳ TODO |
| **TOTAL** | | **~20 hours** | |

---

## Implementation Order

### Sprint 1: Feature Caching (Day 1)
1. Database schema update
2. CacheService implementation
3. Context integration
4. Update existing 3 steps

**Goal:** All pipeline runs use caching - 10-100x speedup

### Sprint 2: Core Pipeline Steps (Day 2)
1. AVA aesthetic scoring
2. Face detection (MediaPipe)
3. Face analysis (pose, eyes, smile)
4. Face filtering

**Goal:** Complete face-aware pipeline available

### Sprint 3: Advanced Features (Day 3)
1. Face embeddings & clustering
2. People feature (global clustering)
3. Selection improvements

**Goal:** Full pipeline feature parity with Streamlit app

### Sprint 4: UI Polish (Day 4)
1. Rich config panel
2. Enhanced progress display
3. Multi-tab results

**Goal:** UI matches/exceeds Streamlit UX

### Sprint 5: UI Completeness (Day 5)
1. Image galleries
2. Performance charts
3. Export features

**Goal:** Production-ready UI

---

## Testing Strategy

### After Each Sprint

1. **Caching Tests:**
   - Run same album twice - verify 2nd run is 10x+ faster
   - Modify image - verify cache invalidation
   - Check cache hit rates in logs

2. **Pipeline Tests:**
   - Run on test album (50-100 images)
   - Verify all steps execute successfully
   - Check output quality

3. **UI Tests:**
   - Manual testing of all interactions
   - Verify WebSocket progress updates
   - Check responsive layout on different screen sizes

### Integration Tests

- End-to-end pipeline run with caching
- Verify results match old Streamlit app
- Performance benchmarks

---

## Success Criteria

✅ **Feature Caching:**
- 2nd run on same album is >10x faster
- Cache hit rate >90% on repeated runs
- No recomputation of unchanged images

✅ **Pipeline Completeness:**
- All 18 steps implemented and tested
- Face-aware pipeline produces quality results
- People feature identifies faces correctly

✅ **UI/UX:**
- Configuration panel as rich as Streamlit
- Real-time progress with metrics
- Gallery displays images beautifully
- Performance charts show timing data

✅ **Overall:**
- NiceGUI app provides same/better UX than Streamlit
- System is production-ready
- Documentation is complete
