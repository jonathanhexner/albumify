# Phase 2 Implementation Plan

## Current Status Summary

**Phase 1 Complete (Core Pipeline):**
- ✅ Pipeline engine (protocol, context, registry, builder, executor)
- ✅ Database models (Album, PipelineRun, PipelineResult, caching tables)
- ✅ API layer (albums, pipeline, steps, websocket routers)
- ✅ 6 core steps registered and working
- ✅ Logging infrastructure
- ✅ Caching infrastructure

**Gaps Identified:**
| Gap | Impact |
|-----|--------|
| 7 steps exist as files but aren't registered | Can't use face/portrait features |
| Face scoring steps missing (pose, eyes, smile) | Can't score face quality |
| People feature not integrated | Can't cluster faces by identity |
| ResultService/PeopleService missing | Can't retrieve detailed results |
| Frontend 80% incomplete | Unusable UI |
| 3 cache tables (redundancy) | Confusing, maintenance burden |

---

## Phase 2 Goals

1. **Register all existing steps** - Make all pipeline steps discoverable
2. **Complete face scoring steps** - Eyes open, smile, pose penalty
3. **Integrate People feature** - Cluster faces by identity
4. **Consolidate caching** - Single cache table/service
5. **Complete API services** - ResultService, PeopleService

---

## Phase 2.1: Register Missing Steps (Quick Win)

**Files that exist but aren't registered in `all_steps.py`:**

| Step File | Step Name | Action |
|-----------|-----------|--------|
| `score_ava.py` | score_ava | Add import |
| `detect_faces.py` | detect_faces | Add import |
| `extract_face_embeddings.py` | extract_face_embeddings | Add import |
| `filter_portraits.py` | filter_portraits | Add import |
| `filter_best_faces.py` | filter_best_faces | Add import |
| `cluster_people.py` | cluster_people | Add import |
| `score_face_quality.py` | score_face_quality | Add import |

**Task:** Update `sim_bench/pipeline/steps/all_steps.py` to import all step modules.

---

## Phase 2.2: Face Scoring Steps

Create pipeline steps that wrap existing face analysis code.

### 2.2.1 score_face_pose Step

**Wraps:** `sim_bench/face_pipeline/pose_estimator.py` (SixDRepNet)

```python
# sim_bench/pipeline/steps/score_face_pose.py
@register_step
class ScoreFacePoseStep(BaseStep):
    """Score face pose (yaw/pitch/roll) using SixDRepNet."""

    metadata = StepMetadata(
        name="score_face_pose",
        requires={"faces"},  # From detect_faces
        produces={"face_pose_scores"},
        depends_on=["detect_faces"]
    )

    def process(self, context, config):
        # For each face, compute frontal score (0-1)
        # Low yaw/pitch = more frontal = higher score
        pass
```

### 2.2.2 score_face_eyes Step

**Wraps:** `sim_bench/portrait_analysis/eye_state.py`

```python
# sim_bench/pipeline/steps/score_face_eyes.py
@register_step
class ScoreFaceEyesStep(BaseStep):
    """Score eye openness using MediaPipe landmarks."""

    metadata = StepMetadata(
        name="score_face_eyes",
        requires={"faces"},
        produces={"face_eyes_scores"},
        depends_on=["detect_faces"]
    )
```

### 2.2.3 score_face_smile Step

**Wraps:** `sim_bench/portrait_analysis/smile_detection.py`

```python
# sim_bench/pipeline/steps/score_face_smile.py
@register_step
class ScoreFaceSmileStep(BaseStep):
    """Score smile using MediaPipe landmarks."""

    metadata = StepMetadata(
        name="score_face_smile",
        requires={"faces"},
        produces={"face_smile_scores"},
        depends_on=["detect_faces"]
    )
```

---

## Phase 2.3: People Feature Integration

### 2.3.1 Person Database Model

Add to `sim_bench/api/database/models.py`:

```python
class Person(Base):
    """A detected person (cluster of faces) in an album."""
    __tablename__ = "people"

    id = Column(String, primary_key=True)
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=False)

    person_index = Column(Integer)  # 0, 1, 2, ...
    name = Column(String, nullable=True)  # User-assigned name

    # Thumbnail (best face)
    thumbnail_image_path = Column(String)
    thumbnail_face_index = Column(Integer)
    thumbnail_bbox = Column(JSON)

    # Statistics
    face_count = Column(Integer)
    image_count = Column(Integer)

    # All face instances for this person
    face_instances = Column(JSON)  # [{image_path, face_index, score}, ...]

    created_at = Column(DateTime, default=datetime.utcnow)

    album = relationship("Album", back_populates="people")
```

### 2.3.2 cluster_people Step

Verify and register existing `cluster_people.py`:

```python
@register_step
class ClusterPeopleStep(BaseStep):
    """Cluster faces by identity using ArcFace embeddings."""

    metadata = StepMetadata(
        name="cluster_people",
        requires={"face_embeddings"},
        produces={"people_clusters"},
        depends_on=["extract_face_embeddings"]
    )
```

### 2.3.3 select_best_per_person Step

```python
@register_step
class SelectBestPerPersonStep(BaseStep):
    """Select best face image for each person."""

    metadata = StepMetadata(
        name="select_best_per_person",
        requires={"people_clusters", "face_pose_scores", "face_eyes_scores"},
        produces={"people_thumbnails", "people_best_images"},
        depends_on=["cluster_people"]
    )
```

### 2.3.4 PeopleService

```python
class PeopleService:
    """Service for managing detected people."""

    def list_people(self, album_id: str) -> list[Person]
    def get_person(self, album_id: str, person_id: str) -> Person
    def get_person_images(self, album_id: str, person_id: str) -> list[ImageInfo]
    def rename_person(self, album_id: str, person_id: str, name: str) -> Person
    def merge_people(self, album_id: str, person_ids: list[str]) -> Person
    def split_person(self, album_id: str, person_id: str, face_ids: list[str]) -> list[Person]
```

### 2.3.5 People API Router

```
POST /api/v1/people/{album_id}         # List people in album
GET  /api/v1/people/{album_id}/{pid}   # Get person details
GET  /api/v1/people/{album_id}/{pid}/images  # Get person's images
PATCH /api/v1/people/{album_id}/{pid}  # Rename person
POST /api/v1/people/{album_id}/merge   # Merge people
POST /api/v1/people/{album_id}/split   # Split person
```

---

## Phase 2.4: Consolidate Caching

### Current State (3 tables)
1. `EmbeddingCache` - Legacy, embedding-only
2. `FeatureCache` - General purpose
3. `UniversalCache` - Latest approach

### Decision: Use `UniversalCache`

The `UniversalCache` table with `UniversalCacheHandler` is the most flexible:
- Stores any serializable data as bytes
- Has proper mtime tracking
- Already integrated into `PipelineContext`

### Actions:
1. Mark `EmbeddingCache` and `FeatureCache` as deprecated in code comments
2. Update documentation to reference `UniversalCache` only
3. Ensure all steps use `context.cache_handler`

---

## Phase 2.5: Complete API Services

### ResultService

```python
class ResultService:
    """Service for pipeline results."""

    def get_result(self, job_id: str) -> PipelineResult
    def get_images(self, job_id: str, cluster_id: int = None) -> list[ImageInfo]
    def get_metrics(self, job_id: str) -> dict
    def export_results(self, job_id: str, output_path: str, format: str) -> bool
```

### Results Router

```
GET /api/v1/results/{job_id}           # Get full result
GET /api/v1/results/{job_id}/images    # Get images (optionally by cluster)
GET /api/v1/results/{job_id}/metrics   # Get pipeline metrics
POST /api/v1/results/{job_id}/export   # Export to filesystem
```

---

## Implementation Order

### Sprint 1 (Quick Wins) - COMPLETE ✅
1. ✅ Register 7 missing steps in `all_steps.py`
2. ✅ Test that all steps are discoverable via `/api/v1/steps`
3. ✅ Run default pipeline end-to-end

### Sprint 2 (Face Scoring) - COMPLETE ✅
1. ✅ Create `score_face_pose.py` step (SixDRepNet for head pose)
2. ✅ Create `score_face_eyes.py` step (MediaPipe for eye openness)
3. ✅ Create `score_face_smile.py` step (MediaPipe for smile detection)
4. ✅ Register and test face scoring steps (16 total steps now)

### Sprint 3 (People Feature) - COMPLETE ✅
1. ✅ Add `Person` model to database (with Album/PipelineRun relationships)
2. ✅ Verify `cluster_people` step (already working)
3. ✅ Create `select_best_per_person` step (quality-based face selection)
4. ✅ Implement `PeopleService` (list, get, rename, merge, split)
5. ✅ Implement `/api/v1/people` router (6 endpoints)

### Sprint 4 (API Completion) - COMPLETE ✅
1. ✅ Implement `ResultService` (get, list, metrics, clusters, images, export)
2. ✅ Implement `/api/v1/results` router (6 endpoints)
3. ✅ Add export functionality (copy/symlink, organize by cluster/person)

### Sprint 5 (Cleanup) - COMPLETE ✅
1. ✅ Deleted old cache tables (`EmbeddingCache`, `FeatureCache` models)
2. ✅ Deleted unused `cache_service.py` (pipeline uses `UniversalCacheHandler`)
3. ✅ Single cache: `UniversalCache` with `UniversalCacheHandler`

---

## Default Pipelines

After Phase 2, these preset pipelines should work:

### Scene Organization (Current)
```python
["discover_images", "score_iqa", "filter_quality",
 "extract_scene_embedding", "cluster_scenes", "select_best"]
```

### Portrait Organization (New)
```python
["discover_images", "score_iqa", "filter_quality",
 "detect_faces", "filter_portraits",
 "score_face_pose", "score_face_eyes", "score_face_smile",
 "extract_face_embeddings", "cluster_people", "select_best_per_person"]
```

### Full Album Organization (New)
```python
["discover_images", "score_iqa", "filter_quality",
 "detect_faces",
 "extract_scene_embedding", "cluster_scenes",
 "score_face_pose", "score_face_eyes",
 "extract_face_embeddings", "cluster_people",
 "select_best", "select_best_per_person"]
```

---

## Verification

After Phase 2 completion:

1. **API Test:**
   ```bash
   # List all steps (should show 15+ steps)
   curl http://localhost:8000/api/v1/steps | jq '.[] | .name'

   # Run portrait pipeline
   curl -X POST http://localhost:8000/api/v1/pipeline/run \
     -d '{"album_id": "...", "steps": ["discover_images", "detect_faces", ...]}'
   ```

2. **People Test:**
   ```bash
   # List detected people
   curl http://localhost:8000/api/v1/people/{album_id}

   # Rename a person
   curl -X PATCH http://localhost:8000/api/v1/people/{album_id}/{person_id} \
     -d '{"name": "Mom"}'
   ```

3. **Cache Test:**
   - Run pipeline twice on same album
   - Second run should be 10-100x faster
   - Check cache stats in logs

---

## Phase 3: Frontend (NiceGUI)

### Implemented Features ✅

**API Client (`app/nicegui/api_client.py`):**
- ✅ Results API: list_results, get_result, get_result_images, get_result_clusters, get_result_metrics, export_result
- ✅ People API: list_people, get_person, get_person_images, rename_person, merge_people, split_person

**Pages (`app/nicegui/main.py`):**
- ✅ Home page (`/`) - Album selection, pipeline configuration, run pipeline with WebSocket progress
- ✅ Results page (`/results`) - Summary stats, selected images, cluster gallery, export functionality
- ✅ People page (`/people`) - People grid with thumbnails, rename, merge selection
- ✅ Person detail page (`/person/{album_id}/{person_id}`) - All images of a person

**Export Options:**
- Include selected / all filtered
- Organize by cluster / person
- Copy or symlink mode
