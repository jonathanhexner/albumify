# Score Persistence Debug & Fix Plan

**Created**: 2026-02-04
**Status**: 🔍 Investigation Required

## Problem Summary

From cluster debug view, we observe:
- ✅ IQA scores are displaying correctly
- ✅ AVA scores are displaying correctly  
- ✅ Sharpness scores are displaying correctly
- ✅ Face count is present (though accuracy needs tuning)
- ❌ **Pose scores are None**
- ❌ **Eyes scores are None**
- ❌ **Smile scores are None**
- ❌ **Final Score (composite_score) is None**
- ❌ **People column is empty**

## Root Cause Hypothesis

The face analysis pipeline steps (`score_face_pose`, `score_face_eyes`, `score_face_smile`) are:
1. ✅ Computing scores (code exists, uses caching)
2. ✅ Storing in `PipelineContext` (face objects updated)
3. ❌ **NOT being persisted to database** (`image_metrics` JSON in `PipelineResult`)

The `composite_score` is:
1. ⚠️ Possibly computed during `select_best` step
2. ❌ **NOT being persisted to `image_metrics`**

The People data is:
1. ✅ Stored in `Person` table
2. ⚠️ Possibly missing display names or proper query structure

---

## Investigation Plan

### Phase 1: Trace Data Flow (30 min)

**Goal**: Understand where scores go from pipeline → database → API → frontend.

#### 1.1 Check Pipeline Context → Database Storage

**Files to examine**:
- `sim_bench/api/services/pipeline_service.py` - Where context is converted to `PipelineResult`
- `sim_bench/pipeline/executor.py` - Final context after all steps run

**Questions**:
- Does `pipeline_service.py` extract face scores from context when saving results?
- What fields from context are saved to `result.image_metrics`?
- Are face scores (pose/eyes/smile) being included?

**Action**:
```bash
# Search for where image_metrics is built
grep -n "image_metrics" sim_bench/api/services/pipeline_service.py -A 10 -B 5
```

#### 1.2 Check Face Score Steps Output

**Files to examine**:
- `sim_bench/pipeline/steps/score_face_pose.py` - Lines 139-169 (`_store_results`)
- `sim_bench/pipeline/steps/score_face_eyes.py` - Similar method
- `sim_bench/pipeline/steps/score_face_smile.py` - Lines 147-202 (`_store_results`)

**Questions**:
- Do these steps store scores in `context.face_pose_scores`, `context.face_eyes_scores`, `context.face_smile_scores`?
- Are these context fields defined in `sim_bench/pipeline/context.py`?

**Action**:
```bash
# Check what face score fields exist in PipelineContext
grep -n "face_pose_scores\|face_eyes_scores\|face_smile_scores" sim_bench/pipeline/context.py
```

#### 1.3 Check Composite Score Computation

**Files to examine**:
- `sim_bench/pipeline/steps/select_best.py` - Where images are ranked and selected
- `sim_bench/pipeline/context.py` - `get_image_score()` method (lines 80-99)

**Questions**:
- When is `composite_score` computed for each image?
- Is it stored back in context or only used transiently for ranking?
- Does `select_best` save composite scores anywhere?

**Action**:
```bash
# Search for composite_score in select_best
grep -rn "composite_score" sim_bench/pipeline/steps/select_best.py
```

#### 1.4 Check Person Data Flow

**Files to examine**:
- `sim_bench/pipeline/steps/identify_people.py` - Where Person records are created
- `sim_bench/api/services/result_service.py` - Lines 145-190 (`get_clusters()`)

**Questions**:
- Are Person records being created with proper `name` or just `person_index`?
- Is `get_clusters()` correctly building the `image_people` map?
- Are there Person records but with null/empty names?

**Action**:
```bash
# Check database directly
sqlite3 data/albums.db "SELECT id, person_index, name, face_count FROM people LIMIT 10;"
```

---

### Phase 2: Fix Pipeline → Database Storage (1-2 hours)

**Priority**: HIGH (Fixes Pose/Eyes/Smile/Final Score issues)

#### 2.1 Update `pipeline_service.py` to Extract Face Scores

**File**: `sim_bench/api/services/pipeline_service.py`

**Current state** (likely):
```python
# Only extracts IQA, AVA, sharpness, face_count
image_metrics[path] = {
    'iqa_score': context.iqa_scores.get(path),
    'ava_score': context.ava_scores.get(path),
    'sharpness': context.sharpness_scores.get(path),
    'face_count': len(context.faces.get(path, [])),
    'cluster_id': cluster_id,
}
```

**Fix needed**:
```python
# Add face quality scores for each image
def _build_image_metrics(context: PipelineContext, path: str, cluster_id: int) -> dict:
    """Build complete metrics dict for an image."""
    faces = context.faces.get(path, [])
    
    # Aggregate face scores (average across all faces in image)
    face_pose_scores = []
    face_eyes_scores = []
    face_smile_scores = []
    
    for face in faces:
        cache_key = f"{face.original_path}:face_{face.face_index}"
        
        if cache_key in context.face_pose_scores:
            face_pose_scores.append(context.face_pose_scores[cache_key])
        
        if cache_key in context.face_eyes_scores:
            face_eyes_scores.append(context.face_eyes_scores[cache_key])
        
        if cache_key in context.face_smile_scores:
            face_smile_scores.append(context.face_smile_scores[cache_key])
    
    return {
        'iqa_score': context.iqa_scores.get(path),
        'ava_score': context.ava_scores.get(path),
        'sharpness': context.sharpness_scores.get(path),
        'face_count': len(faces),
        'cluster_id': cluster_id,
        'is_selected': path in context.selected_images,
        
        # Face quality scores (average if multiple faces)
        'face_pose_scores': (
            sum(face_pose_scores) / len(face_pose_scores)
            if face_pose_scores else None
        ),
        'face_eyes_scores': (
            sum(face_eyes_scores) / len(face_eyes_scores)
            if face_eyes_scores else None
        ),
        'face_smile_scores': (
            sum(face_smile_scores) / len(face_smile_scores)
            if face_smile_scores else None
        ),
        
        # Composite score (if computed)
        'composite_score': context.composite_scores.get(path),
    }
```

**Verification**:
1. Run pipeline on small test album (10 images)
2. Check database: `SELECT path, face_pose_scores, face_eyes_scores FROM ...`
3. Confirm scores are non-null for images with faces

#### 2.2 Compute Composite Score in `select_best` Step

**File**: `sim_bench/pipeline/steps/select_best.py`

**Current state**: Composite score is computed transiently during ranking but not stored.

**Fix needed**: After ranking images, store composite scores in context:

```python
def process(self, context: PipelineContext, config: dict) -> None:
    """Select best images from each cluster."""
    
    # ... existing ranking logic ...
    
    # NEW: Store composite scores in context for persistence
    if not hasattr(context, 'composite_scores'):
        context.composite_scores = {}
    
    for cluster_id, images in context.scene_clusters.items():
        for image_path in images:
            score = self._compute_composite_score(
                image_path, context, config
            )
            context.composite_scores[image_path] = score
    
    # ... rest of selection logic ...
```

**Verification**:
1. Add debug logging in `select_best.py`: `logger.info(f"Composite scores: {context.composite_scores}")`
2. Check if scores appear in logs
3. Verify they're persisted to database via Phase 2.1

#### 2.3 Add Missing Fields to `PipelineContext`

**File**: `sim_bench/pipeline/context.py`

**Check if these fields exist** (around line 40-60):
```python
@dataclass
class PipelineContext:
    # ... existing fields ...
    
    face_pose_scores: dict[str, float] = field(default_factory=dict)  # ← Should exist
    face_eyes_scores: dict[str, float] = field(default_factory=dict)  # ← Should exist
    face_smile_scores: dict[str, float] = field(default_factory=dict) # ← Should exist
    
    composite_scores: dict[str, float] = field(default_factory=dict)  # ← ADD IF MISSING
```

---

### Phase 3: People Management Feature (1-2 hours)

**Priority**: HIGH (User-requested feature)

**User Goal**: 
1. Identify all different people in the album (face clustering) ✅ Already implemented
2. Allow user to name them (UI feature) ❌ **Need to implement**
3. Filter images by people (e.g., "Show me all photos with Mom") ❌ **Need to implement**

#### 3.1 Check Person Records in Database

**Action**:
```bash
# Check if Person records exist
sqlite3 data/albums.db "
SELECT 
    id, 
    person_index, 
    name, 
    face_count,
    image_count,
    thumbnail_image_path
FROM people 
WHERE run_id = '<latest_run_id>'
ORDER BY face_count DESC
LIMIT 20;
"
```

**Expected output**:
- Multiple Person records with `person_index` = 0, 1, 2, ...
- `name` is NULL (users haven't assigned names yet)
- `face_count` > 0 for each person
- `thumbnail_image_path` points to representative face crop

#### 3.2 Create People Gallery Page (NEW FEATURE)

**File**: `app/streamlit/pages/people.py` (CREATE NEW)

**Purpose**: Display all identified people in the album with naming interface

```python
"""People management page - view, name, and filter by people."""

import streamlit as st
from app.streamlit.api_client import APIClient

def render_people_page():
    """Render the people gallery and management interface."""
    st.title("People in Album")
    
    client = APIClient()
    job_id = st.session_state.get("current_job_id")
    
    if not job_id:
        st.warning("No pipeline results available. Run a pipeline first.")
        return
    
    # Fetch all people
    people = client.get_people(job_id)
    
    if not people:
        st.info("No people detected in this album.")
        return
    
    st.write(f"Found {len(people)} people")
    
    # Display people in grid
    cols = st.columns(4)
    
    for idx, person in enumerate(people):
        with cols[idx % 4]:
            # Show thumbnail
            if person['thumbnail_path']:
                st.image(person['thumbnail_path'], width=150)
            
            # Show current name or placeholder
            current_name = person['name'] or f"Person {person['person_index'] + 1}"
            st.caption(f"{person['face_count']} faces in {person['image_count']} images")
            
            # Name input
            new_name = st.text_input(
                "Name",
                value=person['name'] or "",
                key=f"person_name_{person['id']}",
                placeholder="Enter name..."
            )
            
            # Update button
            if st.button("Update", key=f"update_{person['id']}"):
                client.update_person_name(person['id'], new_name)
                st.success(f"Updated to: {new_name}")
                st.rerun()
            
            # View all images button
            if st.button("View Images", key=f"view_{person['id']}"):
                st.session_state['filter_person_id'] = person['id']
                st.switch_page("pages/results.py")

if __name__ == "__main__":
    render_people_page()
```

#### 3.3 Add People Filter to Results Page

**File**: `app/streamlit/pages/results.py` (MODIFY)

**Add filter sidebar**:
```python
# In render_results_page(), add sidebar filters
with st.sidebar:
    st.subheader("Filters")
    
    # Get all people for this job
    people = client.get_people(job_id)
    
    if people:
        person_options = ["All People"] + [
            p['name'] or f"Person {p['person_index'] + 1}"
            for p in people
        ]
        selected_person = st.selectbox("Filter by Person", person_options)
        
        if selected_person != "All People":
            # Find person ID
            person_idx = person_options.index(selected_person) - 1
            person_id = people[person_idx]['id']
            
            # Filter images
            images = client.get_images(job_id, person_id=person_id)
            st.write(f"Showing {len(images)} images with {selected_person}")
```

#### 3.4 Backend API Endpoints (EXTEND EXISTING)

**File**: `sim_bench/api/routes/results.py` (MODIFY)

**Add endpoints**:
```python
@router.get("/results/{job_id}/people")
def get_people(job_id: str, db: Session = Depends(get_db)):
    """Get all identified people for a job."""
    service = ResultService(db)
    return service.get_people(job_id)

@router.patch("/people/{person_id}/name")
def update_person_name(
    person_id: str,
    name: str,
    db: Session = Depends(get_db)
):
    """Update a person's name."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(404, "Person not found")
    
    person.name = name
    db.commit()
    
    return {"id": person_id, "name": name}
```

**File**: `sim_bench/api/services/result_service.py` (ADD METHOD)

```python
def get_people(self, job_id: str) -> list[dict]:
    """Get all people identified in a pipeline run."""
    people = (
        self._session.query(Person)
        .filter(Person.run_id == job_id)
        .order_by(Person.face_count.desc())  # Most frequent people first
        .all()
    )
    
    return [
        {
            'id': p.id,
            'person_index': p.person_index,
            'name': p.name,
            'face_count': p.face_count,
            'image_count': p.image_count,
            'thumbnail_path': p.thumbnail_image_path,
            'thumbnail_bbox': p.thumbnail_bbox,
        }
        for p in people
    ]
```

#### 3.5 Frontend API Client (EXTEND)

**File**: `app/streamlit/api_client.py` (ADD METHODS)

```python
def get_people(self, job_id: str) -> list[dict]:
    """Get all people for a job."""
    response = requests.get(f"{self.base_url}/results/{job_id}/people")
    response.raise_for_status()
    return response.json()

def update_person_name(self, person_id: str, name: str) -> dict:
    """Update a person's name."""
    response = requests.patch(
        f"{self.base_url}/people/{person_id}/name",
        params={"name": name}
    )
    response.raise_for_status()
    return response.json()
```

#### 3.6 Fix Cluster View People Display

**File**: `app/streamlit/components/gallery.py` (line ~55, `_render_cluster_score_table`)

**Current**: "People" column shows empty string

**Fix**: Ensure people labels are displayed:
```python
# In _render_cluster_score_table
people_names = cluster.person_labels.get(img.path, [])
people_str = ", ".join(people_names) if people_names else "-"
```

**Also verify** `result_service.py` is correctly building `person_labels`:
```python
# In get_clusters()
image_people = defaultdict(list)
for person in people:
    # Use name if set, otherwise Person 1, Person 2, etc.
    display_name = person.name if person.name else f"Person {person.person_index + 1}"
    for face in (person.face_instances or []):
        path = face.get('image_path')
        if path:
            image_people[path].append(display_name)
```

---

### Phase 4: Tune Face Detection Threshold (15 min)

**Priority**: MEDIUM (Missing real faces)

**File**: `sim_bench/pipeline/steps/detect_faces.py`

**Current threshold** (likely): `min_detection_confidence=0.5`

**Issue**: False negatives (missing real faces) - User suspects we're being too strict

**Fix options**:
1. **Decrease threshold** → Catch more real faces, but might add some false positives
   ```python
   min_detection_confidence=0.3  # More lenient (recommended)
   # OR
   min_detection_confidence=0.4  # Moderate
   ```

2. **Add size filter** → Still reject tiny faces (likely artifacts) even with lower threshold
   ```python
   min_face_size = 0.02  # Face must be >2% of image area
   max_face_size = 0.95  # Reject if "face" is almost entire image
   ```

3. **Post-processing validation** → Use face quality scorer to filter out poor detections
   ```python
   # After detection, reject faces with very low quality scores
   min_overall_quality = 0.2
   ```

**Recommended**: Start with option 1 (decrease threshold to 0.3) + option 2 (size filtering) to catch more faces while still avoiding obvious false positives.

---

## Testing Strategy

### Test Album
Create small test album: **10 images** with:
- 3 images with clear frontal faces (smiling, eyes open)
- 3 images with faces at angles (pose variation)
- 2 images with faces but eyes closed
- 2 images with no faces (landscapes)

### Test Sequence

1. **Run pipeline** with all steps:
   ```bash
   # Start backend
   cd sim_bench
   python -m sim_bench.api.main
   
   # In another terminal, trigger pipeline via API
   curl -X POST http://localhost:8000/api/pipeline/run \
     -H "Content-Type: application/json" \
     -d '{"album_id": "test_album_123"}'
   ```

2. **Check database** after completion:
   ```sql
   -- Check image_metrics structure
   SELECT 
     json_extract(value, '$.path') as path,
     json_extract(value, '$.face_pose_scores') as pose,
     json_extract(value, '$.face_eyes_scores') as eyes,
     json_extract(value, '$.face_smile_scores') as smile,
     json_extract(value, '$.composite_score') as final_score
   FROM 
     pipeline_results,
     json_each(image_metrics)
   WHERE run_id = '<test_run_id>'
   LIMIT 10;
   ```

3. **Check API response**:
   ```bash
   curl http://localhost:8000/api/results/<test_run_id>/clusters | jq
   ```

4. **Check Streamlit UI**:
   - Open cluster debug view
   - Verify all score columns populated
   - Verify People column shows "Person 1", "Person 2", etc.

---

## Success Criteria

### Sprint 1 (Score Display)
✅ **Pose, Eyes, Smile columns** show numeric values (0.0 - 1.0) for images with faces
✅ **Final Score column** shows numeric values (0.0 - 1.0) for all images
✅ **People column** shows "Person 1", "Person 2", etc. for images with identified faces

### Sprint 2 (People Management)
✅ **People Gallery page** exists and displays all identified people
✅ **Naming interface** allows user to assign names to people
✅ **Filter by person** in Results page works correctly
✅ **Named people** appear in cluster debug table (e.g., "Mom" instead of "Person 1")

### Sprint 3 (Face Detection)
✅ **More faces detected** (fewer false negatives)
✅ **Face count accuracy** improved while maintaining reasonable precision

---

## Implementation Order

### Sprint 1: Fix Score Display (HIGH PRIORITY - 1.5 hours)
1. **Phase 2.3** - Add `composite_scores` field to context (2 min)
2. **Phase 2.1** - Update `pipeline_service.py` to extract all scores (30 min)
3. **Phase 2.2** - Compute and store composite scores in `select_best` (30 min)
4. **Phase 3.6** - Fix People display in cluster debug table (10 min)
5. **Phase 1** - Run test pipeline and verify (20 min)

### Sprint 2: People Management UI (MEDIUM PRIORITY - 2 hours)
6. **Phase 3.4** - Add backend API endpoints for people (30 min)
7. **Phase 3.5** - Extend frontend API client (15 min)
8. **Phase 3.2** - Create People Gallery page (45 min)
9. **Phase 3.3** - Add People filter to Results page (30 min)

### Sprint 3: Face Detection Tuning (LOW PRIORITY - 30 min)
10. **Phase 4** - Decrease detection threshold to catch more faces (15 min)
11. **Test and tune** - Run on full album, adjust threshold as needed (15 min)

**Total estimated time**: 4-5 hours (can be split across multiple sessions)

---

## Files to Modify

### Sprint 1: Score Display (HIGH PRIORITY)
1. `sim_bench/pipeline/context.py` - Add composite_scores field
2. `sim_bench/api/services/pipeline_service.py` - Extract all scores to database
3. `sim_bench/pipeline/steps/select_best.py` - Store composite scores
4. `sim_bench/api/services/result_service.py` - Verify people display logic

### Sprint 2: People Management (MEDIUM PRIORITY)
5. `app/streamlit/pages/people.py` - **NEW FILE** - People gallery page
6. `sim_bench/api/routes/results.py` - Add people endpoints
7. `sim_bench/api/services/result_service.py` - Add get_people() method
8. `app/streamlit/api_client.py` - Add people API methods
9. `app/streamlit/pages/results.py` - Add people filter sidebar
10. `app/streamlit/components/gallery.py` - Fix people column display

### Sprint 3: Face Detection (LOW PRIORITY)
11. `sim_bench/pipeline/steps/detect_faces.py` - Decrease detection threshold

---

## Rollback Plan

If fixes cause issues:
1. Git: `git stash` to save changes
2. Restart backend with previous version
3. Check `CHANGES_LOG.md` for exact files modified
4. Use `git diff` to review changes before committing

---

## Next Steps After Fix

1. Run on full Budapest album to verify at scale
2. Document score computation in `docs/architecture/SCORE_COMPUTATION.md`
3. Add unit tests for `_build_image_metrics()` helper
4. Consider adding score validation (e.g., assert all scores in [0, 1] range)

---

## Questions for User

Before starting implementation:
1. **Start with Sprint 1** (fix score display)? This is the quickest win (~1.5 hours)
2. **Test album**: Create small 10-image test set, or use full Budapest album?
3. **Face detection threshold**: Try 0.3 (catch more faces) or 0.4 (balanced)?
4. **People feature scope**: Implement full People Gallery (Sprint 2) now, or after Sprint 1?

## Recommended Approach

**Option A - Quick Win First** (Recommended):
1. Sprint 1 today (1.5 hrs) → See all scores populated
2. Test on Budapest album
3. Sprint 2 tomorrow (2 hrs) → Full people management UI
4. Sprint 3 as needed → Tune face detection

**Option B - Complete Feature**:
1. All 3 sprints in one session (4-5 hrs)
2. Complete end-to-end people workflow
3. Thorough testing after everything is done
