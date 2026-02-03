# Current vs Target State

## Visual Comparison

### Streamlit UI (Target Experience)

```
┌─────────────────────────────────────────────────────────────────┐
│ 📸 Photo Album Organization                         [STREAMLIT] │
│ Automatically organize and select best photos from your albums  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌────────────────────────────────────┐ │
│  │ SIDEBAR          │  │ MAIN CONTENT                       │ │
│  │                  │  │                                    │ │
│  │ Navigation       │  │ 📂 Album Selection                 │ │
│  │ 1. Configure     │  │ ┌────────────────────────────────┐ │
│  │ 2. Select album  │  │ │ Source Directory: [________]   │ │
│  │ 3. Run workflow  │  │ │ Album Name: [________]         │ │
│  │ 4. View results  │  │ │ Output Directory: [________]   │ │
│  │ ──────────────   │  │ └────────────────────────────────┘ │
│  │                  │  │                                    │ │
│  │ ⚙️ Configuration  │  │ ──────────────────────────────────│
│  │                  │  │                                    │ │
│  │ ▼ Quality        │  │ 🚀 Workflow Execution              │ │
│  │   Thresholds     │  │ Album: Summer 2024                 │ │
│  │   Min IQA [===]  │  │ Source: C:/Photos/...              │ │
│  │   Min AVA [===]  │  │                                    │ │
│  │   Sharpness [==] │  │ [▶️ Start Workflow]                │ │
│  │                  │  │                                    │ │
│  │ ▼ Portrait       │  │ 🔍 Discovering images              │ │
│  │   Preferences    │  │ ████████████████░░░░ 80%          │ │
│  │   ☑ Eyes open    │  │ 📄 Processing IMG_1234.jpg        │ │
│  │   ☑ Prefer smile │  │                                    │ │
│  │   Smile [====]   │  │ ┌─────┬─────┬──────┬──────┐      │ │
│  │   Eyes [====]    │  │ │ 150 │ 8.5 │ 45s  │ 2m   │      │ │
│  │                  │  │ │ img │img/s│elapse│ ETA  │      │ │
│  │ ▶ Selection      │  │ └─────┴─────┴──────┴──────┘      │ │
│  │   Weights        │  │                                    │ │
│  │                  │  │ ──────────────────────────────────│
│  │ ▶ Clustering     │  │                                    │ │
│  │                  │  │ 📸 Results                         │ │
│  │ ▶ Performance    │  │ ✅ Selected 45 best images        │ │
│  │                  │  │    from 12 clusters                │ │
│  │ ▶ Export         │  │                                    │ │
│  │                  │  │ [🖼️Gallery][📊Metrics][⚡Perf][📤] │
│  │ ──────────────   │  │                                    │ │
│  │                  │  │ 🖼️ Gallery View                   │ │
│  │ About            │  │ ▼ Cluster 1 (8 images)            │ │
│  │ Uses: IQA, AVA,  │  │   [img] [img] [img] [img]         │ │
│  │ MediaPipe, etc   │  │   [img] [img] [img] [img]         │ │
│  │                  │  │                                    │ │
│  │                  │  │ ▼ Cluster 2 (5 images)            │ │
│  └──────────────────┘  │   [img] [img] [img] [img] [img]   │ │
│                         └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- ✅ Rich sidebar with collapsible sections
- ✅ Real-time progress with 4 metrics (processed, rate, elapsed, ETA)
- ✅ Visual stage indicators with emojis
- ✅ Multi-tab results (Gallery, Metrics, Performance, Export)
- ✅ Clustered gallery view with expansion panels
- ✅ Configuration validation (weight sums)
- ✅ Export download buttons

---

### Current NiceGUI UI (What We Have)

```
┌─────────────────────────────────────────────────────────────────┐
│ Album Organizer                          [Home] [Results]       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Album Organization                                              │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Select or Create Album                                      │ │
│ │                                                             │ │
│ │ Select Album: [Dropdown ▼]                                 │ │
│ │ ─────────────────────────────────────────────────────────  │ │
│ │                                                             │ │
│ │ Album Name: [________]  Source Path: [________]            │ │
│ │ [Create Album]                                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Pipeline Configuration                                      │ │
│ │                                                             │ │
│ │ ☐ Discover Images (analysis)                               │ │
│ │ ☑ Score IQA (analysis)                                     │ │
│ │ ☑ Filter Quality (filtering)                               │ │
│ │ ☑ Extract Scene Embedding (embedding)                      │ │
│ │ ☑ Cluster Scenes (clustering)                              │ │
│ │ ☑ Select Best (selection)                                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Starting pipeline...                                            │
│ ████████████░░░░░░░░ 60%                                       │
│                                                                 │
│ [Run Pipeline]                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**What's Missing:**
- ❌ No sidebar navigation
- ❌ No configuration sliders/inputs (only step checkboxes)
- ❌ No collapsible sections
- ❌ Progress is minimal (just bar + text)
- ❌ No real-time metrics (rate, ETA, etc.)
- ❌ No visual stage descriptions
- ❌ Results page is basic (no tabs, galleries)
- ❌ No cluster visualization
- ❌ No performance charts

---

## Pipeline Steps Comparison

### Streamlit (Full Pipeline)

```
discover_images
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 ▼
score_iqa       score_ava       detect_faces
    │                 │                 │
    │                 │                 ├─> score_face_pose
    │                 │                 ├─> score_face_eyes
    │                 │                 └─> score_face_smile
    │                 │                 
    └─────────────────┴─────────────────┘
                      │
                      ▼
            filter_quality
                      │
                      ├─> filter_portrait
                      ├─> filter_pose
                      │
                      ▼
          extract_scene_embedding
                      │
                      ▼
             cluster_scenes
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    Face-dominant         Non-face images
           │                     │
           ▼                     │
  extract_face_embedding         │
           │                     │
           ▼                     │
    cluster_faces                │
    cluster_people               │
           │                     │
           ▼                     │
  select_best_per_identity       │
  select_best_per_person         │
           │                     │
           └──────────┬──────────┘
                      ▼
              Final Selection
```

**Total Steps:** 18

---

### Current NiceGUI (Phase 1 Only)

```
discover_images
    │
    ▼
score_iqa
    │
    ▼
filter_quality
    │
    ▼
extract_scene_embedding
    │
    ▼
cluster_scenes
    │
    ▼
select_best
```

**Total Steps:** 6 (33% complete)

**Missing:**
- ❌ AVA aesthetic scoring
- ❌ Face detection
- ❌ Face analysis (pose, eyes, smile)
- ❌ Portrait filtering
- ❌ Face embeddings
- ❌ Face clustering
- ❌ People feature
- ❌ Advanced selection

---

## Feature Caching Comparison

### Streamlit (Has Caching)

```python
# Cached workflow run
Run 1: 100 images
├─ score_iqa: 45s (compute all)
├─ extract_embedding: 120s (compute all)
└─ Total: 165s

Run 2: Same 100 images, different config
├─ score_iqa: 2s (cache hit: 100%)
├─ extract_embedding: 3s (cache hit: 100%)
└─ Total: 5s (33x faster!)

Run 3: 100 images + 10 new
├─ score_iqa: 6s (cache hit: 90%, compute 10)
├─ extract_embedding: 15s (cache hit: 90%, compute 10)
└─ Total: 21s (8x faster!)
```

**Cache Storage:**
- Thumbnails: `cache/album_analysis/medium/*.jpg`
- Features: In-memory or database

---

### Current NiceGUI (No Caching)

```python
# Every run recomputes everything
Run 1: 100 images
├─ score_iqa: 45s
├─ extract_embedding: 120s
└─ Total: 165s

Run 2: SAME 100 images, different config
├─ score_iqa: 45s (recomputed!)
├─ extract_embedding: 120s (recomputed!)
└─ Total: 165s (no speedup)

Run 3: 100 images + 10 new
├─ score_iqa: 50s (all 110 recomputed!)
├─ extract_embedding: 132s (all 110 recomputed!)
└─ Total: 182s (slower!)
```

**Result:** Every experiment takes full time, making iteration painfully slow.

---

## Database Comparison

### Current Schema

```sql
-- ✅ Implemented
Album(id, name, source_path, image_count, created_at)
PipelineRun(id, album_id, config, status, progress, timestamps)
PipelineResult(id, run_id, stats, clusters, selected_images, telemetry)

-- ❌ Missing
FeatureCache(image_path, feature_type, model_name, value, mtime)
Person(id, album_id, name, thumbnail, face_count)
```

**Impact:** No persistence of computed features = slow iteration

---

### Target Schema

```sql
-- ✅ Already have
Album(...)
PipelineRun(...)
PipelineResult(...)

-- 🎯 Need to add
FeatureCache(
    image_path TEXT,
    feature_type TEXT,      -- 'scene_embedding', 'iqa_score', 'face_detection'
    model_name TEXT,        -- 'dinov2', 'pyiqa', 'mediapipe'
    value_float REAL,       -- For scores
    value_vector BLOB,      -- For embeddings
    value_json TEXT,        -- For structured data
    image_mtime REAL,       -- File modification time
    UNIQUE(image_path, feature_type, model_name)
)

Person(
    id TEXT PRIMARY KEY,
    album_id TEXT,
    name TEXT,              -- User-assigned name
    thumbnail_path TEXT,
    face_count INTEGER,
    face_instances JSON
)
```

**Benefit:** Persistent cache survives app restarts, shared across pipeline runs

---

## Performance Impact

### Without Caching (Current)

| Operation | First Run | Second Run | Third Run |
|-----------|-----------|------------|-----------|
| IQA scoring (100 images) | 45s | 45s | 45s |
| DINOv2 embeddings (100) | 120s | 120s | 120s |
| Face detection (100) | 60s | 60s | 60s |
| **Total** | **225s** | **225s** | **225s** |

**Development workflow:**
- Try config A: 225s
- Try config B: 225s (wasted 225s recomputing)
- Try config C: 225s (wasted 450s total)
- **Time to experiment with 3 configs: 11.25 minutes**

---

### With Caching (Target)

| Operation | First Run | Second Run | Third Run |
|-----------|-----------|------------|-----------|
| IQA scoring (100 images) | 45s | 0.5s | 0.5s |
| DINOv2 embeddings (100) | 120s | 1.0s | 1.0s |
| Face detection (100) | 60s | 0.8s | 0.8s |
| **Total** | **225s** | **2.3s** | **2.3s** |

**Development workflow:**
- Try config A: 225s (first time)
- Try config B: 2.3s (97x faster!)
- Try config C: 2.3s (97x faster!)
- **Time to experiment with 3 configs: 229.6s = 3.8 minutes**

**Speedup: 3x faster for iterative development**

---

## Summary

| Aspect | Current | Target | Gap |
|--------|---------|--------|-----|
| **Pipeline Steps** | 6 steps | 18 steps | 12 missing (67%) |
| **UI Richness** | Basic | Rich (Streamlit level) | Missing sidebar, metrics, tabs |
| **Feature Caching** | None | Database-backed | 10-100x speedup needed |
| **Face Processing** | None | Full pipeline | Completely missing |
| **People Feature** | None | Google Photos-style | Not implemented |
| **Progress Display** | Basic bar | 4 real-time metrics | Missing rate/ETA |
| **Results View** | Single page | Multi-tab with gallery | Missing organization |
| **Configuration** | Step checkboxes | Rich sliders/controls | Minimal controls |

**Overall Completion: ~30%**

**Critical Path:**
1. **Caching** (blocks fast iteration)
2. **Pipeline Steps** (blocks feature parity)
3. **UI Polish** (blocks user experience parity)
