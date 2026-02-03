# Streamlit Frontend Design

## Overview

A pure frontend Streamlit application that communicates exclusively with the FastAPI backend via HTTP. Mirrors the look and feel of the existing `app/album/` app but adds multi-page navigation and people-based image browsing.

**Key Principle:** Streamlit is ONLY a UI layer. All business logic lives in the FastAPI backend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Home   │  │ Results │  │ People  │  │ Albums  │        │
│  │  Page   │  │  Page   │  │  Page   │  │  Page   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┴─────┬──────┴────────────┘              │
│                          │                                   │
│                   ┌──────┴──────┐                           │
│                   │  APIClient  │                           │
│                   │  (HTTP)     │                           │
│                   └──────┬──────┘                           │
└──────────────────────────┼──────────────────────────────────┘
                           │ HTTP/REST
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                            │
│  /api/v1/albums, /api/v1/pipeline, /api/v1/results, etc.     │
└──────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
app/streamlit/
├── __init__.py                # Package exports
├── main.py                    # Entry point, page routing, CSS
├── api_client.py              # Sync HTTP client for FastAPI
├── session.py                 # Session state management
├── config.py                  # App configuration (env vars)
├── models.py                  # Data models (Album, Person, etc.)
│
├── pages/
│   ├── __init__.py            # Page exports
│   ├── home.py                # Welcome + quick start
│   ├── albums.py              # Album management (CRUD)
│   ├── results.py             # Pipeline runner + results viewer
│   └── people.py              # People browser (Google Photos style)
│
└── components/
    ├── __init__.py            # Component exports
    ├── sidebar.py             # Navigation + API status
    ├── album_selector.py      # Album dropdown + create form
    ├── pipeline_runner.py     # Pipeline config + progress
    ├── gallery.py             # Image grid + cluster gallery
    ├── metrics.py             # Pipeline metrics + charts
    ├── people_browser.py      # People grid + person detail
    └── export_panel.py        # Export options form
```

---

## Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           config.py                              │
├─────────────────────────────────────────────────────────────────┤
│ @dataclass                                                       │
│ class AppConfig:                                                 │
│   api_base_url: str = "http://localhost:8000"                   │
│   poll_interval_sec: float = 2.0                                │
│   max_poll_attempts: int = 300                                  │
│   image_thumbnail_size: int = 200                               │
│   gallery_columns: int = 4                                      │
│   people_columns: int = 5                                       │
├─────────────────────────────────────────────────────────────────┤
│ def get_config() -> AppConfig                                   │
│ def load_from_yaml(path: str) -> AppConfig                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         api_client.py                            │
├─────────────────────────────────────────────────────────────────┤
│ class APIClient:                                                 │
│   _base_url: str                                                │
│   _timeout: int                                                 │
├─────────────────────────────────────────────────────────────────┤
│ # Albums                                                         │
│   list_albums() -> List[Album]                                  │
│   get_album(album_id: str) -> Album                             │
│   create_album(name: str, source_path: str) -> Album            │
│   delete_album(album_id: str) -> bool                           │
│                                                                  │
│ # Pipeline                                                       │
│   list_steps(category: str = None) -> List[Step]                │
│   run_pipeline(album_id, steps, config) -> str  # job_id        │
│   get_pipeline_status(job_id: str) -> PipelineStatus            │
│   poll_until_complete(job_id, callback) -> PipelineResult       │
│                                                                  │
│ # Results                                                        │
│   list_results(album_id: str = None) -> List[ResultSummary]     │
│   get_result(job_id: str) -> PipelineResult                     │
│   get_result_images(job_id, cluster_id, person_id) -> List[Img] │
│   get_result_clusters(job_id: str) -> List[Cluster]             │
│   export_result(job_id, output_path, options) -> ExportResult   │
│                                                                  │
│ # People                                                         │
│   list_people(album_id: str) -> List[Person]                    │
│   get_person(album_id, person_id) -> Person                     │
│   get_person_images(album_id, person_id) -> List[PersonImage]   │
│   rename_person(album_id, person_id, name) -> Person            │
│   merge_people(album_id, person_ids: List[str]) -> Person       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          session.py                              │
├─────────────────────────────────────────────────────────────────┤
│ class SessionState:                                              │
│   """Wrapper around st.session_state"""                         │
├─────────────────────────────────────────────────────────────────┤
│ # State keys                                                     │
│   current_album_id: Optional[str]                               │
│   current_job_id: Optional[str]                                 │
│   selected_steps: List[str]                                     │
│   step_configs: Dict[str, Dict]                                 │
│   cached_result: Optional[PipelineResult]                       │
│   cached_people: Optional[List[Person]]                         │
│                                                                  │
│ # Methods                                                        │
│   @staticmethod initialize()                                    │
│   @staticmethod get(key, default=None)                          │
│   @staticmethod set(key, value)                                 │
│   @staticmethod clear()                                         │
│                                                                  │
│   # Convenience                                                  │
│   @staticmethod get_current_album() -> Optional[Album]          │
│   @staticmethod set_current_album(album: Album)                 │
│   @staticmethod get_current_result() -> Optional[PipelineResult]│
│   @staticmethod load_latest_result(api: APIClient)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          models.py                               │
├─────────────────────────────────────────────────────────────────┤
│ @dataclass                                                       │
│ class Album:                                                     │
│   id: str                                                       │
│   name: str                                                     │
│   source_path: str                                              │
│   image_count: int                                              │
│   created_at: datetime                                          │
│                                                                  │
│ @dataclass                                                       │
│ class Step:                                                      │
│   name: str                                                     │
│   display_name: str                                             │
│   category: str                                                 │
│   description: str                                              │
│   config_schema: dict                                           │
│                                                                  │
│ @dataclass                                                       │
│ class PipelineStatus:                                            │
│   job_id: str                                                   │
│   status: str  # pending, running, completed, failed            │
│   progress: float                                               │
│   current_step: str                                             │
│   message: str                                                  │
│                                                                  │
│ @dataclass                                                       │
│ class PipelineResult:                                            │
│   job_id: str                                                   │
│   album_id: str                                                 │
│   total_images: int                                             │
│   filtered_images: int                                          │
│   num_clusters: int                                             │
│   num_selected: int                                             │
│   selected_images: List[str]                                    │
│   scene_clusters: Dict[int, List[str]]                          │
│   image_metrics: Dict[str, ImageMetrics]                        │
│   step_timings: Dict[str, int]                                  │
│   total_duration_ms: int                                        │
│                                                                  │
│ @dataclass                                                       │
│ class ImageMetrics:                                              │
│   iqa_score: Optional[float]                                    │
│   ava_score: Optional[float]                                    │
│   sharpness: Optional[float]                                    │
│   cluster_id: Optional[int]                                     │
│   is_selected: bool                                             │
│                                                                  │
│ @dataclass                                                       │
│ class Person:                                                    │
│   id: str                                                       │
│   person_index: int                                             │
│   name: Optional[str]                                           │
│   face_count: int                                               │
│   image_count: int                                              │
│   thumbnail_path: Optional[str]                                 │
│                                                                  │
│ @dataclass                                                       │
│ class PersonImage:                                               │
│   image_path: str                                               │
│   face_count: int                                               │
│   is_selected: bool                                             │
│   metrics: ImageMetrics                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                   │
│  - Page config (title, icon, layout)                            │
│  - Initialize session state                                      │
│  - Load persisted state from API                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     pages/1_Home.py                              │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌────────────────────────────────────────┐ │
│ │ Sidebar         │  │ Main Content                           │ │
│ │                 │  │                                        │ │
│ │ sidebar_config  │  │  album_selector                        │ │
│ │ - Quality       │  │  ├─ Dropdown: existing albums          │ │
│ │ - Portrait      │  │  └─ Form: create new album             │ │
│ │ - Selection     │  │                                        │ │
│ │ - Clustering    │  │  pipeline_steps                        │ │
│ │ - Performance   │  │  └─ Checkboxes for each step           │ │
│ │                 │  │                                        │ │
│ │                 │  │  pipeline_runner                       │ │
│ │                 │  │  ├─ Run button                         │ │
│ │                 │  │  ├─ Progress bar                       │ │
│ │                 │  │  └─ Status metrics                     │ │
│ └─────────────────┘  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    pages/2_Results.py                            │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Result Selector (dropdown of completed runs)                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Summary Stats: Total → Filtered → Clusters → Selected       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Tabs                                                         │ │
│ │ ┌──────────┬──────────┬─────────────┬──────────┐            │ │
│ │ │ Gallery  │ Metrics  │ Performance │ Export   │            │ │
│ │ └──────────┴──────────┴─────────────┴──────────┘            │ │
│ │                                                              │ │
│ │ Gallery Tab:                                                 │ │
│ │   - Filter: All / Selected / By Cluster                     │ │
│ │   - Image grid with cards                                   │ │
│ │                                                              │ │
│ │ Metrics Tab:                                                 │ │
│ │   - Sortable dataframe                                      │ │
│ │   - Quality distribution charts                             │ │
│ │                                                              │ │
│ │ Performance Tab:                                             │ │
│ │   - Step timing breakdown                                   │ │
│ │   - Bar chart                                               │ │
│ │                                                              │ │
│ │ Export Tab:                                                  │ │
│ │   - Output path input                                       │ │
│ │   - Options checkboxes                                      │ │
│ │   - Export button                                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    pages/3_People.py                             │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Album Selector                                               │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ People Grid (Google Photos style)                           │ │
│ │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │ │
│ │ │ [face]  │ │ [face]  │ │ [face]  │ │ [face]  │ │ [face]  │ │ │
│ │ │  Mom    │ │  Dad    │ │  Child  │ │ Person4 │ │ Person5 │ │ │
│ │ │ 23 imgs │ │ 18 imgs │ │ 31 imgs │ │ 5 imgs  │ │ 2 imgs  │ │ │
│ │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Selected Person: Mom                                         │ │
│ │ [Rename] [Merge with...]                                    │ │
│ │                                                              │ │
│ │ Images containing Mom:                                       │ │
│ │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │ │
│ │ │     │ │     │ │     │ │     │ │     │ │     │ │     │    │ │
│ │ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘    │ │
│ │                                                              │ │
│ │ Filter: [All] [Selected only] [With others]                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    pages/4_Albums.py                             │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Albums List                                                  │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ Album Name       │ Images │ Runs │ Created    │ Actions │ │ │
│ │ ├─────────────────────────────────────────────────────────┤ │ │
│ │ │ Summer Vacation  │ 245    │ 3    │ 2024-01-15 │ [Del]   │ │ │
│ │ │ Birthday Party   │ 89     │ 1    │ 2024-02-20 │ [Del]   │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Create New Album                                             │ │
│ │ Name: [____________]                                        │ │
│ │ Path: [____________] [Browse]                               │ │
│ │ [Create]                                                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Running a Pipeline

```
┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│  User    │    │ Streamlit │    │ APIClient │    │ FastAPI  │
│          │    │           │    │           │    │          │
└────┬─────┘    └─────┬─────┘    └─────┬─────┘    └────┬─────┘
     │                │                │                │
     │ Click "Run"    │                │                │
     │───────────────>│                │                │
     │                │ run_pipeline() │                │
     │                │───────────────>│                │
     │                │                │ POST /pipeline │
     │                │                │───────────────>│
     │                │                │    job_id      │
     │                │                │<───────────────│
     │                │    job_id      │                │
     │                │<───────────────│                │
     │                │                │                │
     │                │ poll_until_complete()           │
     │                │────┐           │                │
     │  Progress bar  │    │ loop      │                │
     │<───────────────│    │           │                │
     │                │    │ get_status│                │
     │                │    │──────────>│ GET /pipeline/ │
     │                │    │           │───────────────>│
     │                │    │  status   │    status      │
     │                │<───┴───────────│<───────────────│
     │                │                │                │
     │                │ get_result()   │                │
     │                │───────────────>│ GET /result/   │
     │                │                │───────────────>│
     │                │    result      │    result      │
     │                │<───────────────│<───────────────│
     │  Show results  │                │                │
     │<───────────────│                │                │
```

### Loading Persisted Results

```
┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│  User    │    │ Streamlit │    │ APIClient │    │ FastAPI  │
└────┬─────┘    └─────┬─────┘    └─────┬─────┘    └────┬─────┘
     │                │                │                │
     │ Open app       │                │                │
     │───────────────>│                │                │
     │                │ list_albums()  │                │
     │                │───────────────>│ GET /albums    │
     │                │                │───────────────>│
     │                │                │                │
     │                │ list_results() │                │
     │                │───────────────>│ GET /results   │
     │                │                │───────────────>│
     │                │                │                │
     │  Show albums   │ cached in      │                │
     │  + last result │ session_state  │                │
     │<───────────────│                │                │
```

---

## Implementation Plan

### Phase 1: Core Infrastructure
**Files:** `config.py`, `api_client.py`, `session.py`, `models.py`, `main.py`

| Task | Description |
|------|-------------|
| 1.1 | Create `config.py` with AppConfig dataclass |
| 1.2 | Create `models.py` with all data models |
| 1.3 | Create `api_client.py` with sync HTTP methods |
| 1.4 | Add `poll_until_complete()` with configurable interval |
| 1.5 | Create `session.py` for state management |
| 1.6 | Create `main.py` entry point with page config |

### Phase 2: Components (Adapt from existing)
**Files:** `components/*.py`

| Task | Description |
|------|-------------|
| 2.1 | `sidebar_config.py` - Quality, portrait, selection, clustering config |
| 2.2 | `album_selector.py` - Dropdown + create form |
| 2.3 | `pipeline_runner.py` - Run button + polling progress |
| 2.4 | `image_card.py` - Image with status badge + metrics |
| 2.5 | `gallery.py` - Image grid with cluster grouping |
| 2.6 | `metrics_table.py` - Dataframe + charts |
| 2.7 | `export_panel.py` - Export options form |

### Phase 3: Pages
**Files:** `pages/*.py`

| Task | Description |
|------|-------------|
| 3.1 | `1_Home.py` - Album selection + pipeline config + run |
| 3.2 | `2_Results.py` - Tabbed results view |
| 3.3 | `3_People.py` - People grid + image browser |
| 3.4 | `4_Albums.py` - Album management |

### Phase 4: People Features
**Files:** `components/people_grid.py`, `components/person_browser.py`

| Task | Description |
|------|-------------|
| 4.1 | `people_grid.py` - Google Photos style face grid |
| 4.2 | `person_browser.py` - Browse images by person |
| 4.3 | Add rename functionality |
| 4.4 | Add merge functionality |
| 4.5 | Add filter: "with others", "alone", "selected only" |

### Phase 5: Polish & Persistence
| Task | Description |
|------|-------------|
| 5.1 | Load last album/result on app startup |
| 5.2 | Remember selected steps in session |
| 5.3 | Add error handling and user feedback |
| 5.4 | Add loading spinners for API calls |
| 5.5 | Test end-to-end flow |

---

## Configuration

### `configs/streamlit.yaml`

```yaml
# Streamlit Frontend Configuration

api:
  base_url: "http://localhost:8000"
  timeout_sec: 30

polling:
  interval_sec: 2.0
  max_attempts: 300  # 10 minutes max

ui:
  gallery_columns: 4
  people_columns: 5
  thumbnail_size: 200
  max_images_per_page: 50

defaults:
  pipeline_steps:
    - discover_images
    - detect_faces
    - score_iqa
    - score_ava
    - score_face_pose
    - score_face_eyes
    - score_face_smile
    - filter_quality
    - extract_scene_embedding
    - cluster_scenes
    - extract_face_embeddings
    - cluster_by_identity
    - select_best
```

---

## People Browser (Google Photos Style)

### Features

1. **Face Grid View**
   - Circular face thumbnails
   - Name below (editable)
   - Image count badge
   - Click to expand

2. **Person Detail View**
   - All images containing this person
   - Filter options:
     - All images
     - Selected only
     - Solo (person alone)
     - With others (group shots)
   - Sort by: Date, Score, Cluster

3. **Management**
   - Rename: Click name to edit
   - Merge: Select multiple people, click merge
   - Visual indicator for merged people

### UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│  People in "Summer Vacation 2024"                    [Manage]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐    │
│   │ ○───○ │   │ ○───○ │   │ ○───○ │   │ ○───○ │   │ ○───○ │    │
│   │ (   ) │   │ (   ) │   │ (   ) │   │ (   ) │   │ (   ) │    │
│   │  ───  │   │  ───  │   │  ───  │   │  ───  │   │  ───  │    │
│   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘    │
│     Mom         Dad        Emma       Person 4    Person 5      │
│    23 imgs     18 imgs    31 imgs     5 imgs      2 imgs       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ▼ Emma (31 images)                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Filter: [All ▼]  Sort: [Score ▼]  [ ] Selected only     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐   │
│  │ ⭐    │ │ ⭐    │ │       │ │       │ │       │ │       │   │
│  │       │ │       │ │       │ │       │ │       │ │       │   │
│  │       │ │       │ │       │ │       │ │       │ │       │   │
│  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘   │
│  Beach 1    Beach 2   Museum    Park      Pool      Garden      │
│  +Mom,Dad   +Dad      Solo      +Mom      Solo      +Everyone   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

```
# requirements-streamlit.txt
streamlit>=1.30.0
requests>=2.31.0
pandas>=2.0.0
Pillow>=10.0.0
pydantic>=2.0.0
PyYAML>=6.0.0
```

---

## Running

```bash
# Start FastAPI backend first
uvicorn sim_bench.api.main:app --reload --port 8000

# Then start Streamlit
streamlit run app/streamlit/main.py --server.port 8501
```

---

## Migration Notes

### From existing `app/album/`

| Old | New | Notes |
|-----|-----|-------|
| `AlbumService` direct call | `api_client.run_pipeline()` | No direct service access |
| `WorkflowResult` | `PipelineResult` | From API response |
| Progress callback | Polling loop | Check status every N sec |
| `session.py` stores service | `session.py` stores IDs | Stateless frontend |

### Reusable UI patterns

- Config panel expanders
- Progress bar with metrics
- Image card with badges
- Tabbed results view
- Metrics dataframe with download

---

## Status

- [x] Phase 1: Core Infrastructure ✅ Complete
  - [x] `config.py` - AppConfig with env vars
  - [x] `models.py` - All data models
  - [x] `api_client.py` - Sync HTTP client with all endpoints
  - [x] `session.py` - Session state management
  - [x] `main.py` - Entry point with page routing

- [x] Phase 2: Components ✅ Complete
  - [x] `sidebar.py` - Navigation, API status, settings
  - [x] `album_selector.py` - Dropdown, create form, album list
  - [x] `pipeline_runner.py` - Config, run button, progress display
  - [x] `gallery.py` - Image grid, cluster gallery, comparison
  - [x] `metrics.py` - Pipeline metrics, step timings, summaries
  - [x] `people_browser.py` - People grid, person detail, rename
  - [x] `export_panel.py` - Export options form

- [x] Phase 3: Pages ✅ Complete
  - [x] `pages/home.py` - Welcome, quick start, album selection
  - [x] `pages/albums.py` - Album management with CRUD
  - [x] `pages/results.py` - Pipeline runner + results viewer tabs
  - [x] `pages/people.py` - Google Photos style people browser

- [x] Phase 4: People Features ✅ Complete
  - [x] People grid view with search/sort
  - [x] Person detail view with images
  - [x] Rename functionality
  - [x] Merge people functionality (multi-select mode)
  - [x] Split person functionality (face indices dialog)
  - [x] Filter: "Solo (alone)", "With others", "Selected only"
  - [x] Sort: Score, Face count, Filename
  - [x] Manage mode toggle for bulk operations

- [x] Phase 5: Polish & Persistence ✅ Complete
  - [x] Load last album on startup (auto-select first album)
  - [x] Absolute imports throughout (no relative imports)
  - [x] Simplified error handling (removed excessive try/except)
  - [x] Clean page routing in main.py
  - [ ] End-to-end testing (manual)
