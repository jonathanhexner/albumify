# Pipeline Architecture Plan

## Overview

A composable, AI-agent-friendly pipeline architecture for album organization with:
- **FastAPI backend** - Clean router/service separation
- **NiceGUI frontend** - Vue-based Python UI
- **Pluggable pipeline steps** - Mix and match capabilities
- **Two processing flows** - Scene-dominant and Face-dominant paths

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NiceGUI Frontend                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Upload    │  │   Config    │  │  Progress   │  │   Gallery   │        │
│  │    Page     │  │    Panel    │  │   Display   │  │   Results   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ HTTP/WebSocket
┌────────────────────────────────▼────────────────────────────────────────────┐
│                              FastAPI Backend                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           ROUTERS (API Layer)                         │   │
│  │  /api/v1/albums     - Album CRUD, upload                             │   │
│  │  /api/v1/pipeline   - Run pipeline, get status                       │   │
│  │  /api/v1/steps      - List available steps, get schemas              │   │
│  │  /api/v1/results    - Get results, export                            │   │
│  │  /ws/progress       - WebSocket for real-time updates                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────▼───────────────────────────────────┐   │
│  │                         SERVICES (Business Logic)                     │   │
│  │  AlbumService        - Album management                              │   │
│  │  PipelineService     - Pipeline orchestration                        │   │
│  │  StepRegistry        - Step discovery and metadata                   │   │
│  │  ResultService       - Result storage and retrieval                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────▼───────────────────────────────────┐   │
│  │                         PIPELINE ENGINE                               │   │
│  │  PipelineBuilder     - Compose pipelines from step names             │   │
│  │  PipelineExecutor    - Run pipeline with context                     │   │
│  │  DependencyResolver  - Auto-add required steps                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────▼───────────────────────────────────┐   │
│  │                         PIPELINE STEPS                                │   │
│  │  Analysis   │  Filtering   │  Clustering   │  Selection   │  Export  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────▼───────────────────────────────────┐   │
│  │                         MODEL HUB (ML Models)                         │   │
│  │  IQA │ AVA │ MediaPipe │ SixDRepNet │ DINOv2 │ ArcFace │ HDBSCAN    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Step Interface

### Core Protocol

```python
from typing import Protocol, Any, Set
from pydantic import BaseModel

class StepMetadata(BaseModel):
    """Metadata for AI agent discovery."""
    name: str                      # Unique identifier: "score_pose"
    display_name: str              # Human readable: "Score Head Pose"
    description: str               # What this step does
    category: str                  # "analysis", "filtering", "clustering", "selection"

    requires: Set[str]             # Context keys this step reads
    produces: Set[str]             # Context keys this step writes
    depends_on: list[str]          # Steps that must run before this one

    config_schema: dict            # JSON Schema for configuration options


class PipelineStep(Protocol):
    """Standard interface for all pipeline steps."""

    @property
    def metadata(self) -> StepMetadata:
        """Return step metadata for discovery."""
        ...

    def process(self, context: "PipelineContext", config: dict) -> None:
        """
        Execute this step.

        Reads from context what it needs (metadata.requires).
        Writes to context what it produces (metadata.produces).
        Raises StepError on failure.
        """
        ...

    def validate(self, context: "PipelineContext") -> list[str]:
        """
        Validate that required inputs exist in context.
        Returns list of error messages (empty if valid).
        """
        ...
```

### Pipeline Context

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np

@dataclass
class PipelineContext:
    """Shared state passed through all pipeline steps."""

    # Input
    source_directory: Path
    image_paths: list[Path] = field(default_factory=list)

    # Analysis results (keyed by image path string)
    iqa_scores: dict[str, float] = field(default_factory=dict)
    ava_scores: dict[str, float] = field(default_factory=dict)
    sharpness_scores: dict[str, float] = field(default_factory=dict)

    # Face-specific (keyed by image path string)
    faces: dict[str, list["CroppedFace"]] = field(default_factory=dict)
    face_pose_scores: dict[str, list[float]] = field(default_factory=dict)
    face_eyes_scores: dict[str, list[float]] = field(default_factory=dict)
    face_smile_scores: dict[str, list[float]] = field(default_factory=dict)

    # Embeddings
    scene_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    face_embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict)

    # Filtering
    quality_passed: set[str] = field(default_factory=set)      # Images passing quality filter
    portrait_passed: set[str] = field(default_factory=set)     # Images passing portrait filter
    active_images: set[str] = field(default_factory=set)       # Currently active (not filtered out)

    # Clustering results
    scene_clusters: dict[int, list[str]] = field(default_factory=dict)  # cluster_id -> [image_paths]
    face_clusters: dict[int, dict[int, list[str]]] = field(default_factory=dict)
    # scene_cluster_id -> {face_cluster_id -> [image_paths]}

    # Selection
    selected_images: list[str] = field(default_factory=list)

    # Metadata
    is_face_dominant: dict[str, bool] = field(default_factory=dict)  # Per-image flag

    # People feature (Google Photos-style)
    all_faces: list["FaceInstance"] = field(default_factory=list)  # All faces from all images
    all_face_embeddings: np.ndarray = None  # (N_faces, 512) ArcFace embeddings
    people_clusters: dict[int, list["FaceInstance"]] = field(default_factory=dict)  # person_id -> faces
    people_thumbnails: dict[int, "FaceInstance"] = field(default_factory=dict)  # person_id -> best face

    # Progress callback
    on_progress: Callable[[str, float], None] | None = None

    # Configuration (set by pipeline builder)
    step_configs: dict[str, dict] = field(default_factory=dict)
```

---

## Step Catalog

Steps are **thin wrappers** around existing code. The `requires` field indicates what context keys the step reads (must be populated by a previous step). The `produces` field indicates what context keys the step writes.

### Analysis Steps

| Step Name | Requires | Produces | Wraps |
|-----------|----------|----------|-------|
| `discover_images` | - | `image_paths` | `album/stages.discover_images()` |
| `score_iqa` | `image_paths` | `iqa_scores`, `sharpness_scores` | IQA model from ModelHub |
| `score_ava` | `image_paths` | `ava_scores` | AVA model from ModelHub |
| `detect_faces` | `image_paths` | `faces`, `is_face_dominant` | `face_pipeline/crop_service.py` |
| `score_face_pose` | `faces` | `face_pose_scores` | `face_pipeline/pose_estimator.py` |
| `score_face_eyes` | `faces` | `face_eyes_scores` | `portrait_analysis/eye_state.py` |
| `score_face_smile` | `faces` | `face_smile_scores` | `portrait_analysis/smile_detection.py` |

### Filtering Steps

| Step Name | Requires | Produces | Wraps |
|-----------|----------|----------|-------|
| `filter_quality` | `iqa_scores`, `ava_scores` | `quality_passed`, `active_images` | `album/stages.filter_by_quality()` |
| `filter_portrait` | `face_eyes_scores`, `is_face_dominant` | `portrait_passed`, `active_images` | `album/stages.filter_by_portrait()` |
| `filter_pose` | `face_pose_scores`, `is_face_dominant` | `active_images` | New (simple threshold check) |

### Embedding Steps

| Step Name | Requires | Produces | Wraps |
|-----------|----------|----------|-------|
| `extract_scene_embedding` | `active_images` | `scene_embeddings` | `feature_extraction/dinov2.py` |
| `extract_face_embedding` | `faces` | `face_embeddings` | `album/services/face_embedding_service.py` |

### Clustering Steps

| Step Name | Requires | Produces | Wraps |
|-----------|----------|----------|-------|
| `cluster_scenes` | `scene_embeddings` | `scene_clusters` | `clustering/hdbscan.py` |
| `cluster_faces` | `face_embeddings`, `scene_clusters` | `face_clusters` | `album/services/face_embedding_service.cluster_faces()` |

### Selection Steps

| Step Name | Requires | Produces | Wraps |
|-----------|----------|----------|-------|
| `select_best_per_cluster` | `scene_clusters`, `*_scores` | `selected_images` | `album/services/selection_service.py` |
| `select_best_face_per_identity` | `face_clusters`, `face_*_scores` | `selected_images` | New (extends selection_service) |

### People Feature Steps (Google Photos-style)

| Step Name | Requires | Produces | Wraps |
|-----------|----------|----------|-------|
| `extract_all_faces` | `image_paths` | `all_faces` | `face_pipeline/crop_service.py` (batch) |
| `embed_all_faces` | `all_faces` | `all_face_embeddings` | `face_embedding_service.py` |
| `cluster_people` | `all_face_embeddings` | `people_clusters` | ArcFace + Agglomerative clustering |
| `select_best_per_person` | `people_clusters`, `face_*_scores` | `people_thumbnails` | New |

**Key difference from scene-scoped face clustering:**
- `cluster_people` operates on ALL faces across ALL images (global)
- `face_clusters` operates within each scene cluster (local)

This enables the "People" album view where each person has a representative thumbnail.

---

## Data Structures for People Feature

```python
@dataclass
class FaceInstance:
    """A single face occurrence in an image."""
    face_id: str                    # Unique ID: "{image_path}_{face_index}"
    image_path: str                 # Source image
    face_index: int                 # Which face in image (0, 1, 2...)
    cropped_face: CroppedFace       # The cropped face data

    # Quality scores (populated by scoring steps)
    pose_score: float = 0.0         # 0-1, higher = more frontal
    eyes_open_score: float = 0.0    # 0-1, higher = more open
    smile_score: float = 0.0        # 0-1, higher = more smile
    sharpness_score: float = 0.0    # 0-1, technical quality

    # Computed
    overall_score: float = 0.0      # Weighted combination

    # Assigned after clustering
    person_id: int = -1             # Which person cluster (-1 = unassigned)


@dataclass
class Person:
    """A person identified across multiple images."""
    person_id: int
    face_instances: list[FaceInstance]  # All faces of this person
    thumbnail: FaceInstance              # Best face (for People view)
    image_count: int                     # How many images contain this person

    @property
    def name(self) -> str:
        return f"Person {self.person_id}"  # Can be renamed by user later
```

---

## Dependency Resolution

```python
# Step dependencies (auto-added when building pipeline)
STEP_DEPENDENCIES = {
    # Analysis
    "score_iqa": ["discover_images"],
    "score_ava": ["discover_images"],
    "detect_faces": ["discover_images"],
    "score_face_pose": ["detect_faces"],
    "score_face_eyes": ["detect_faces"],
    "score_face_smile": ["detect_faces"],

    # Filtering
    "filter_quality": ["score_iqa", "score_ava"],
    "filter_portrait": ["score_face_eyes", "detect_faces"],
    "filter_pose": ["score_face_pose"],

    # Embedding
    "extract_scene_embedding": ["filter_quality"],
    "extract_face_embedding": ["detect_faces"],

    # Scene clustering
    "cluster_scenes": ["extract_scene_embedding"],
    "cluster_faces": ["extract_face_embedding", "cluster_scenes"],
    "select_best_per_cluster": ["cluster_scenes"],
    "select_best_face_per_identity": ["cluster_faces"],

    # People feature (Google Photos-style)
    "extract_all_faces": ["discover_images"],
    "embed_all_faces": ["extract_all_faces"],
    "cluster_people": ["embed_all_faces"],
    "select_best_per_person": ["cluster_people", "score_face_pose", "score_face_eyes", "score_face_smile"],
}


class PipelineBuilder:
    """Build pipeline with automatic dependency resolution."""

    def __init__(self, registry: StepRegistry):
        self.registry = registry

    def build(self, step_names: list[str]) -> list[PipelineStep]:
        """
        Build ordered pipeline from step names.
        Automatically inserts missing dependencies.
        """
        # Topological sort with dependency injection
        resolved = []
        seen = set()

        def add_with_deps(name: str):
            if name in seen:
                return
            for dep in STEP_DEPENDENCIES.get(name, []):
                add_with_deps(dep)
            seen.add(name)
            resolved.append(self.registry.get(name))

        for name in step_names:
            add_with_deps(name)

        return resolved
```

---

## Two Processing Flows

### Flow Decision Logic

```python
def determine_flow(context: PipelineContext, image_path: str) -> str:
    """
    Determine which flow an image should follow.

    Returns: "scene" or "face"
    """
    if image_path not in context.is_face_dominant:
        return "scene"  # No face detection yet

    return "face" if context.is_face_dominant[image_path] else "scene"
```

### Scene-Dominant Flow

```
discover_images
      │
      ▼
┌─────────────────┐
│  score_iqa      │──┐
│  score_ava      │  │ parallel
│  detect_faces   │──┘ (marks is_face_dominant)
└─────────────────┘
      │
      ▼
filter_quality (IQA > 0.3, AVA > 4.0)
      │
      ▼
extract_scene_embedding (DINOv2)
      │
      ▼
cluster_scenes (HDBSCAN)
      │
      ▼
select_best_per_cluster (composite score)
      │
      ▼
Selected Images
```

### Face-Dominant Flow (within scene clusters)

```
[After scene clustering]
      │
      ▼
For each scene_cluster:
  │
  ▼
filter images where is_face_dominant=True
  │
  ▼
┌─────────────────────┐
│  score_face_pose    │──┐
│  score_face_eyes    │  │ parallel
│  score_face_smile   │──┘
└─────────────────────┘
  │
  ▼
filter_pose (penalize non-frontal)
  │
  ▼
filter_portrait (require eyes open)
  │
  ▼
extract_face_embedding (ArcFace)
  │
  ▼
cluster_faces (within this scene cluster)
  │
  ▼
select_best_face_per_identity
  │
  ▼
Selected: Best photo of each person in this scene cluster
```

### Combined Pipeline (Hierarchical)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ALBUM PROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

discover_images
      │
      ├──────────────────┬──────────────────┐
      ▼                  ▼                  ▼
  score_iqa          score_ava        detect_faces
      │                  │                  │
      └──────────────────┴──────────────────┘
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
          ┌──────────────┴──────────────┐
          ▼                             ▼
   Scene Cluster 0               Scene Cluster 1
          │                             │
    ┌─────┴─────┐                 ┌─────┴─────┐
    ▼           ▼                 ▼           ▼
  Face-      Non-face           Face-      Non-face
  dominant   images             dominant   images
    │           │                 │           │
    ▼           │                 ▼           │
 [Face Flow]    │              [Face Flow]    │
    │           │                 │           │
    ▼           │                 ▼           │
 Face sub-      │              Face sub-      │
 clusters       │              clusters       │
    │           │                 │           │
    ▼           ▼                 ▼           ▼
 Select best  Select best     Select best  Select best
 per identity from scene      per identity from scene
    │           │                 │           │
    └─────┬─────┘                 └─────┬─────┘
          │                             │
          ▼                             ▼
   Selected from              Selected from
   Scene Cluster 0            Scene Cluster 1
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  Final Selection
```

---

## Code Conventions

Follow project conventions throughout:

1. **Absolute imports only** - No relative imports (`from . import`)
2. **Empty `__init__.py` files** - No re-exports or initialization code
3. **Avoid try/except** - Let errors propagate, handle at API boundary only
4. **Avoid excessive if/else** - Use early returns, polymorphism, dict lookups
5. **Type hints everywhere** - All function signatures fully typed
6. **Dataclasses for data** - Use `@dataclass` or Pydantic `BaseModel` for structured data

```python
# Good
from sim_bench.pipeline.context import PipelineContext
from sim_bench.face_pipeline.crop_service import FaceCropService

# Bad
from .context import PipelineContext
from ..face_pipeline import crop_service
```

---

## API Endpoints (Detailed)

### Albums API (`/api/v1/albums`)

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| POST | `/albums` | `{name, source_path}` | `Album` | Register album (validates path exists) |
| GET | `/albums` | - | `List[Album]` | List all registered albums |
| GET | `/albums/{id}` | - | `Album` | Get album details |
| DELETE | `/albums/{id}` | - | `{ok: true}` | Remove album record (images untouched) |

**Note**: Albums are just metadata. Images stay on disk at `source_path`.

### Steps API (`/api/v1/steps`)

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| GET | `/steps` | `?category=analysis` | `List[StepInfo]` | List available steps |
| GET | `/steps/{name}` | - | `StepInfo` | Get step details + config schema |

Used by UI to build config forms and by AI agents to discover capabilities.

### Pipeline API (`/api/v1/pipeline`)

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| POST | `/pipeline/run` | `{album_id, steps?, config?}` | `{job_id}` | Start pipeline (background task) |
| GET | `/pipeline/{job_id}` | - | `PipelineStatus` | Get status, progress, current step |
| DELETE | `/pipeline/{job_id}` | - | `{ok: true}` | Cancel running pipeline |

### Results API (`/api/v1/results`)

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| GET | `/results/{job_id}` | - | `PipelineResult` | Get full results (clusters, selected, metrics) |
| GET | `/results/{job_id}/images` | `?cluster_id=0` | `List[ImageInfo]` | Get images with thumbnails |
| POST | `/results/{job_id}/export` | `{output_path, format}` | `{ok: true}` | Export selected to folder |

### People API (`/api/v1/people`)

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| GET | `/people/{album_id}` | - | `List[Person]` | List all people in album |
| GET | `/people/{album_id}/{person_id}` | - | `Person` | Get person details + all their faces |
| GET | `/people/{album_id}/{person_id}/images` | - | `List[ImageInfo]` | Get all images containing this person |
| PATCH | `/people/{album_id}/{person_id}` | `{name}` | `Person` | Rename person (e.g., "Mom") |
| POST | `/people/{album_id}/merge` | `{person_ids: [1,2]}` | `Person` | Merge two people (same identity) |
| POST | `/people/{album_id}/split` | `{person_id, face_ids}` | `List[Person]` | Split person (wrong grouping) |

### WebSocket (`/ws`)

| Endpoint | Direction | Message Types |
|----------|-----------|---------------|
| `/ws/progress/{job_id}` | Server→Client | `progress`, `step_complete`, `complete`, `error` |

```json
// Progress update
{"type": "progress", "step": "score_iqa", "progress": 0.45, "message": "45/100 images"}

// Step complete
{"type": "step_complete", "step": "score_iqa", "duration_ms": 1234}

// Pipeline complete
{"type": "complete", "result_id": "abc123"}

// Error
{"type": "error", "step": "score_ava", "message": "Model failed to load"}
```

---

## FastAPI Structure

### Directory Structure

```
sim_bench/
└── api/
    ├── __init__.py
    ├── main.py                 # FastAPI app, middleware, startup
    ├── dependencies.py         # Dependency injection (get_db, get_services)
    │
    ├── routers/
    │   ├── __init__.py
    │   ├── albums.py           # /api/v1/albums
    │   ├── pipeline.py         # /api/v1/pipeline
    │   ├── steps.py            # /api/v1/steps
    │   ├── results.py          # /api/v1/results
    │   └── websocket.py        # /ws/progress
    │
    ├── schemas/                # Pydantic models for API
    │   ├── __init__.py
    │   ├── album.py            # AlbumCreate, AlbumResponse
    │   ├── pipeline.py         # PipelineRequest, PipelineStatus
    │   ├── step.py             # StepInfo, StepConfig
    │   └── result.py           # ResultResponse, ExportRequest
    │
    └── services/               # Business logic (thin layer calling core)
        ├── __init__.py
        ├── album_service.py
        ├── pipeline_service.py
        ├── step_registry.py
        └── result_service.py
```

### Router Examples

```python
# routers/pipeline.py
from fastapi import APIRouter, Depends, BackgroundTasks
from ..schemas.pipeline import PipelineRequest, PipelineStatus, PipelineResult
from ..services.pipeline_service import PipelineService
from ..dependencies import get_pipeline_service

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineStatus)
async def run_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    service: PipelineService = Depends(get_pipeline_service)
):
    """
    Start a pipeline run.

    Request body:
    - album_id: ID of uploaded album
    - steps: List of step names (optional, uses default if not provided)
    - config: Step configurations (optional)

    Returns job_id for tracking progress via WebSocket.
    """
    job_id = await service.start_pipeline(
        album_id=request.album_id,
        steps=request.steps,
        config=request.config,
        background_tasks=background_tasks
    )
    return PipelineStatus(job_id=job_id, status="started")


@router.get("/status/{job_id}", response_model=PipelineStatus)
async def get_status(
    job_id: str,
    service: PipelineService = Depends(get_pipeline_service)
):
    """Get current status of a pipeline run."""
    return await service.get_status(job_id)


@router.get("/result/{job_id}", response_model=PipelineResult)
async def get_result(
    job_id: str,
    service: PipelineService = Depends(get_pipeline_service)
):
    """Get results of a completed pipeline run."""
    return await service.get_result(job_id)
```

```python
# routers/steps.py
from fastapi import APIRouter, Depends
from ..schemas.step import StepInfo, StepListResponse
from ..services.step_registry import StepRegistry
from ..dependencies import get_step_registry

router = APIRouter(prefix="/api/v1/steps", tags=["steps"])


@router.get("/", response_model=StepListResponse)
async def list_steps(
    category: str | None = None,
    registry: StepRegistry = Depends(get_step_registry)
):
    """
    List all available pipeline steps.

    Used by AI agents to discover capabilities.
    Optionally filter by category: analysis, filtering, clustering, selection
    """
    steps = registry.list_steps(category=category)
    return StepListResponse(steps=steps)


@router.get("/{step_name}", response_model=StepInfo)
async def get_step(
    step_name: str,
    registry: StepRegistry = Depends(get_step_registry)
):
    """Get detailed info about a specific step including config schema."""
    return registry.get_step_info(step_name)


@router.get("/{step_name}/schema")
async def get_step_config_schema(
    step_name: str,
    registry: StepRegistry = Depends(get_step_registry)
):
    """Get JSON Schema for step configuration (for dynamic form generation)."""
    return registry.get_config_schema(step_name)
```

### WebSocket for Progress

```python
# routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.pipeline_service import PipelineService

router = APIRouter()


@router.websocket("/ws/progress/{job_id}")
async def pipeline_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time pipeline progress.

    Sends JSON messages:
    - {"type": "progress", "step": "score_iqa", "progress": 0.45, "message": "Processing 45/100 images"}
    - {"type": "step_complete", "step": "score_iqa", "duration_ms": 1234}
    - {"type": "complete", "result_id": "abc123"}
    - {"type": "error", "message": "Failed to process image X"}
    """
    await websocket.accept()

    try:
        async for update in PipelineService.subscribe(job_id):
            await websocket.send_json(update)
    except WebSocketDisconnect:
        pass
```

---

## API Schemas

```python
# schemas/pipeline.py
from pydantic import BaseModel
from typing import Optional

class PipelineRequest(BaseModel):
    album_id: str
    steps: Optional[list[str]] = None  # None = use default pipeline
    config: Optional[dict[str, dict]] = None  # step_name -> config

class PipelineStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    current_step: Optional[str] = None
    progress: float = 0.0  # 0-1
    message: Optional[str] = None

class PipelineResult(BaseModel):
    job_id: str
    album_id: str
    total_images: int
    filtered_images: int
    scene_clusters: dict[int, list[str]]
    face_clusters: dict[int, dict[int, list[str]]]
    selected_images: list[str]
    metrics: dict[str, dict]  # image_path -> metrics dict
    telemetry: dict  # timing info


# schemas/step.py
class StepInfo(BaseModel):
    name: str
    display_name: str
    description: str
    category: str
    requires: list[str]
    produces: list[str]
    depends_on: list[str]
    config_schema: dict  # JSON Schema

class StepListResponse(BaseModel):
    steps: list[StepInfo]
```

---

## NiceGUI Frontend Structure

```
app/
└── nicegui_app/
    ├── __init__.py
    ├── main.py                 # NiceGUI app entry point
    ├── api_client.py           # HTTP/WebSocket client for FastAPI
    │
    ├── pages/
    │   ├── __init__.py
    │   ├── home.py             # Landing page, album upload
    │   ├── configure.py        # Pipeline configuration
    │   ├── progress.py         # Processing progress view
    │   ├── results.py          # Gallery and selection results
    │   └── export.py           # Export options
    │
    ├── components/
    │   ├── __init__.py
    │   ├── upload.py           # Drag-drop upload component
    │   ├── step_config.py      # Dynamic step configuration form
    │   ├── pipeline_builder.py # Visual pipeline builder
    │   ├── progress_bar.py     # Real-time progress display
    │   ├── image_gallery.py    # Image grid with selection
    │   ├── cluster_view.py     # Cluster visualization
    │   └── face_grid.py        # Face clusters display
    │
    ├── state/
    │   ├── __init__.py
    │   └── app_state.py        # Reactive state management
    │
    └── styles/
        └── main.css            # Custom styling
```

### NiceGUI Example Components

```python
# pages/configure.py
from nicegui import ui
from ..api_client import APIClient
from ..components.pipeline_builder import PipelineBuilder
from ..components.step_config import StepConfigPanel

async def configure_page(album_id: str):
    """Pipeline configuration page."""

    client = APIClient()
    available_steps = await client.get_steps()

    with ui.card().classes('w-full'):
        ui.label('Configure Pipeline').classes('text-h5')

        # Pipeline builder - drag and drop steps
        pipeline_builder = PipelineBuilder(
            available_steps=available_steps,
            on_change=lambda steps: update_preview(steps)
        )

        # Step configuration panels
        with ui.expansion('Step Settings'):
            for step in available_steps:
                StepConfigPanel(step=step)

        # Run button
        ui.button('Run Pipeline', on_click=lambda: run_pipeline())


# components/pipeline_builder.py
from nicegui import ui

class PipelineBuilder:
    """Visual pipeline step selector."""

    def __init__(self, available_steps: list, on_change):
        self.selected_steps = []
        self.on_change = on_change

        with ui.row().classes('w-full gap-4'):
            # Available steps
            with ui.card().classes('w-1/2'):
                ui.label('Available Steps')
                for step in available_steps:
                    with ui.row().classes('items-center'):
                        ui.checkbox(
                            step['display_name'],
                            on_change=lambda e, s=step: self.toggle_step(s, e.value)
                        )
                        ui.icon('info').tooltip(step['description'])

            # Selected pipeline
            with ui.card().classes('w-1/2'):
                ui.label('Pipeline')
                self.pipeline_display = ui.column()

    def toggle_step(self, step, selected):
        if selected:
            self.selected_steps.append(step['name'])
        else:
            self.selected_steps.remove(step['name'])
        self.on_change(self.selected_steps)
```

---

## What "Wrapping" Means

Existing code stays intact. Steps are thin adapters that:
1. Read inputs from `PipelineContext`
2. Call existing service/function
3. Write outputs to `PipelineContext`

```python
# Existing code (UNCHANGED)
# sim_bench/face_pipeline/crop_service.py
class FaceCropService:
    def crop_faces(self, image_path: Path) -> list[CroppedFace]:
        ...

# New step (THIN WRAPPER)
# sim_bench/pipeline/steps/detect_faces.py
from sim_bench.pipeline.base import PipelineStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.face_pipeline.crop_service import FaceCropService

class DetectFacesStep(PipelineStep):

    def __init__(self):
        self._service = None

    @property
    def metadata(self) -> StepMetadata:
        return StepMetadata(
            name="detect_faces",
            requires={"image_paths"},
            produces={"faces", "is_face_dominant"},
            ...
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if self._service is None:
            self._service = FaceCropService(config)

        for image_path in context.image_paths:
            faces = self._service.crop_faces(image_path)
            context.faces[str(image_path)] = faces
            context.is_face_dominant[str(image_path)] = self._is_dominant(faces, config)

    def _is_dominant(self, faces: list, config: dict) -> bool:
        if not faces:
            return False
        min_ratio = config.get("min_face_ratio", 0.02)
        return any(f.face_ratio >= min_ratio for f in faces)
```

**Key points:**
- `FaceCropService` unchanged
- Step handles context read/write
- Step handles config injection
- Lazy initialization of service

---

## Step Implementation Example

```python
# sim_bench/pipeline/steps/score_face_pose.py
from dataclasses import dataclass
from ..base import PipelineStep, StepMetadata
from ..context import PipelineContext
from ...face_pipeline.pose_estimator import SixDRepNetEstimator

@dataclass
class ScoreFacePoseConfig:
    yaw_threshold: float = 20.0
    pitch_threshold: float = 15.0
    roll_threshold: float = 10.0
    penalty_steepness: float = 0.1  # For sigmoid calculation


class ScoreFacePoseStep(PipelineStep):
    """Score head pose using SixDRepNet."""

    def __init__(self):
        self._estimator = None

    @property
    def metadata(self) -> StepMetadata:
        return StepMetadata(
            name="score_face_pose",
            display_name="Score Head Pose",
            description="Estimate head pose (yaw, pitch, roll) using SixDRepNet. "
                       "Penalizes faces that deviate from frontal orientation.",
            category="analysis",
            requires={"faces"},
            produces={"face_pose_scores"},
            depends_on=["detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "yaw_threshold": {
                        "type": "number",
                        "default": 20.0,
                        "description": "Max yaw angle (degrees) before penalty"
                    },
                    "pitch_threshold": {
                        "type": "number",
                        "default": 15.0,
                        "description": "Max pitch angle (degrees) before penalty"
                    },
                    "roll_threshold": {
                        "type": "number",
                        "default": 10.0,
                        "description": "Max roll angle (degrees) before penalty"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        cfg = ScoreFacePoseConfig(**config)

        if self._estimator is None:
            self._estimator = SixDRepNetEstimator()

        total_faces = sum(len(faces) for faces in context.faces.values())
        processed = 0

        for image_path, faces in context.faces.items():
            scores = []
            for face in faces:
                # Get pose angles
                yaw, pitch, roll = self._estimator.estimate(face.image)

                # Calculate penalty (0 = perfect frontal, 1 = very off-angle)
                penalty = self._calculate_penalty(yaw, pitch, roll, cfg)
                score = 1.0 - penalty  # Convert to score (higher = better)
                scores.append(score)

                processed += 1
                if context.on_progress:
                    context.on_progress("score_face_pose", processed / total_faces)

            context.face_pose_scores[image_path] = scores

    def _calculate_penalty(self, yaw, pitch, roll, cfg) -> float:
        """Soft sigmoid penalty for pose deviation."""
        import math

        def sigmoid_penalty(angle, threshold, k=0.1):
            deviation = abs(angle) - threshold
            if deviation <= 0:
                return 0.0
            return 1 / (1 + math.exp(-k * deviation))

        yaw_penalty = sigmoid_penalty(yaw, cfg.yaw_threshold, cfg.penalty_steepness)
        pitch_penalty = sigmoid_penalty(pitch, cfg.pitch_threshold, cfg.penalty_steepness)
        roll_penalty = sigmoid_penalty(roll, cfg.roll_threshold, cfg.penalty_steepness)

        # Combine penalties (max or average?)
        return max(yaw_penalty, pitch_penalty, roll_penalty)

    def validate(self, context: PipelineContext) -> list[str]:
        errors = []
        if not context.faces:
            errors.append("No faces detected. Run 'detect_faces' step first.")
        return errors
```

---

## Default Pipelines

```python
# Predefined pipeline configurations

PIPELINES = {
    "default": [
        "discover_images",
        "score_iqa",
        "score_ava",
        "detect_faces",
        "filter_quality",
        "extract_scene_embedding",
        "cluster_scenes",
        "select_best_per_cluster"
    ],

    "face_aware": [
        "discover_images",
        "score_iqa",
        "score_ava",
        "detect_faces",
        "score_face_pose",
        "score_face_eyes",
        "score_face_smile",
        "filter_quality",
        "filter_portrait",
        "filter_pose",
        "extract_scene_embedding",
        "cluster_scenes",
        "extract_face_embedding",
        "cluster_faces",
        "select_best_face_per_identity"
    ],

    "people": [
        # Google Photos-style "People" album
        "discover_images",
        "extract_all_faces",
        "score_face_pose",
        "score_face_eyes",
        "score_face_smile",
        "embed_all_faces",
        "cluster_people",
        "select_best_per_person"
    ],

    "full": [
        # Everything: scene clustering + people identification
        "discover_images",
        "score_iqa",
        "score_ava",
        "extract_all_faces",
        "score_face_pose",
        "score_face_eyes",
        "score_face_smile",
        "filter_quality",
        "filter_portrait",
        "extract_scene_embedding",
        "cluster_scenes",
        "embed_all_faces",
        "cluster_people",
        "select_best_per_cluster",
        "select_best_per_person"
    ],

    "quick": [
        "discover_images",
        "score_iqa",
        "filter_quality",
        "extract_scene_embedding",
        "cluster_scenes",
        "select_best_per_cluster"
    ]
}
```

---

## People Pipeline Flow (Google Photos-style)

```
discover_images
      │
      ▼
extract_all_faces (FaceCropService on all images)
      │
      ├─────────────────────────────────┐
      ▼                                 ▼
score_face_pose                   score_face_eyes
(SixDRepNet)                      (MediaPipe EAR)
      │                                 │
      └─────────────┬───────────────────┘
                    │
                    ▼
            score_face_smile
            (MediaPipe landmarks)
                    │
                    ▼
            embed_all_faces
            (ArcFace → 512-dim per face)
                    │
                    ▼
            cluster_people
            (Agglomerative clustering, cosine distance)
            Groups faces by identity across ALL images
                    │
                    ▼
         ┌──────────┴──────────┐
         ▼                     ▼
    Person 0              Person 1         ...
    (15 faces)            (8 faces)
         │                     │
         ▼                     ▼
select_best_per_person    select_best_per_person
  (highest overall_score)   (highest overall_score)
         │                     │
         ▼                     ▼
    Thumbnail 0           Thumbnail 1

Output: people_thumbnails = {
    0: FaceInstance(best face of person 0),
    1: FaceInstance(best face of person 1),
    ...
}
```

### Selection Criteria for Best Face Per Person

```python
def select_best_face(faces: list[FaceInstance], weights: dict) -> FaceInstance:
    """
    Select best face from a person cluster.

    Default weights:
    - pose_score: 0.35 (frontal face most important for thumbnail)
    - eyes_open_score: 0.30 (must have eyes open)
    - smile_score: 0.15 (nice to have)
    - sharpness_score: 0.20 (technical quality)
    """
    for face in faces:
        face.overall_score = (
            face.pose_score * weights.get("pose", 0.35) +
            face.eyes_open_score * weights.get("eyes", 0.30) +
            face.smile_score * weights.get("smile", 0.15) +
            face.sharpness_score * weights.get("sharpness", 0.20)
        )

    return max(faces, key=lambda f: f.overall_score)
```

---

## Implementation Phases

### Phase 1: Core Pipeline Engine
1. Define `PipelineStep` protocol and `StepMetadata`
2. Implement `PipelineContext` dataclass
3. Create `StepRegistry` with auto-discovery
4. Build `PipelineBuilder` with dependency resolution
5. Implement `PipelineExecutor` with progress callbacks

### Phase 2: Database & Persistence
1. Set up SQLAlchemy models (Album, PipelineRun, PipelineResult, EmbeddingCache)
2. Create database session management
3. Implement repository pattern for data access
4. Add embedding cache read/write utilities

### Phase 3: Migrate Existing Steps
1. Wrap existing analysis code as steps (`score_iqa`, `score_ava`)
2. Wrap MediaPipe detection as `detect_faces`
3. Wrap DINOv2/clustering as `extract_scene_embedding`, `cluster_scenes`
4. Wrap selection logic as `select_best_per_cluster`

### Phase 4: Add Face-Specific Steps
1. Implement `score_face_pose` (SixDRepNet integration)
2. Implement `score_face_eyes` (wrap existing)
3. Implement `score_face_smile` (wrap existing)
4. Implement `filter_pose`, `filter_portrait`
5. Implement `extract_face_embedding`, `cluster_faces`
6. Implement `select_best_face_per_identity`

### Phase 5: FastAPI Backend
1. Set up FastAPI app structure
2. Implement `/api/v1/steps` router
3. Implement `/api/v1/pipeline` router with background tasks
4. Add WebSocket progress endpoint
5. Add `/api/v1/albums` for upload/management
6. Add `/api/v1/results` for retrieval/export

### Phase 6: NiceGUI Frontend
1. Set up NiceGUI app structure
2. Create API client with WebSocket support
3. Build upload page
4. Build pipeline configuration page
5. Build progress display with real-time updates
6. Build results gallery with cluster visualization
7. Build export page

### Phase 7: AI Agent Integration
1. Document step schemas for agent discovery
2. Add natural language step descriptions
3. Create agent-friendly pipeline composition endpoint
4. Test with sample agent interactions

---

## Design Decisions

### 1. Sequential Execution (For Now)
Start with sequential step execution. Parallel execution adds complexity (resource contention, progress tracking, error handling) with limited benefit since ML models are the bottleneck. Can optimize later if profiling shows clear wins.

### 2. Configurable Error Handling
```python
class PipelineConfig:
    fail_fast: bool = True  # True = stop on first error, False = continue and collect all
```
- **Debug mode**: `fail_fast=True` - stop immediately, easier to diagnose
- **Production mode**: `fail_fast=False` - process what we can, report all errors at end

### 3. SQLite + SQLAlchemy for Persistence
SQLite is ideal for desktop/single-user photo organization:
- Zero configuration, file-based
- Easy backup (just copy the .db file)
- SQLAlchemy abstraction makes PostgreSQL migration trivial if needed later

**What to persist:**
- Albums (metadata, source paths, created_at)
- Pipeline runs (album_id, config used, status, timestamps)
- Results (selected images, cluster assignments, metrics)
- Embeddings cache (optional, for expensive recomputation avoidance)

### 4. All Parameters Configurable
Every threshold and weight:
- Has sensible default in step implementation
- Exposed via JSON schema for UI generation
- Editable in NiceGUI config panel
- Stored with pipeline run for reproducibility

### 5. Caching Strategy
- **Embeddings**: Cache at context level, keyed by (image_path, model_name, model_version)
- **Thumbnails**: Already cached by ImagePreprocessor
- **Invalidation**: Manual clear, or automatic when source image modified (check mtime)

### 6. Single Album Focus
Process one album at a time for simplicity. Queue multiple albums if needed (background task queue).

---

## Database Schema (SQLAlchemy)

```python
# sim_bench/api/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime

class Base(DeclarativeBase):
    pass


class Album(Base):
    """An uploaded photo album."""
    __tablename__ = "albums"

    id = Column(String, primary_key=True)  # UUID
    name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    image_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pipeline_runs = relationship("PipelineRun", back_populates="album")


class PipelineRun(Base):
    """A single pipeline execution."""
    __tablename__ = "pipeline_runs"

    id = Column(String, primary_key=True)  # UUID
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)

    # Configuration (stored for reproducibility)
    pipeline_name = Column(String)  # "default", "face_aware", "custom"
    steps = Column(JSON)  # ["step1", "step2", ...]
    step_configs = Column(JSON)  # {"step1": {...}, "step2": {...}}
    fail_fast = Column(Boolean, default=True)

    # Status
    status = Column(String, default="pending")  # pending, running, completed, failed
    current_step = Column(String, nullable=True)
    progress = Column(Float, default=0.0)
    error_message = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    album = relationship("Album", back_populates="pipeline_runs")
    result = relationship("PipelineResult", back_populates="run", uselist=False)


class PipelineResult(Base):
    """Results of a completed pipeline run."""
    __tablename__ = "pipeline_results"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=False)

    # Summary stats
    total_images = Column(Integer)
    filtered_images = Column(Integer)
    num_scene_clusters = Column(Integer)
    num_face_clusters = Column(Integer)
    num_people = Column(Integer)  # People identified
    num_selected = Column(Integer)

    # Detailed results (JSON for flexibility)
    scene_clusters = Column(JSON)  # {cluster_id: [image_paths]}
    face_clusters = Column(JSON)   # {scene_id: {face_id: [image_paths]}}
    selected_images = Column(JSON) # [image_paths]
    image_metrics = Column(JSON)   # {image_path: {scores...}}

    # People feature
    people_summary = Column(JSON)  # [{person_id, name, face_count, thumbnail_path}, ...]

    # Telemetry
    step_timings = Column(JSON)  # {step_name: duration_ms}
    total_duration_ms = Column(Integer)

    # Relationships
    run = relationship("PipelineRun", back_populates="result")


class Person(Base):
    """A person identified across an album (for People view)."""
    __tablename__ = "people"

    id = Column(String, primary_key=True)  # UUID
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=False)

    person_index = Column(Integer)  # 0, 1, 2... within this album
    name = Column(String, nullable=True)  # User-assigned name (optional)

    # Thumbnail (best face)
    thumbnail_image_path = Column(String)  # Source image containing best face
    thumbnail_face_index = Column(Integer)  # Which face in that image
    thumbnail_bbox = Column(JSON)  # Bounding box for cropping

    # Stats
    face_count = Column(Integer)  # Total face occurrences
    image_count = Column(Integer)  # Unique images containing this person

    # All face instances (JSON for flexibility)
    face_instances = Column(JSON)  # [{image_path, face_index, scores...}, ...]

    created_at = Column(DateTime, default=datetime.utcnow)


class EmbeddingCache(Base):
    """Cached embeddings for expensive recomputation avoidance."""
    __tablename__ = "embedding_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False)  # "dinov2", "arcface"
    model_version = Column(String, nullable=True)

    # Store as binary blob or base64 string
    embedding = Column(String, nullable=False)  # Base64 encoded numpy array
    embedding_dim = Column(Integer)

    # For invalidation
    image_mtime = Column(Float)  # os.path.getmtime() of source image
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite index for lookups
    __table_args__ = (
        Index('idx_embedding_lookup', 'image_path', 'model_name'),
    )
```

### Database Initialization

```python
# sim_bench/api/database/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

def get_database_url(db_path: Path | None = None) -> str:
    """Get SQLite database URL."""
    if db_path is None:
        db_path = Path.home() / ".sim_bench" / "sim_bench.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"

def create_db_engine(db_url: str = None):
    if db_url is None:
        db_url = get_database_url()
    return create_engine(db_url, echo=False)

def init_db(engine):
    """Create all tables."""
    from .models import Base
    Base.metadata.create_all(engine)

# Dependency injection for FastAPI
def get_session():
    engine = create_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

---

## File Changes Summary

### New Files
```
sim_bench/
├── api/
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py           # SQLAlchemy models
│   │   └── session.py          # Engine, session factory
│   │
├── pipeline/
│   ├── __init__.py
│   ├── base.py              # PipelineStep protocol, StepMetadata
│   ├── context.py           # PipelineContext
│   ├── registry.py          # StepRegistry
│   ├── builder.py           # PipelineBuilder with dependency resolution
│   ├── executor.py          # PipelineExecutor
│   └── steps/
│       ├── __init__.py
│       ├── discover.py
│       ├── score_iqa.py
│       ├── score_ava.py
│       ├── detect_faces.py
│       ├── score_face_pose.py
│       ├── score_face_eyes.py
│       ├── score_face_smile.py
│       ├── filter_quality.py
│       ├── filter_portrait.py
│       ├── filter_pose.py
│       ├── extract_scene_embedding.py
│       ├── extract_face_embedding.py
│       ├── cluster_scenes.py
│       ├── cluster_faces.py
│       ├── select_best.py
│       ├── select_best_face.py
│       ├── extract_all_faces.py    # People feature
│       ├── embed_all_faces.py      # People feature
│       ├── cluster_people.py       # People feature
│       └── select_best_per_person.py  # People feature
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies.py
│   ├── routers/
│   │   ├── albums.py
│   │   ├── pipeline.py
│   │   ├── steps.py
│   │   ├── results.py
│   │   ├── people.py           # People feature endpoints
│   │   └── websocket.py
│   ├── schemas/
│   │   ├── album.py
│   │   ├── pipeline.py
│   │   ├── step.py
│   │   └── result.py
│   └── services/
│       ├── album_service.py
│       ├── pipeline_service.py
│       └── result_service.py

app/
└── nicegui_app/
    ├── main.py
    ├── api_client.py
    ├── pages/
    ├── components/
    └── state/
```

### Files to Refactor
- `sim_bench/album/services/album_service.py` → Extract step logic to pipeline steps
- `sim_bench/album/stages.py` → Migrate to individual step classes
- `sim_bench/face_pipeline/` → Wrap as pipeline steps
- `sim_bench/portrait_analysis/` → Wrap as pipeline steps
