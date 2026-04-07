# Face Management - Module Specifications

## Table of Contents

1. [Backend Modules](#backend-modules)
   - [1.1 FaceOverride Model](#11-faceoverride-model)
   - [1.2 Face Schemas](#12-face-schemas)
   - [1.3 FaceService](#13-faceservice)
   - [1.4 Faces Router](#14-faces-router)
2. [Frontend Modules](#frontend-modules)
   - [2.1 Face Management Page](#21-face-management-page)
   - [2.2 FaceCard Component](#22-facecard-component)
   - [2.3 FaceGrid Component](#23-facegrid-component)
   - [2.4 ActionMenu Component](#24-actionmenu-component)
   - [2.5 NeedsHelpWizard Component](#25-needshelpwizard-component)
   - [2.6 PendingChangesPanel Component](#26-pendingchangespanel-component)
   - [2.7 PersonDetail Component](#27-persondetail-component)
   - [2.8 FaceDetailSheet Component](#28-facedetailsheet-component)
   - [2.9 Toasts Component](#29-toasts-component)
3. [API Client Extensions](#3-api-client-extensions)
4. [Dependency Graph](#4-dependency-graph)

---

# Backend Modules

---

## 1.1 FaceOverride Model

### Purpose
Persist user corrections to face classifications that survive pipeline re-runs.

### File
`sim_bench/api/database/models.py` (add to existing file)

### Dependencies
- Existing `Base` from SQLAlchemy
- Existing `Album`, `Person` models for relationships

### Schema

```python
class FaceOverride(Base):
    """Persistent face classification overrides."""
    __tablename__ = "face_overrides"

    # Primary key
    id: str                    # UUID

    # Context
    album_id: str              # FK to albums.id (required)
    run_id: str                # FK to pipeline_runs.id (optional - applies to future runs too)

    # Face identifier
    face_key: str              # Format: "image_path:face_N"

    # Classification
    status: str                # Enum: "assigned", "untagged", "not_a_face"
    person_id: str             # FK to persons.id (nullable - only for "assigned")

    # For "not_a_face" learning
    embedding: JSON            # numpy array as list (nullable)

    # Audit
    created_at: datetime
    created_by: str            # "user" or "system"
```

### Relationships

```python
# In FaceOverride
album = relationship("Album", back_populates="face_overrides")
person = relationship("Person", back_populates="face_overrides")

# In Album (add)
face_overrides = relationship("FaceOverride", back_populates="album")

# In Person (add)
face_overrides = relationship("FaceOverride", back_populates="person")
```

### Indexes

```python
Index("ix_face_overrides_album_face", "album_id", "face_key")
Index("ix_face_overrides_status", "status")
```

### Implementation Steps

1. [ ] Add `FaceOverride` class to models.py
2. [ ] Add relationships to `Album` and `Person`
3. [ ] Add indexes for performance
4. [ ] Test that table is created on startup

### Test Cases

```python
def test_face_override_create():
    """Can create a face override record."""

def test_face_override_unique_per_album_face():
    """Only one override per album+face_key combination."""

def test_face_override_cascade_delete_album():
    """Deleting album deletes its face overrides."""
```

---

## 1.2 Face Schemas

### Purpose
Pydantic models for API request/response serialization.

### File
`sim_bench/api/schemas/face.py` (new file)

### Dependencies
- `pydantic.BaseModel`
- `typing.Optional, List`

### Schemas

```python
# ============================================================
# Response Models
# ============================================================

class FaceInfo(BaseModel):
    """Complete information about a single face."""

    # Identity
    face_key: str              # "image_path:face_N"
    image_path: str            # Original image path
    face_index: int            # Face index within image

    # Visual
    thumbnail_base64: str      # Base64 encoded JPEG of cropped face
    bbox: dict                 # {x, y, w, h} in relative coords

    # Classification
    status: str                # "assigned" | "unassigned" | "untagged" | "not_a_face"
    person_id: Optional[str]
    person_name: Optional[str]

    # Assignment details
    assignment_method: Optional[str]    # "core" | "auto" | "user"
    assignment_confidence: Optional[float]

    # Quality metrics (for debug view)
    frontal_score: Optional[float]
    centroid_distance: Optional[float]
    exemplar_matches: Optional[int]

    class Config:
        from_attributes = True


class PersonDistance(BaseModel):
    """Distance from a face to a specific person."""

    person_id: str
    person_name: str
    thumbnail_base64: str      # Person's representative face

    centroid_distance: float
    exemplar_matches: int
    min_exemplar_distance: float
    would_attach: bool         # Would meet attachment criteria?


class BorderlineFace(BaseModel):
    """A face in the uncertainty zone needing user decision."""

    face: FaceInfo

    # Closest match
    closest_person_id: str
    closest_person_name: str
    closest_person_thumbnail: str
    distance: float

    # Uncertainty (0 = very uncertain, 1 = clear decision)
    uncertainty_score: float

    # Thresholds for context
    attach_threshold: float
    reject_threshold: float


class PersonSummary(BaseModel):
    """Summary of a person for listing."""

    person_id: str
    name: str
    thumbnail_base64: str
    face_count: int
    exemplar_count: int
    cluster_tightness: float   # Average internal distance


# ============================================================
# Request Models
# ============================================================

class FaceAction(BaseModel):
    """Single face action in a batch."""

    face_key: str
    action: str                # "assign" | "unassign" | "untag" | "not_a_face"
    target_person_id: Optional[str] = None
    new_person_name: Optional[str] = None  # For creating new person


class BatchChangeRequest(BaseModel):
    """Request to apply multiple face changes."""

    changes: List[FaceAction]
    recluster: bool = True     # Trigger reclustering after apply?


class BatchChangeResponse(BaseModel):
    """Response after applying batch changes."""

    applied_count: int
    failed_count: int
    failures: List[dict]       # [{face_key, error}, ...]

    # Reclustering results (if recluster=True)
    auto_assigned_count: int
    new_unassigned_count: int
```

### Implementation Steps

1. [ ] Create `sim_bench/api/schemas/face.py`
2. [ ] Define all request/response models
3. [ ] Add validation rules (e.g., action enum)
4. [ ] Export from `schemas/__init__.py`

### Test Cases

```python
def test_face_info_serialization():
    """FaceInfo serializes correctly."""

def test_batch_change_request_validation():
    """BatchChangeRequest validates action enum."""
```

---

## 1.3 FaceService

### Purpose
Business logic for face management operations.

### File
`sim_bench/api/services/face_service.py` (new file)

### Dependencies
- `sqlalchemy.orm.Session`
- `sim_bench.api.database.models.FaceOverride, Person, PipelineResult`
- `sim_bench.pipeline.steps.attachment_strategies.cosine_distance`
- `numpy`

### Class Interface

```python
class FaceService:
    """Service for face management operations."""

    def __init__(self, session: Session, logger: Logger = None):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    # ========================================================
    # Query Methods
    # ========================================================

    def get_all_faces(
        self,
        album_id: str,
        run_id: str,
        status_filter: List[str] = None
    ) -> List[FaceInfo]:
        """
        Get all faces for a pipeline run with their current status.

        Args:
            album_id: Album ID
            run_id: Pipeline run ID
            status_filter: Optional list of statuses to include

        Returns:
            List of FaceInfo objects

        Implementation:
            1. Load pipeline result for run_id
            2. Get all faces from result.image_metrics
            3. Get face overrides from FaceOverride table
            4. Merge: override status takes precedence
            5. Load person info for assigned faces
            6. Generate thumbnails (base64)
        """
        pass

    def get_face(
        self,
        album_id: str,
        run_id: str,
        face_key: str
    ) -> Optional[FaceInfo]:
        """Get single face with full details."""
        pass

    def get_face_distances(
        self,
        album_id: str,
        run_id: str,
        face_key: str
    ) -> List[PersonDistance]:
        """
        Get distances from a face to all people.

        Args:
            face_key: The face to measure from

        Returns:
            List of PersonDistance, sorted by centroid_distance ascending

        Implementation:
            1. Get face embedding from cache
            2. Get all people for this run
            3. For each person:
               a. Compute centroid distance
               b. Count exemplar matches
               c. Determine if would_attach
            4. Sort by distance
        """
        pass

    def get_borderline_faces(
        self,
        album_id: str,
        run_id: str,
        limit: int = 10
    ) -> List[BorderlineFace]:
        """
        Get faces in the uncertainty zone for user review.

        Implementation:
            1. Get all unassigned faces
            2. For each, compute distance to closest person
            3. Filter: keep those where attach_threshold < distance < reject_threshold
            4. Compute uncertainty_score = 1 - abs(distance - midpoint) / range
            5. Sort by uncertainty_score ascending (most uncertain first)
            6. Return top N
        """
        pass

    def get_people_summary(
        self,
        album_id: str,
        run_id: str
    ) -> List[PersonSummary]:
        """Get summary of all people for the sidebar."""
        pass

    # ========================================================
    # Mutation Methods
    # ========================================================

    def apply_single_change(
        self,
        album_id: str,
        run_id: str,
        change: FaceAction
    ) -> bool:
        """
        Apply a single face change immediately.

        Implementation:
            1. Validate change
            2. Create/update FaceOverride record
            3. Update Person.face_instances if needed
            4. Record UserEvent for undo
            5. Commit
        """
        pass

    def apply_batch_changes(
        self,
        album_id: str,
        run_id: str,
        changes: List[FaceAction],
        recluster: bool = True
    ) -> BatchChangeResponse:
        """
        Apply multiple changes and optionally recluster.

        Implementation:
            1. Validate all changes
            2. Apply each change (create FaceOverride, update Person)
            3. Record UserEvents
            4. If recluster:
               a. Load face embeddings
               b. Rebuild exemplars (include user-assigned faces)
               c. Re-run attachment on unassigned faces
               d. Update people_clusters
            5. Return summary
        """
        pass

    # ========================================================
    # Helper Methods
    # ========================================================

    def _get_face_embedding(
        self,
        face_key: str,
        run_id: str
    ) -> Optional[np.ndarray]:
        """Load face embedding from cache."""
        pass

    def _get_person_exemplars(
        self,
        person_id: str,
        run_id: str
    ) -> List[np.ndarray]:
        """Get exemplar embeddings for a person."""
        pass

    def _compute_uncertainty_score(
        self,
        distance: float,
        attach_threshold: float,
        reject_threshold: float
    ) -> float:
        """
        Compute how uncertain a face assignment is.

        0 = very uncertain (right at midpoint)
        1 = very certain (at threshold boundaries)
        """
        midpoint = (attach_threshold + reject_threshold) / 2
        range_half = (reject_threshold - attach_threshold) / 2
        return abs(distance - midpoint) / range_half

    def _generate_face_thumbnail(
        self,
        image_path: str,
        bbox: dict,
        size: int = 80
    ) -> str:
        """Generate base64 thumbnail for a face."""
        pass

    def _trigger_recluster(
        self,
        album_id: str,
        run_id: str
    ) -> dict:
        """
        Re-run identity refinement with updated overrides.

        Returns:
            {auto_assigned: int, still_unassigned: int}
        """
        pass
```

### Implementation Steps

1. [ ] Create `sim_bench/api/services/face_service.py`
2. [ ] Implement `__init__` with session/logger
3. [ ] Implement `get_all_faces` - load and merge data
4. [ ] Implement `get_face_distances` - compute distances
5. [ ] Implement `get_borderline_faces` - find uncertain faces
6. [ ] Implement `apply_single_change` - immediate apply
7. [ ] Implement `apply_batch_changes` - batch with recluster
8. [ ] Implement helper methods
9. [ ] Add comprehensive logging

### Test Cases

```python
class TestFaceServiceQueries:
    def test_get_all_faces_returns_all_statuses(self):
        """Returns faces with all status types."""

    def test_get_all_faces_applies_overrides(self):
        """Override status takes precedence over computed status."""

    def test_get_face_distances_sorted_by_distance(self):
        """Results sorted by centroid distance ascending."""

    def test_get_borderline_faces_filters_correctly(self):
        """Only returns faces in uncertainty zone."""

    def test_get_borderline_faces_sorted_by_uncertainty(self):
        """Most uncertain faces first."""


class TestFaceServiceMutations:
    def test_apply_single_change_creates_override(self):
        """Creates FaceOverride record."""

    def test_apply_single_change_updates_person(self):
        """Updates Person.face_instances."""

    def test_apply_single_change_records_event(self):
        """Creates UserEvent for undo."""

    def test_apply_batch_changes_atomic(self):
        """All changes succeed or all fail."""

    def test_apply_batch_recluster_assigns_new_faces(self):
        """Reclustering can auto-assign previously unassigned faces."""
```

---

## 1.4 Faces Router

### Purpose
REST API endpoints for face management operations.

### File
`sim_bench/api/routers/faces.py` (new file)

### Dependencies
- `fastapi.APIRouter, Depends, HTTPException, Query`
- `sim_bench.api.services.face_service.FaceService`
- `sim_bench.api.schemas.face.*`
- `sim_bench.api.database.session.get_session`

### Endpoints

```python
router = APIRouter(prefix="/api/v1/albums/{album_id}/runs/{run_id}/faces", tags=["faces"])


@router.get("", response_model=List[FaceInfo])
def list_faces(
    album_id: str,
    run_id: str,
    status: Optional[str] = Query(None, description="Filter by status (comma-separated)"),
    session: Session = Depends(get_session)
):
    """
    List all faces for a pipeline run.

    Query params:
        status: Comma-separated list of statuses to include
                Options: assigned, unassigned, untagged, not_a_face
                Default: all

    Returns:
        List of FaceInfo objects
    """
    pass


@router.get("/needs-help", response_model=List[BorderlineFace])
def get_needs_help(
    album_id: str,
    run_id: str,
    limit: int = Query(10, ge=1, le=50),
    session: Session = Depends(get_session)
):
    """
    Get borderline faces needing user decision.

    Returns faces where distance is between attach and reject thresholds,
    sorted by uncertainty (most uncertain first).
    """
    pass


@router.get("/people", response_model=List[PersonSummary])
def list_people(
    album_id: str,
    run_id: str,
    session: Session = Depends(get_session)
):
    """
    Get summary of all identified people.

    Returns list of people with face counts and thumbnails.
    """
    pass


@router.get("/{face_key}", response_model=FaceInfo)
def get_face(
    album_id: str,
    run_id: str,
    face_key: str,
    session: Session = Depends(get_session)
):
    """
    Get details for a single face.

    Note: face_key must be URL-encoded (colons and slashes).
    """
    pass


@router.get("/{face_key}/distances", response_model=List[PersonDistance])
def get_face_distances(
    album_id: str,
    run_id: str,
    face_key: str,
    session: Session = Depends(get_session)
):
    """
    Get distances from a face to all people.

    Useful for showing "Assign to" menu sorted by likelihood.
    """
    pass


@router.post("/{face_key}/action", response_model=FaceInfo)
def apply_face_action(
    album_id: str,
    run_id: str,
    face_key: str,
    action: FaceAction,
    session: Session = Depends(get_session)
):
    """
    Apply a single action to a face (live mode).

    Actions:
        - assign: Assign to existing person (requires target_person_id)
        - unassign: Remove from current person, move to unassigned
        - untag: Mark as "don't care"
        - not_a_face: Mark as false positive
    """
    pass


@router.post("/batch", response_model=BatchChangeResponse)
def apply_batch_changes(
    album_id: str,
    run_id: str,
    request: BatchChangeRequest,
    session: Session = Depends(get_session)
):
    """
    Apply multiple face changes at once (batch mode).

    If recluster=true, re-runs identity refinement after applying changes.
    This may auto-assign additional faces based on new exemplars.
    """
    pass


@router.post("/person", response_model=PersonSummary)
def create_person(
    album_id: str,
    run_id: str,
    name: str = Query(..., description="Name for new person"),
    face_keys: List[str] = Query(..., description="Face keys to assign"),
    session: Session = Depends(get_session)
):
    """
    Create a new person from selected faces.
    """
    pass
```

### Registration

```python
# In sim_bench/api/main.py
from sim_bench.api.routers import faces

app.include_router(faces.router)
```

### Implementation Steps

1. [ ] Create `sim_bench/api/routers/faces.py`
2. [ ] Implement `list_faces` endpoint
3. [ ] Implement `get_needs_help` endpoint
4. [ ] Implement `list_people` endpoint
5. [ ] Implement `get_face` endpoint
6. [ ] Implement `get_face_distances` endpoint
7. [ ] Implement `apply_face_action` endpoint
8. [ ] Implement `apply_batch_changes` endpoint
9. [ ] Implement `create_person` endpoint
10. [ ] Register router in main.py
11. [ ] Add OpenAPI documentation

### Test Cases

```python
class TestFacesRouter:
    def test_list_faces_no_filter(self):
        """Returns all faces when no status filter."""

    def test_list_faces_with_filter(self):
        """Filters by status correctly."""

    def test_get_needs_help_returns_borderline(self):
        """Only returns faces in uncertainty zone."""

    def test_get_face_not_found(self):
        """Returns 404 for unknown face_key."""

    def test_apply_face_action_assign(self):
        """Assign action updates face status."""

    def test_apply_batch_changes_recluster(self):
        """Batch with recluster=true triggers reclustering."""

    def test_create_person_from_faces(self):
        """Creates new person and assigns faces."""
```

---

# Frontend Modules

---

## 2.1 Face Management Page

### Purpose
Main page container that orchestrates all face management components.

### File
`app/streamlit/pages/face_management.py` (new file)

### Dependencies
- `streamlit`
- All face management components
- `app/streamlit/api_client.py`

### Page Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FACE MANAGEMENT                    [Mode ▼]  [N pending] [Apply]      │
├─────────────────────────────────────────────────────────────────────────┤
│  Album: [dropdown]    Run: [dropdown]                    [Refresh]     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │ Needs Help │ │   People   │ │ Unassigned │ │   Other ▼  │           │
│  │    (3)     │ │   (12)     │ │    (7)     │ │            │           │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                     [TAB CONTENT AREA]                                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  [First-time tip: Right-click or long-press faces for options]   [×]   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Session State

```python
def init_session_state():
    if "face_mgmt" not in st.session_state:
        st.session_state.face_mgmt = {
            # Context
            "album_id": None,
            "run_id": None,

            # UI State
            "active_tab": "needs_help",
            "batch_mode": True,
            "selection_mode": False,
            "selected_faces": set(),
            "show_debug": False,
            "show_first_time_tip": True,

            # Pending changes (batch mode)
            "pending_changes": [],

            # Wizard state
            "wizard_index": 0,
            "wizard_faces": [],

            # Data cache
            "faces_cache": None,
            "people_cache": None,
            "cache_time": None,
        }
```

### Main Function

```python
def render_face_management_page():
    """Main entry point for Face Management page."""

    init_session_state()

    # Header
    render_header()

    # Album/Run selector
    album_id, run_id = render_context_selector()
    if not album_id or not run_id:
        st.info("Select an album and pipeline run to manage faces.")
        return

    # Load data
    faces, people = load_data(album_id, run_id)

    # Count faces by status
    counts = count_by_status(faces)

    # Tabs
    tab = render_tabs(counts)

    # Tab content
    if tab == "needs_help":
        render_needs_help_tab(album_id, run_id)
    elif tab == "people":
        render_people_tab(people, faces)
    elif tab == "unassigned":
        render_faces_tab(faces, status="unassigned")
    elif tab == "untagged":
        render_faces_tab(faces, status="untagged")
    elif tab == "not_a_face":
        render_faces_tab(faces, status="not_a_face")

    # Pending changes panel (if batch mode and changes exist)
    if st.session_state.face_mgmt["batch_mode"]:
        render_pending_panel()

    # First-time tip
    render_first_time_tip()

    # Toast area (for notifications)
    render_toast_area()
```

### Implementation Steps

1. [ ] Create page file with skeleton
2. [ ] Implement `init_session_state()`
3. [ ] Implement `render_header()` with mode toggle
4. [ ] Implement `render_context_selector()` (reuse from results)
5. [ ] Implement `load_data()` with caching
6. [ ] Implement `render_tabs()` with counts
7. [ ] Implement `render_needs_help_tab()`
8. [ ] Implement `render_people_tab()`
9. [ ] Implement `render_faces_tab()` (generic for status)
10. [ ] Implement `render_pending_panel()`
11. [ ] Implement `render_first_time_tip()`
12. [ ] Add to sidebar navigation

### Test Cases

Manual testing checklist:
- [ ] Page loads without errors
- [ ] Album/run dropdowns populate correctly
- [ ] Tab switching works
- [ ] Counts update correctly
- [ ] Mode toggle persists
- [ ] Pending panel shows/hides correctly

---

## 2.2 FaceCard Component

### Purpose
Renders a single face with thumbnail, status badge, and action menu.

### File
`app/streamlit/components/face_management/face_card.py` (new file)

### Dependencies
- `streamlit`
- `PIL.Image`
- `base64`

### Interface

```python
def render_face_card(
    face: FaceInfo,
    people: List[PersonSummary],
    on_action: Callable[[str, FaceAction], None],
    show_checkbox: bool = False,
    is_selected: bool = False,
    on_select: Callable[[str, bool], None] = None,
    show_debug: bool = False,
    compact: bool = False,
) -> None:
    """
    Render a face card with thumbnail and actions.

    Args:
        face: Face information
        people: List of people for action menu
        on_action: Callback when action selected (face_key, action)
        show_checkbox: Show selection checkbox
        is_selected: Checkbox state
        on_select: Callback when checkbox toggled
        show_debug: Show distance/score info
        compact: Use compact layout (for grids)
    """
```

### Layout

```
Standard:                    Compact:
┌─────────────────┐          ┌───────────┐
│ [✓]        [...] │          │ [✓]  [...] │
│   ┌─────────┐   │          │  ┌─────┐  │
│   │         │   │          │  │     │  │
│   │  face   │   │          │  │face │  │
│   │         │   │          │  └─────┘  │
│   └─────────┘   │          │  [auto]   │
│     [auto]      │          └───────────┘
│  IMG_1234.jpg   │
│  d=0.31 ★0.85   │
└─────────────────┘
```

### Implementation

```python
def render_face_card(...):
    with st.container():
        # Top row: checkbox + menu
        col1, col2 = st.columns([3, 1])

        with col1:
            if show_checkbox:
                checked = st.checkbox(
                    "Select",
                    value=is_selected,
                    key=f"cb_{face.face_key}",
                    label_visibility="collapsed"
                )
                if checked != is_selected and on_select:
                    on_select(face.face_key, checked)

        with col2:
            # Menu button using popover
            with st.popover("⋯"):
                render_action_menu(face, people, on_action)

        # Face thumbnail
        if face.thumbnail_base64:
            st.image(
                f"data:image/jpeg;base64,{face.thumbnail_base64}",
                use_container_width=True
            )

        # Status badge
        badge_color = {
            "core": "green",
            "auto": "blue",
            "user": "orange",
        }.get(face.assignment_method, "gray")

        st.markdown(
            f'<span style="background:{badge_color};padding:2px 6px;'
            f'border-radius:4px;font-size:12px">{face.assignment_method or face.status}</span>',
            unsafe_allow_html=True
        )

        # Debug info
        if show_debug and not compact:
            st.caption(f"d={face.centroid_distance:.2f} ★{face.frontal_score:.2f}")

        # Filename (not in compact mode)
        if not compact:
            filename = Path(face.image_path).name
            st.caption(filename[:20] + "..." if len(filename) > 20 else filename)
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement basic layout
3. [ ] Add thumbnail rendering
4. [ ] Add status badge with colors
5. [ ] Add checkbox functionality
6. [ ] Integrate action menu popover
7. [ ] Add debug info display
8. [ ] Add compact mode variant

### Test Cases

- [ ] Renders thumbnail correctly
- [ ] Badge shows correct status
- [ ] Checkbox toggles correctly
- [ ] Menu opens on click
- [ ] Debug info shows/hides

---

## 2.3 FaceGrid Component

### Purpose
Renders a responsive grid of FaceCard components.

### File
`app/streamlit/components/face_management/face_grid.py` (new file)

### Dependencies
- `streamlit`
- `face_card.py`

### Interface

```python
def render_face_grid(
    faces: List[FaceInfo],
    people: List[PersonSummary],
    on_action: Callable[[str, FaceAction], None],
    columns: int = 6,
    enable_selection: bool = False,
    selected_faces: Set[str] = None,
    on_selection_change: Callable[[Set[str]], None] = None,
    show_debug: bool = False,
    empty_message: str = "No faces to display.",
) -> None:
    """
    Render a grid of face cards.

    Args:
        faces: List of faces to display
        people: List of people for action menus
        on_action: Callback for face actions
        columns: Number of columns (responsive)
        enable_selection: Show checkboxes for multi-select
        selected_faces: Set of selected face keys
        on_selection_change: Callback when selection changes
        show_debug: Show debug info on cards
        empty_message: Message when no faces
    """
```

### Implementation

```python
def render_face_grid(...):
    if not faces:
        st.info(empty_message)
        return

    # Selection toolbar (if enabled)
    if enable_selection and selected_faces:
        render_selection_toolbar(selected_faces, people, on_action)

    # Grid
    cols = st.columns(columns)

    for i, face in enumerate(faces):
        with cols[i % columns]:
            render_face_card(
                face=face,
                people=people,
                on_action=on_action,
                show_checkbox=enable_selection,
                is_selected=face.face_key in (selected_faces or set()),
                on_select=lambda key, checked: handle_select(key, checked, selected_faces, on_selection_change),
                show_debug=show_debug,
                compact=True,
            )


def render_selection_toolbar(selected: Set[str], people, on_action):
    """Toolbar for bulk actions on selected faces."""
    st.markdown(f"**{len(selected)} selected**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        person_id = st.selectbox(
            "Assign to",
            options=[""] + [p.person_id for p in people],
            format_func=lambda x: "Select person..." if x == "" else next(p.name for p in people if p.person_id == x),
            key="bulk_assign"
        )
        if person_id and st.button("Assign"):
            for face_key in selected:
                on_action(face_key, FaceAction(face_key=face_key, action="assign", target_person_id=person_id))

    with col2:
        if st.button("Untag All"):
            for face_key in selected:
                on_action(face_key, FaceAction(face_key=face_key, action="untag"))

    with col3:
        if st.button("Not a Face"):
            for face_key in selected:
                on_action(face_key, FaceAction(face_key=face_key, action="not_a_face"))

    with col4:
        if st.button("Clear Selection"):
            on_selection_change(set())
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement basic grid layout
3. [ ] Add empty state handling
4. [ ] Add selection toolbar
5. [ ] Implement bulk actions
6. [ ] Add responsive column handling

### Test Cases

- [ ] Grid renders correct number of columns
- [ ] Selection toolbar appears when faces selected
- [ ] Bulk assign works correctly
- [ ] Empty message shows when no faces

---

## 2.4 ActionMenu Component

### Purpose
Context menu for face actions, used in FaceCard popover.

### File
`app/streamlit/components/face_management/action_menu.py` (new file)

### Dependencies
- `streamlit`
- `FaceInfo, PersonSummary, FaceAction`

### Interface

```python
def render_action_menu(
    face: FaceInfo,
    people: List[PersonSummary],
    on_action: Callable[[str, FaceAction], None],
    distances: List[PersonDistance] = None,
) -> None:
    """
    Render action menu for a face.

    Args:
        face: The face to act on
        people: Available people to assign to
        on_action: Callback when action selected
        distances: Optional pre-computed distances (sorted)
    """
```

### Layout

```
┌───────────────────────────┐
│ Assign to:                │
│ ─────────────────────     │
│  ○ Mom (d=0.31)          │
│  ○ Dad (d=0.42)          │
│  ○ Person 3 (d=0.58)     │
│ ─────────────────────     │
│  + Create New Person     │
│  ↩ Move to Unassigned    │
├───────────────────────────┤
│  🚫 Untag (don't care)   │
│  ❌ Not a Face           │
└───────────────────────────┘
```

### Implementation

```python
def render_action_menu(face, people, on_action, distances=None):
    st.markdown("**Assign to:**")

    # Sort people by distance if available
    if distances:
        sorted_people = sorted(
            people,
            key=lambda p: next((d.centroid_distance for d in distances if d.person_id == p.person_id), 999)
        )
    else:
        sorted_people = people

    # Person options
    for person in sorted_people[:5]:  # Top 5
        dist_str = ""
        if distances:
            dist = next((d.centroid_distance for d in distances if d.person_id == person.person_id), None)
            if dist:
                dist_str = f" (d={dist:.2f})"

        if st.button(f"{person.name}{dist_str}", key=f"assign_{face.face_key}_{person.person_id}"):
            on_action(face.face_key, FaceAction(
                face_key=face.face_key,
                action="assign",
                target_person_id=person.person_id
            ))

    st.divider()

    # Create new person
    new_name = st.text_input("New person name", key=f"new_person_{face.face_key}")
    if st.button("+ Create New Person", key=f"create_{face.face_key}"):
        if new_name:
            on_action(face.face_key, FaceAction(
                face_key=face.face_key,
                action="assign",
                new_person_name=new_name
            ))

    # Unassign (if currently assigned)
    if face.status == "assigned":
        if st.button("↩ Remove from Person", key=f"unassign_{face.face_key}"):
            on_action(face.face_key, FaceAction(
                face_key=face.face_key,
                action="unassign"
            ))

    st.divider()

    # Other actions
    if st.button("🚫 Untag (don't care)", key=f"untag_{face.face_key}"):
        on_action(face.face_key, FaceAction(
            face_key=face.face_key,
            action="untag"
        ))

    if st.button("❌ Not a Face", key=f"notface_{face.face_key}"):
        on_action(face.face_key, FaceAction(
            face_key=face.face_key,
            action="not_a_face"
        ))
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement person list with distances
3. [ ] Add create new person flow
4. [ ] Add unassign option (contextual)
5. [ ] Add untag and not-a-face actions

### Test Cases

- [ ] People sorted by distance
- [ ] Create new person requires name
- [ ] Unassign only shows for assigned faces
- [ ] All actions trigger callback

---

## 2.5 NeedsHelpWizard Component

### Purpose
Tinder-style wizard for reviewing borderline faces.

### File
`app/streamlit/components/face_management/needs_help_wizard.py` (new file)

### Dependencies
- `streamlit`
- `BorderlineFace`

### Interface

```python
def render_needs_help_wizard(
    borderline_faces: List[BorderlineFace],
    on_decision: Callable[[str, str], None],  # (face_key, "yes"/"no"/"skip")
    on_close: Callable[[], None],
) -> None:
    """
    Render the "Needs Your Help" wizard.

    Args:
        borderline_faces: Faces needing user decision
        on_decision: Callback when user decides (face_key, decision)
        on_close: Callback when wizard closed
    """
```

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│              Is this the same person as "Mom"?               │
│                                                              │
│    ┌────────────────┐            ┌────────────────┐         │
│    │                │            │                │         │
│    │    Unknown     │     ?      │      Mom       │         │
│    │     face       │   ═══      │   (exemplar)   │         │
│    │                │            │                │         │
│    └────────────────┘            └────────────────┘         │
│                                                              │
│    Distance: 0.41                                            │
│    Threshold: 0.38 (attach) - 0.45 (reject)                 │
│    ━━━━━━━━━━━━━━━━●━━━━━━━━━━━                             │
│                     ↑                                        │
│                                                              │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│    │          │   │          │   │          │              │
│    │   ✗ No   │   │   Skip   │   │  ✓ Yes   │              │
│    │          │   │          │   │          │              │
│    └──────────┘   └──────────┘   └──────────┘              │
│                                                              │
│    ← → keys or swipe                                         │
│                                                              │
│    ●●●○○○○  3 of 7                              [Close]     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
def render_needs_help_wizard(borderline_faces, on_decision, on_close):
    if not borderline_faces:
        st.success("No faces need your help right now!")
        return

    # Get current face
    idx = st.session_state.face_mgmt.get("wizard_index", 0)
    if idx >= len(borderline_faces):
        st.success("All done! Thanks for your help.")
        if st.button("Close"):
            on_close()
        return

    face = borderline_faces[idx]

    # Question
    st.markdown(f"### Is this the same person as **{face.closest_person_name}**?")

    # Side-by-side comparison
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.image(
            f"data:image/jpeg;base64,{face.face.thumbnail_base64}",
            caption="Unknown face",
            use_container_width=True
        )

    with col2:
        st.markdown("<h1 style='text-align:center'>?</h1>", unsafe_allow_html=True)

    with col3:
        st.image(
            f"data:image/jpeg;base64,{face.closest_person_thumbnail}",
            caption=face.closest_person_name,
            use_container_width=True
        )

    # Distance visualization
    st.markdown(f"**Distance:** {face.distance:.2f}")
    render_threshold_bar(face.distance, face.attach_threshold, face.reject_threshold)

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✗ No", key="wizard_no", use_container_width=True):
            on_decision(face.face.face_key, "no")
            advance_wizard()

    with col2:
        if st.button("Skip", key="wizard_skip", use_container_width=True):
            on_decision(face.face.face_key, "skip")
            advance_wizard()

    with col3:
        if st.button("✓ Yes", key="wizard_yes", type="primary", use_container_width=True):
            on_decision(face.face.face_key, "yes")
            advance_wizard()

    # Progress
    st.progress((idx + 1) / len(borderline_faces))
    st.caption(f"{idx + 1} of {len(borderline_faces)}")

    # Close button
    if st.button("Close", key="wizard_close"):
        on_close()


def render_threshold_bar(distance, attach, reject):
    """Render visual threshold bar."""
    # Normalize to 0-100 range for display
    range_size = reject - attach
    position = ((distance - attach) / range_size) * 100

    st.markdown(f"""
    <div style="position:relative;height:20px;background:#eee;border-radius:10px">
        <div style="position:absolute;left:0;width:50%;height:100%;background:#90EE90;border-radius:10px 0 0 10px"></div>
        <div style="position:absolute;right:0;width:50%;height:100%;background:#FFB6C1;border-radius:0 10px 10px 0"></div>
        <div style="position:absolute;left:{position}%;top:-5px;width:4px;height:30px;background:black"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:12px">
        <span>Attach ({attach})</span>
        <span>Reject ({reject})</span>
    </div>
    """, unsafe_allow_html=True)


def advance_wizard():
    """Move to next face in wizard."""
    st.session_state.face_mgmt["wizard_index"] = st.session_state.face_mgmt.get("wizard_index", 0) + 1
    st.rerun()
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement basic layout with side-by-side
3. [ ] Add threshold bar visualization
4. [ ] Add action buttons
5. [ ] Add progress indicator
6. [ ] Add keyboard navigation (via JS)
7. [ ] Handle completion state

### Test Cases

- [ ] Shows correct face pair
- [ ] Buttons trigger correct decisions
- [ ] Progress updates correctly
- [ ] Completion message shows at end
- [ ] Close button works

---

## 2.6 PendingChangesPanel Component

### Purpose
Shows queued changes in batch mode with apply/discard actions.

### File
`app/streamlit/components/face_management/pending_changes_panel.py` (new file)

### Dependencies
- `streamlit`
- `FaceAction`

### Interface

```python
def render_pending_changes_panel(
    changes: List[FaceAction],
    on_apply: Callable[[], None],
    on_discard: Callable[[], None],
    on_remove: Callable[[int], None],  # Remove single change by index
) -> None:
    """
    Render panel showing pending changes.

    Args:
        changes: List of pending changes
        on_apply: Callback to apply all changes
        on_discard: Callback to discard all changes
        on_remove: Callback to remove single change
    """
```

### Layout

```
┌───────────────────────────────────────────────────────────┐
│ PENDING CHANGES (3)                                       │
├───────────────────────────────────────────────────────────┤
│ • Assign face_123 to "Mom"                          [×]  │
│ • Remove face_456 from "Dad"                        [×]  │
│ • Untag face_789                                    [×]  │
├───────────────────────────────────────────────────────────┤
│ [Discard All]                      [Apply & Recluster]   │
└───────────────────────────────────────────────────────────┘
```

### Implementation

```python
def render_pending_changes_panel(changes, on_apply, on_discard, on_remove):
    if not changes:
        return

    with st.container(border=True):
        st.markdown(f"### Pending Changes ({len(changes)})")

        for i, change in enumerate(changes):
            col1, col2 = st.columns([5, 1])

            with col1:
                description = format_change_description(change)
                st.markdown(f"• {description}")

            with col2:
                if st.button("×", key=f"remove_change_{i}"):
                    on_remove(i)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Discard All", type="secondary"):
                on_discard()

        with col2:
            if st.button("Apply & Recluster", type="primary"):
                on_apply()


def format_change_description(change: FaceAction) -> str:
    """Format a change for display."""
    face_short = change.face_key.split("/")[-1][:20]

    if change.action == "assign":
        if change.new_person_name:
            return f"Create person '{change.new_person_name}' with {face_short}"
        return f"Assign {face_short} to person"
    elif change.action == "unassign":
        return f"Remove {face_short} from person"
    elif change.action == "untag":
        return f"Untag {face_short}"
    elif change.action == "not_a_face":
        return f"Mark {face_short} as not a face"

    return f"{change.action}: {face_short}"
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement change list display
3. [ ] Add remove button per change
4. [ ] Add apply/discard buttons
5. [ ] Format change descriptions

### Test Cases

- [ ] Shows all pending changes
- [ ] Remove button removes single change
- [ ] Discard clears all changes
- [ ] Apply triggers callback

---

## 2.7 PersonDetail Component

### Purpose
Expanded view of a person showing all faces and exemplars.

### File
`app/streamlit/components/face_management/person_detail.py` (new file)

### Interface

```python
def render_person_detail(
    person: PersonSummary,
    faces: List[FaceInfo],
    exemplar_keys: List[str],
    on_action: Callable[[str, FaceAction], None],
    on_close: Callable[[], None],
) -> None:
    """
    Render expanded view of a person.
    """
```

### Layout

```
┌───────────────────────────────────────────────────────────────────┐
│ Person: Mom (12 faces)                                    [×]    │
├───────────────────────────────────────────────────────────────────┤
│ Name: [Mom                    ] [Rename]                         │
│                                                                   │
│ EXEMPLARS (5):                                                    │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                         │
│ │ ★   │ │ ★   │ │ ★   │ │ ★   │ │ ★   │                         │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                         │
│                                                                   │
│ ALL FACES (12):                                 [Select Mode]    │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                 │
│ │core │ │core │ │auto │ │auto │ │user │ │core │                 │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                 │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                 │
│ │core │ │core │ │auto │ │core │ │core │ │core │                 │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                 │
│                                                                   │
│ Cluster tightness: 0.23 (tight ✓)                                │
└───────────────────────────────────────────────────────────────────┘
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement header with name edit
3. [ ] Show exemplars with star indicator
4. [ ] Show all faces in grid
5. [ ] Add cluster tightness metric
6. [ ] Enable selection mode for faces

---

## 2.8 FaceDetailSheet Component

### Purpose
Mobile-friendly full-screen detail view for a face.

### File
`app/streamlit/components/face_management/face_detail_sheet.py` (new file)

### Interface

```python
def render_face_detail_sheet(
    face: FaceInfo,
    distances: List[PersonDistance],
    on_action: Callable[[str, FaceAction], None],
    on_close: Callable[[], None],
) -> None:
    """
    Render full-screen detail sheet for mobile.
    """
```

### Layout

```
┌─────────────────────────────────────────┐
│                                    [×]  │
│  ┌───────────────────────────────────┐  │
│  │                                   │  │
│  │         [Large Face Image]        │  │
│  │                                   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  From: IMG_1234.jpg                     │
│  Status: Unassigned                     │
│  Frontal Score: 0.85                    │
│                                         │
│  ─────────────────────────────────────  │
│                                         │
│  CLOSEST MATCHES:                       │
│  ┌─────────────────────────────────┐   │
│  │  Mom           d=0.31    [→]    │   │
│  ├─────────────────────────────────┤   │
│  │  Dad           d=0.42    [→]    │   │
│  ├─────────────────────────────────┤   │
│  │  Person 3      d=0.58    [→]    │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ─────────────────────────────────────  │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │       + Create New Person        │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │       🚫 Untag (don't care)     │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │       ❌ Not a Face              │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement full-screen modal layout
3. [ ] Show large face image
4. [ ] Show metadata
5. [ ] Show distances to all people
6. [ ] Add large tap-target buttons

---

## 2.9 Toasts Component

### Purpose
Toast notifications for action feedback.

### File
`app/streamlit/components/face_management/toasts.py` (new file)

### Interface

```python
def show_toast(
    message: str,
    type: str = "success",  # "success", "error", "info"
    duration: int = 3,
    undo_action: Callable[[], None] = None,
) -> None:
    """
    Show a toast notification.
    """

def render_toast_area() -> None:
    """
    Render area for displaying toasts.
    Must be called once per page.
    """
```

### Implementation

```python
def show_toast(message, type="success", duration=3, undo_action=None):
    """Add toast to queue."""
    if "toasts" not in st.session_state:
        st.session_state.toasts = []

    st.session_state.toasts.append({
        "message": message,
        "type": type,
        "undo_action": undo_action,
        "created": time.time(),
        "duration": duration,
    })


def render_toast_area():
    """Render active toasts."""
    if "toasts" not in st.session_state:
        return

    # Filter expired toasts
    now = time.time()
    st.session_state.toasts = [
        t for t in st.session_state.toasts
        if now - t["created"] < t["duration"]
    ]

    # Render active toasts
    for i, toast in enumerate(st.session_state.toasts):
        color = {"success": "green", "error": "red", "info": "blue"}[toast["type"]]

        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(
                    f'<div style="background:{color};padding:10px;border-radius:5px;color:white">'
                    f'{toast["message"]}</div>',
                    unsafe_allow_html=True
                )

            with col2:
                if toast["undo_action"]:
                    if st.button("Undo", key=f"undo_toast_{i}"):
                        toast["undo_action"]()
                        st.session_state.toasts.remove(toast)
                        st.rerun()
```

### Implementation Steps

1. [ ] Create component file
2. [ ] Implement toast queue in session state
3. [ ] Implement auto-dismiss logic
4. [ ] Add undo button support
5. [ ] Style different toast types

---

# 3. API Client Extensions

### File
`app/streamlit/api_client.py` (modify existing)

### New Methods

```python
# Add to AlbumAppClient class:

def get_faces(
    self,
    album_id: str,
    run_id: str,
    status: List[str] = None
) -> List[FaceInfo]:
    """Get all faces for a pipeline run."""
    params = {}
    if status:
        params["status"] = ",".join(status)

    response = self._get(f"/albums/{album_id}/runs/{run_id}/faces", params=params)
    return [FaceInfo(**f) for f in response]


def get_borderline_faces(
    self,
    album_id: str,
    run_id: str,
    limit: int = 10
) -> List[BorderlineFace]:
    """Get faces needing user decision."""
    response = self._get(
        f"/albums/{album_id}/runs/{run_id}/faces/needs-help",
        params={"limit": limit}
    )
    return [BorderlineFace(**f) for f in response]


def get_face_distances(
    self,
    album_id: str,
    run_id: str,
    face_key: str
) -> List[PersonDistance]:
    """Get distances from face to all people."""
    response = self._get(
        f"/albums/{album_id}/runs/{run_id}/faces/{quote(face_key)}/distances"
    )
    return [PersonDistance(**d) for d in response]


def apply_face_action(
    self,
    album_id: str,
    run_id: str,
    face_key: str,
    action: FaceAction
) -> FaceInfo:
    """Apply single face action (live mode)."""
    response = self._post(
        f"/albums/{album_id}/runs/{run_id}/faces/{quote(face_key)}/action",
        json=action.dict()
    )
    return FaceInfo(**response)


def apply_batch_changes(
    self,
    album_id: str,
    run_id: str,
    changes: List[FaceAction],
    recluster: bool = True
) -> BatchChangeResponse:
    """Apply multiple face changes (batch mode)."""
    response = self._post(
        f"/albums/{album_id}/runs/{run_id}/faces/batch",
        json={"changes": [c.dict() for c in changes], "recluster": recluster}
    )
    return BatchChangeResponse(**response)


def get_people_summary(
    self,
    album_id: str,
    run_id: str
) -> List[PersonSummary]:
    """Get summary of all people."""
    response = self._get(f"/albums/{album_id}/runs/{run_id}/faces/people")
    return [PersonSummary(**p) for p in response]
```

### Implementation Steps

1. [ ] Add FaceInfo, PersonDistance, etc. to models.py
2. [ ] Implement get_faces()
3. [ ] Implement get_borderline_faces()
4. [ ] Implement get_face_distances()
5. [ ] Implement apply_face_action()
6. [ ] Implement apply_batch_changes()
7. [ ] Implement get_people_summary()

---

# 4. Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IMPLEMENTATION ORDER                            │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: Backend Foundation
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ FaceOverride │───▶│ Face Schemas │───▶│ FaceService  │───▶│ Faces Router │
│    Model     │    │              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘

Phase 2: Frontend Foundation
┌──────────────┐    ┌──────────────┐
│ API Client   │───▶│ Face Mgmt    │
│ Extensions   │    │ Page Skeleton│
└──────────────┘    └──────────────┘

Phase 3: Core Components
                    ┌──────────────┐
                    │  FaceCard    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌────────────┐ ┌──────────┐ ┌────────────┐
       │ ActionMenu │ │ FaceGrid │ │   Toasts   │
       └────────────┘ └──────────┘ └────────────┘

Phase 4: Advanced Components
       ┌────────────────────┐    ┌────────────────────┐
       │ NeedsHelpWizard    │    │ PendingChangesPanel│
       └────────────────────┘    └────────────────────┘

Phase 5: Detail Views
       ┌────────────────────┐    ┌────────────────────┐
       │   PersonDetail     │    │  FaceDetailSheet   │
       └────────────────────┘    └────────────────────┘
```

---

# Summary

| Module | File | Priority | Effort |
|--------|------|----------|--------|
| FaceOverride Model | models.py | HIGH | S |
| Face Schemas | schemas/face.py | HIGH | S |
| FaceService | services/face_service.py | HIGH | L |
| Faces Router | routers/faces.py | HIGH | M |
| API Client Extensions | api_client.py | HIGH | M |
| Face Management Page | pages/face_management.py | HIGH | L |
| FaceCard | face_card.py | HIGH | M |
| FaceGrid | face_grid.py | HIGH | M |
| ActionMenu | action_menu.py | HIGH | M |
| Toasts | toasts.py | MEDIUM | S |
| PendingChangesPanel | pending_changes_panel.py | MEDIUM | M |
| NeedsHelpWizard | needs_help_wizard.py | MEDIUM | L |
| PersonDetail | person_detail.py | LOW | M |
| FaceDetailSheet | face_detail_sheet.py | LOW | M |

**Effort Key:** S = Small (< 1 hour), M = Medium (1-3 hours), L = Large (3+ hours)

---

*Document Version: 1.0*
*Created: 2026-02-14*
