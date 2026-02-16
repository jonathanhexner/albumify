# Face Management UI/UX Implementation Plan

## Overview

This document outlines the complete implementation plan for the Face Management page in the sim-bench application. The page enables users to review, correct, and refine face clustering results.

---

## Table of Contents

1. [Goals & Success Criteria](#1-goals--success-criteria)
2. [Architecture Overview](#2-architecture-overview)
3. [Backend Requirements](#3-backend-requirements)
4. [Frontend Components](#4-frontend-components)
5. [State Management](#5-state-management)
6. [Implementation Phases](#6-implementation-phases)
7. [Detailed Task Breakdown](#7-detailed-task-breakdown)
8. [Testing Strategy](#8-testing-strategy)
9. [Accessibility Requirements](#9-accessibility-requirements)
10. [Open Questions](#10-open-questions)

---

## 1. Goals & Success Criteria

### Primary Goals
- Allow users to correct face clustering mistakes
- Provide transparency into why faces were grouped/not grouped
- Enable batch corrections with deferred reclustering
- Support "Needs Your Help" recommender for borderline cases

### Success Criteria
- [ ] User can assign unassigned face to existing person
- [ ] User can remove face from person (move to unassigned)
- [ ] User can reassign face from one person to another
- [ ] User can create new person from unassigned faces
- [ ] User can mark faces as "Untag" (don't care)
- [ ] User can mark false positives as "Not a Face"
- [ ] User can undo any action
- [ ] User can answer "Needs Your Help" prompts
- [ ] Changes queue in batch mode until "Apply" clicked
- [ ] UI works on desktop (mouse) and tablet (touch)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Streamlit)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Face Management Page                          │   │
│  │                 app/streamlit/pages/face_management.py           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Components                                │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ NeedsHelp    │ │ PeopleGrid   │ │ FaceGrid     │             │   │
│  │  │ Wizard       │ │              │ │              │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ FaceCard     │ │ ActionMenu   │ │ PendingPanel │             │   │
│  │  │              │ │ (context)    │ │              │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      API Client                                  │   │
│  │                 app/streamlit/api_client.py                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         Routers                                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ events.py    │ │ people.py    │ │ faces.py     │             │   │
│  │  │ (exists)     │ │ (exists)     │ │ (NEW)        │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Services                                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ EventService │ │ PeopleService│ │ FaceService  │             │   │
│  │  │ (exists)     │ │ (exists)     │ │ (NEW)        │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Database                                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ Person       │ │ UserEvent    │ │ FaceOverride │             │   │
│  │  │ (exists)     │ │ (exists)     │ │ (NEW)        │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Backend Requirements

### 3.1 New Database Model: FaceOverride

```python
# sim_bench/api/database/models.py

class FaceOverride(Base):
    """Persistent face classification overrides."""
    __tablename__ = "face_overrides"

    id = Column(String, primary_key=True)
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=True)

    face_key = Column(String, nullable=False)  # "image_path:face_N"

    # Classification
    status = Column(String, nullable=False)  # "assigned", "untagged", "not_a_face"
    person_id = Column(String, ForeignKey("persons.id"), nullable=True)

    # For "not_a_face" - store embedding for similarity rejection
    embedding = Column(JSON, nullable=True)  # numpy array as list

    # Audit
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, default="user")

    # Relationships
    album = relationship("Album")
    person = relationship("Person")
```

### 3.2 New Router: faces.py

```python
# sim_bench/api/routers/faces.py

# Endpoints needed:

GET  /api/v1/albums/{album_id}/runs/{run_id}/faces
     # Returns all faces with their status, person assignment, distances
     # Query params: ?status=unassigned,assigned,untagged,not_a_face

GET  /api/v1/albums/{album_id}/runs/{run_id}/faces/needs-help
     # Returns borderline faces for "Needs Your Help" wizard
     # Sorted by uncertainty (closest to threshold)

GET  /api/v1/albums/{album_id}/runs/{run_id}/faces/{face_key}
     # Returns single face with full details including distances to all people

GET  /api/v1/albums/{album_id}/runs/{run_id}/faces/{face_key}/distances
     # Returns distances to all people (for context menu sorting)

POST /api/v1/albums/{album_id}/runs/{run_id}/faces/batch
     # Apply multiple changes at once
     # Body: { changes: [...], recluster: true/false }
```

### 3.3 New Service: FaceService

```python
# sim_bench/api/services/face_service.py

class FaceService:
    """Service for face management operations."""

    def get_all_faces(self, album_id, run_id, status_filter=None) -> List[FaceInfo]:
        """Get all faces with their current status and assignments."""
        pass

    def get_borderline_faces(self, album_id, run_id, limit=10) -> List[BorderlineFace]:
        """Get faces in the uncertainty zone for user review."""
        pass

    def get_face_distances(self, face_key, album_id, run_id) -> List[PersonDistance]:
        """Get distances from a face to all people."""
        pass

    def apply_batch_changes(self, album_id, run_id, changes, recluster=False):
        """Apply multiple face changes, optionally trigger reclustering."""
        pass

    def get_not_a_face_embeddings(self, album_id) -> List[np.ndarray]:
        """Get embeddings of false positives for similarity filtering."""
        pass
```

### 3.4 API Response Models

```python
# sim_bench/api/schemas/face.py

class FaceInfo(BaseModel):
    face_key: str
    image_path: str
    face_index: int
    thumbnail_url: str  # Base64 or path to cropped face

    status: str  # "assigned", "unassigned", "untagged", "not_a_face"
    person_id: Optional[str]
    person_name: Optional[str]

    assignment_method: Optional[str]  # "core", "auto", "user"
    assignment_confidence: Optional[float]

    # Debug info
    centroid_distance: Optional[float]
    exemplar_matches: Optional[int]
    frontal_score: Optional[float]

class BorderlineFace(BaseModel):
    face: FaceInfo
    closest_person_id: str
    closest_person_name: str
    closest_person_thumbnail: str
    distance: float
    uncertainty_score: float  # How close to threshold (0 = very uncertain)

class PersonDistance(BaseModel):
    person_id: str
    person_name: str
    person_thumbnail: str
    centroid_distance: float
    exemplar_matches: int
    would_attach: bool  # If user assigns, would it meet criteria?

class BatchChange(BaseModel):
    face_key: str
    action: str  # "assign", "unassign", "untag", "not_a_face"
    target_person_id: Optional[str]
    new_person_name: Optional[str]  # If creating new person
```

---

## 4. Frontend Components

### 4.1 Page Structure

```
app/streamlit/pages/face_management.py      # Main page
app/streamlit/components/face_management/
    __init__.py
    needs_help_wizard.py                     # "Needs Your Help" component
    people_grid.py                           # Grid of people with faces
    face_grid.py                             # Grid of faces (reusable)
    face_card.py                             # Single face card with menu
    action_menu.py                           # Context menu component
    pending_changes_panel.py                 # Shows queued changes
    person_detail_modal.py                   # Expanded view of person
    face_detail_sheet.py                     # Mobile detail sheet
```

### 4.2 Component Specifications

#### 4.2.1 FaceCard

```python
# app/streamlit/components/face_management/face_card.py

def render_face_card(
    face: FaceInfo,
    show_checkbox: bool = False,
    is_selected: bool = False,
    on_select: Callable = None,
    on_action: Callable = None,
    show_debug_info: bool = False,
):
    """
    Renders a single face card with:
    - Face thumbnail (square crop)
    - Assignment badge (core/auto/user)
    - Checkbox (if in selection mode)
    - "..." menu button
    - Optional debug info (distance, frontal score)

    Layout:
    ┌─────────────┐
    │ [✓]    [...] │  <- checkbox + menu
    │  ┌───────┐  │
    │  │ face  │  │
    │  └───────┘  │
    │   [auto]    │  <- badge
    │  d=0.31     │  <- debug info (optional)
    └─────────────┘
    """
```

#### 4.2.2 ActionMenu

```python
# app/streamlit/components/face_management/action_menu.py

def render_action_menu(
    face: FaceInfo,
    people: List[PersonInfo],
    distances: List[PersonDistance],  # Sorted by distance
    on_action: Callable,
):
    """
    Renders context menu for face actions.

    Uses st.popover() for desktop, full modal for mobile.

    Menu structure:
    ┌─────────────────────────┐
    │ Assign to:              │
    │   ○ Mom (0.31)         │  <- sorted by distance
    │   ○ Dad (0.42)         │
    │   ○ Person 3 (0.58)    │
    │   ──────────────────   │
    │   + Create New Person   │
    │   ↩ Move to Unassigned │
    ├─────────────────────────┤
    │ 🚫 Untag (don't care)   │
    │ ❌ Not a Face           │
    └─────────────────────────┘
    """
```

#### 4.2.3 NeedsHelpWizard

```python
# app/streamlit/components/face_management/needs_help_wizard.py

def render_needs_help_wizard(
    borderline_faces: List[BorderlineFace],
    on_decision: Callable,  # (face_key, decision: "yes"/"no"/"skip") -> None
    on_close: Callable,
):
    """
    Renders the "Needs Your Help" wizard.

    Tinder-style one-at-a-time interface:
    ┌──────────────────────────────────────────┐
    │  Is this the same person as "Mom"?       │
    │                                          │
    │   ┌────────┐        ┌────────┐          │
    │   │ unknown│   ≟    │  Mom   │          │
    │   │  face  │        │        │          │
    │   └────────┘        └────────┘          │
    │                                          │
    │   Distance: 0.41 (borderline)            │
    │                                          │
    │   [✗ No]    [Skip]    [✓ Yes]           │
    │                                          │
    │   ● ○ ○ ○ ○  1 of 5                      │
    └──────────────────────────────────────────┘

    Supports:
    - Arrow keys (← No, → Yes, ↓ Skip)
    - Swipe gestures (via JS injection)
    - Progress indicator
    """
```

#### 4.2.4 PendingChangesPanel

```python
# app/streamlit/components/face_management/pending_changes_panel.py

def render_pending_changes_panel(
    changes: List[PendingChange],
    on_apply: Callable,
    on_discard: Callable,
    on_remove_change: Callable,
):
    """
    Shows queued changes in batch mode.

    ┌─────────────────────────────────────────────┐
    │ PENDING CHANGES (3)                         │
    ├─────────────────────────────────────────────┤
    │ • Assign IMG_1234:face_0 to Mom       [×]  │
    │ • Remove IMG_5678:face_1 from Dad     [×]  │
    │ • Untag IMG_9999:face_0               [×]  │
    ├─────────────────────────────────────────────┤
    │ [Discard All]          [Apply & Recluster] │
    └─────────────────────────────────────────────┘
    """
```

### 4.3 Page Layout

```python
# app/streamlit/pages/face_management.py

def render_face_management_page():
    """
    Main page layout.
    """

    # Header with mode toggle and pending count
    # ┌─────────────────────────────────────────────────────┐
    # │ FACE MANAGEMENT    [Batch ▼]  [3 pending] [Apply]  │
    # └─────────────────────────────────────────────────────┘

    # Tabs
    # ┌────────────┬────────────┬────────────┬────────────┐
    # │ Needs Help │   People   │ Unassigned │   Other    │
    # │    (3)     │   (12)     │    (7)     │     ▼      │
    # └────────────┴────────────┴────────────┴────────────┘

    # Tab content (only one visible at a time)

    # First-time tooltip (stored in session state)

    # Toast notifications area
```

---

## 5. State Management

### 5.1 Session State Structure

```python
# Session state keys for face management

st.session_state.face_mgmt = {
    # Current album/run context
    "album_id": str,
    "run_id": str,

    # UI state
    "active_tab": str,  # "needs_help", "people", "unassigned", "untagged", "not_a_face"
    "selection_mode": bool,
    "selected_faces": Set[str],  # Set of face_keys
    "show_debug_info": bool,
    "show_first_time_tip": bool,

    # Mode
    "batch_mode": bool,  # True = batch, False = live

    # Pending changes (batch mode)
    "pending_changes": List[Dict],  # [{face_key, action, target_person_id, ...}]

    # Wizard state
    "needs_help_index": int,  # Current position in wizard
    "needs_help_faces": List[BorderlineFace],  # Cached borderline faces

    # Expanded views
    "expanded_person_id": Optional[str],  # Which person is expanded
    "detail_face_key": Optional[str],  # Face shown in detail sheet (mobile)

    # Cache
    "people_cache": List[PersonInfo],
    "faces_cache": Dict[str, List[FaceInfo]],  # Keyed by status
    "last_refresh": datetime,
}
```

### 5.2 Change Flow

```
User Action
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Is Batch Mode?                                          │
│                                                         │
│   YES                              NO (Live Mode)       │
│    │                                    │               │
│    ▼                                    ▼               │
│ Add to pending_changes            Call API immediately  │
│ Show toast "Change queued"        Show toast "Applied"  │
│ Update pending count              Refresh face list     │
│                                   (auto-recluster if    │
│                                    new assignments)     │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼ (When user clicks "Apply" in batch mode)
┌─────────────────────────────────────────────────────────┐
│ POST /api/v1/.../faces/batch                            │
│ Body: { changes: [...], recluster: true }               │
│                                                         │
│ Backend:                                                │
│   1. Apply all UserEvent records                        │
│   2. Update Person.face_instances                       │
│   3. Re-run identity_refinement step                    │
│   4. Return updated face list + newly auto-assigned     │
│                                                         │
│ Frontend:                                               │
│   1. Clear pending_changes                              │
│   2. Refresh all caches                                 │
│   3. Show toast "Applied N changes, M faces auto-assigned"
│   4. Refresh needs_help list                            │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Phases

### Phase 1: Backend Foundation (Priority: HIGH)
- [ ] Create FaceOverride model
- [ ] Create faces.py router with basic endpoints
- [ ] Create FaceService with core methods
- [ ] Add face distance calculation utilities
- [ ] Add borderline face detection logic

### Phase 2: Basic UI Structure (Priority: HIGH)
- [ ] Create face_management.py page
- [ ] Add to sidebar navigation
- [ ] Implement tab structure
- [ ] Create FaceCard component
- [ ] Create basic FaceGrid component

### Phase 3: Core Actions (Priority: HIGH)
- [ ] Implement ActionMenu component
- [ ] Implement assign action
- [ ] Implement unassign/remove action
- [ ] Implement untag action
- [ ] Implement not-a-face action
- [ ] Add toast notifications

### Phase 4: Batch Mode (Priority: MEDIUM)
- [ ] Implement PendingChangesPanel
- [ ] Add batch/live mode toggle
- [ ] Implement batch apply endpoint
- [ ] Add recluster trigger

### Phase 5: Needs Help Wizard (Priority: MEDIUM)
- [ ] Implement borderline detection algorithm
- [ ] Create NeedsHelpWizard component
- [ ] Add keyboard navigation
- [ ] Add progress tracking

### Phase 6: Debug & Transparency (Priority: MEDIUM)
- [ ] Add debug info toggle
- [ ] Show distances in UI
- [ ] Show exemplars for each person
- [ ] Add "why not assigned" explanations

### Phase 7: Polish & Mobile (Priority: LOW)
- [ ] Mobile detail sheet
- [ ] Touch gesture support
- [ ] Accessibility audit
- [ ] Performance optimization
- [ ] First-time user tips

---

## 7. Detailed Task Breakdown

### Phase 1: Backend Foundation

#### Task 1.1: FaceOverride Model
**File**: `sim_bench/api/database/models.py`
**Effort**: Small
```
- Add FaceOverride class
- Add relationship to Album, Person
- Create migration (if using Alembic) or rely on auto-create
```

#### Task 1.2: Face Schemas
**File**: `sim_bench/api/schemas/face.py` (NEW)
**Effort**: Small
```
- Create FaceInfo model
- Create BorderlineFace model
- Create PersonDistance model
- Create BatchChange model
- Create BatchChangeResponse model
```

#### Task 1.3: FaceService
**File**: `sim_bench/api/services/face_service.py` (NEW)
**Effort**: Medium
```
- Implement get_all_faces()
- Implement get_borderline_faces()
- Implement get_face_distances()
- Implement apply_batch_changes()
- Add helper: _compute_uncertainty_score()
- Add helper: _get_face_embedding()
```

#### Task 1.4: Faces Router
**File**: `sim_bench/api/routers/faces.py` (NEW)
**Effort**: Medium
```
- GET /faces endpoint
- GET /faces/needs-help endpoint
- GET /faces/{face_key} endpoint
- GET /faces/{face_key}/distances endpoint
- POST /faces/batch endpoint
- Register router in main.py
```

#### Task 1.5: Distance Utilities
**File**: `sim_bench/pipeline/steps/attachment_strategies.py`
**Effort**: Small
```
- Ensure cosine_distance is importable
- Add uncertainty_score calculation
- Add threshold comparison helpers
```

### Phase 2: Basic UI Structure

#### Task 2.1: Page Skeleton
**File**: `app/streamlit/pages/face_management.py` (NEW)
**Effort**: Medium
```
- Create page with header
- Add album/run selector (reuse from results page)
- Add tab structure
- Initialize session state
- Add placeholder content per tab
```

#### Task 2.2: Navigation
**File**: `app/streamlit/components/sidebar.py`
**Effort**: Small
```
- Add "Face Management" to navigation
- Add icon
- Link to new page
```

#### Task 2.3: API Client Methods
**File**: `app/streamlit/api_client.py`
**Effort**: Medium
```
- Add get_faces() method
- Add get_borderline_faces() method
- Add get_face_distances() method
- Add apply_face_changes() method
```

#### Task 2.4: FaceCard Component
**File**: `app/streamlit/components/face_management/face_card.py` (NEW)
**Effort**: Medium
```
- Render face thumbnail
- Render assignment badge
- Render checkbox (optional)
- Render menu button
- Handle click events
```

#### Task 2.5: FaceGrid Component
**File**: `app/streamlit/components/face_management/face_grid.py` (NEW)
**Effort**: Medium
```
- Render grid of FaceCards
- Handle selection mode
- Pass through callbacks
- Responsive column count
```

### Phase 3: Core Actions

#### Task 3.1: ActionMenu Component
**File**: `app/streamlit/components/face_management/action_menu.py` (NEW)
**Effort**: Medium
```
- Use st.popover() for menu
- List people sorted by distance
- Show distance values
- Handle all action types
- Create new person flow
```

#### Task 3.2: Action Handlers
**File**: `app/streamlit/pages/face_management.py`
**Effort**: Medium
```
- Implement handle_assign()
- Implement handle_unassign()
- Implement handle_untag()
- Implement handle_not_a_face()
- Implement handle_create_person()
- Wire up to ActionMenu callbacks
```

#### Task 3.3: Toast Notifications
**File**: `app/streamlit/components/face_management/toasts.py` (NEW)
**Effort**: Small
```
- Create toast component
- Success/error variants
- Undo button in toast
- Auto-dismiss timer
```

### Phase 4: Batch Mode

#### Task 4.1: PendingChangesPanel
**File**: `app/streamlit/components/face_management/pending_changes_panel.py` (NEW)
**Effort**: Medium
```
- List pending changes
- Remove individual change
- Apply all button
- Discard all button
- Change count badge
```

#### Task 4.2: Mode Toggle
**File**: `app/streamlit/pages/face_management.py`
**Effort**: Small
```
- Add batch/live toggle in header
- Persist preference
- Update action handlers for mode
```

#### Task 4.3: Batch Apply
**File**: `app/streamlit/pages/face_management.py`
**Effort**: Medium
```
- Collect all pending changes
- Call batch API endpoint
- Handle response
- Show summary toast
- Refresh all data
```

### Phase 5: Needs Help Wizard

#### Task 5.1: Borderline Detection
**File**: `sim_bench/api/services/face_service.py`
**Effort**: Medium
```
- Implement get_borderline_faces()
- Calculate uncertainty scores
- Sort by uncertainty
- Include closest person info
```

#### Task 5.2: NeedsHelpWizard Component
**File**: `app/streamlit/components/face_management/needs_help_wizard.py` (NEW)
**Effort**: Large
```
- Side-by-side comparison UI
- Yes/No/Skip buttons
- Progress indicator
- Navigation between items
- Keyboard shortcuts (JS injection)
```

#### Task 5.3: Wizard Integration
**File**: `app/streamlit/pages/face_management.py`
**Effort**: Medium
```
- Load borderline faces
- Handle wizard decisions
- Update pending changes
- Refresh when complete
```

### Phase 6: Debug & Transparency

#### Task 6.1: Debug Toggle
**File**: `app/streamlit/pages/face_management.py`
**Effort**: Small
```
- Add "Show Debug Info" checkbox
- Pass to FaceCard components
```

#### Task 6.2: Distance Display
**File**: `app/streamlit/components/face_management/face_card.py`
**Effort**: Small
```
- Show centroid distance
- Show exemplar matches
- Show frontal score
```

#### Task 6.3: Exemplar Display
**File**: `app/streamlit/components/face_management/person_detail.py` (NEW)
**Effort**: Medium
```
- Show exemplar faces for person
- Show cluster tightness
- Show face count by method
```

#### Task 6.4: Why Not Assigned
**File**: `app/streamlit/components/face_management/face_detail.py` (NEW)
**Effort**: Medium
```
- For unassigned faces
- Show distances to all people
- Explain why not attached
- Show threshold comparison
```

### Phase 7: Polish & Mobile

#### Task 7.1: Mobile Detail Sheet
**File**: `app/streamlit/components/face_management/face_detail_sheet.py` (NEW)
**Effort**: Medium
```
- Full-screen modal for mobile
- Large tap targets
- All actions available
```

#### Task 7.2: First-Time Tips
**File**: `app/streamlit/pages/face_management.py`
**Effort**: Small
```
- Show tip on first visit
- "Right-click or long-press for options"
- Dismiss and remember
```

#### Task 7.3: Accessibility
**Effort**: Medium
```
- Keyboard navigation
- ARIA labels
- Focus indicators
- Screen reader testing
```

---

## 8. Testing Strategy

### 8.1 Backend Tests

```python
# tests/api/test_face_service.py

class TestFaceService:
    def test_get_all_faces_returns_correct_structure(self):
        pass

    def test_get_borderline_faces_sorted_by_uncertainty(self):
        pass

    def test_apply_batch_changes_updates_person_records(self):
        pass

    def test_apply_batch_changes_triggers_recluster(self):
        pass

    def test_untag_face_removes_from_person(self):
        pass

    def test_not_a_face_stores_embedding(self):
        pass

# tests/api/test_faces_router.py

class TestFacesRouter:
    def test_get_faces_filters_by_status(self):
        pass

    def test_get_face_distances_returns_all_people(self):
        pass

    def test_batch_endpoint_validates_input(self):
        pass
```

### 8.2 Frontend Tests

Manual testing checklist:
- [ ] Tab navigation works
- [ ] Face cards render correctly
- [ ] Context menu opens on click
- [ ] Assign action works (batch mode)
- [ ] Assign action works (live mode)
- [ ] Pending changes panel shows changes
- [ ] Apply changes triggers recluster
- [ ] Needs Help wizard navigates correctly
- [ ] Keyboard shortcuts work
- [ ] Mobile: long-press opens menu
- [ ] Toast notifications appear and dismiss

### 8.3 Integration Tests

```python
# tests/integration/test_face_management_flow.py

class TestFaceManagementFlow:
    def test_assign_unassigned_face_to_person(self):
        """
        1. Get unassigned faces
        2. Assign one to a person
        3. Verify person's face_instances updated
        4. Verify face no longer in unassigned
        """
        pass

    def test_batch_mode_queues_changes(self):
        """
        1. Enable batch mode
        2. Make 3 changes
        3. Verify nothing persisted yet
        4. Apply changes
        5. Verify all persisted
        """
        pass

    def test_recluster_assigns_additional_faces(self):
        """
        1. Get unassigned face A close to person X
        2. Assign face B (very good exemplar) to person X
        3. Trigger recluster
        4. Verify face A auto-assigned to X
        """
        pass
```

---

## 9. Accessibility Requirements

### 9.1 Keyboard Navigation

| Key | Action |
|-----|--------|
| Tab | Move between faces |
| Enter | Open action menu |
| Arrow keys | Navigate menu |
| Escape | Close menu |
| Space | Toggle checkbox |
| ← | Wizard: No |
| → | Wizard: Yes |
| ↓ | Wizard: Skip |

### 9.2 Screen Reader Support

- All images have alt text: "Face from {filename}"
- Buttons have descriptive labels
- Status changes announced
- Progress announced in wizard

### 9.3 Visual Requirements

- Minimum contrast ratio 4.5:1
- Focus indicators visible
- Touch targets minimum 44x44px
- No color-only indicators

---

## 10. Open Questions

### Resolved
- [x] Batch vs Live mode? → Batch default, user configurable
- [x] Right-click discoverability? → Add menu button + first-time tip
- [x] Confirm action needed? → No, implicit via assignment

### Still Open
1. **Merge people UI**: How to merge two people? Drag one onto another? Select two and click merge?

2. **Split person UI**: If we detect a person might contain multiple people, how to present the split option?

3. **Undo granularity**: Undo individual changes, or undo entire batch?

4. **Persistence scope**: Should "Not a Face" embeddings be album-specific or global?

5. **Performance**: For albums with 1000+ faces, need pagination or virtual scroll?

---

## Appendix A: File Checklist

### New Files to Create
```
Backend:
  sim_bench/api/routers/faces.py
  sim_bench/api/services/face_service.py
  sim_bench/api/schemas/face.py

Frontend:
  app/streamlit/pages/face_management.py
  app/streamlit/components/face_management/__init__.py
  app/streamlit/components/face_management/face_card.py
  app/streamlit/components/face_management/face_grid.py
  app/streamlit/components/face_management/action_menu.py
  app/streamlit/components/face_management/needs_help_wizard.py
  app/streamlit/components/face_management/pending_changes_panel.py
  app/streamlit/components/face_management/person_detail.py
  app/streamlit/components/face_management/face_detail_sheet.py
  app/streamlit/components/face_management/toasts.py

Tests:
  tests/api/test_face_service.py
  tests/api/test_faces_router.py
  tests/integration/test_face_management_flow.py
```

### Files to Modify
```
Backend:
  sim_bench/api/database/models.py  (add FaceOverride)
  sim_bench/api/main.py  (register faces router)

Frontend:
  app/streamlit/components/sidebar.py  (add navigation)
  app/streamlit/api_client.py  (add face methods)
```

---

## Appendix B: API Contract Summary

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| GET | /albums/{id}/runs/{id}/faces | ?status=... | List[FaceInfo] |
| GET | /albums/{id}/runs/{id}/faces/needs-help | ?limit=10 | List[BorderlineFace] |
| GET | /albums/{id}/runs/{id}/faces/{key} | - | FaceInfo |
| GET | /albums/{id}/runs/{id}/faces/{key}/distances | - | List[PersonDistance] |
| POST | /albums/{id}/runs/{id}/faces/batch | BatchChangeRequest | BatchChangeResponse |

---

*Document Version: 1.0*
*Created: 2026-02-14*
*Last Updated: 2026-02-14*
