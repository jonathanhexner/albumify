# Plan: YAML Config as Single Source of Truth

## Goal

Establish `configs/pipeline.yaml` as the **single source of truth** for pipeline configuration, with user customizations persisted to the database.

## Simplification

**We will delete `sim_bench.db` and start fresh.** No migration needed.

---

## Current State (Problems)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT: 3 SOURCES OF TRUTH                          │
└─────────────────────────────────────────────────────────────────────────────┘

  configs/pipeline.yaml          Frontend                    Backend
  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ default_pipeline │      │ DEFAULT_PIPELINE │      │ DEFAULT_PIPELINE │
  │ - discover_images│      │ - discover_images│      │ - discover_images│
  │ - detect_faces   │      │ - detect_faces   │ ←──  │ - detect_persons │
  │ - score_face_eyes│      │ - score_face_eyes│      │ - insightface_*  │
  │   (OLD MediaPipe)│      │   (OLD MediaPipe)│      │   (NEW InsightFace)
  └──────────────────┘      └────────┬─────────┘      └──────────────────┘
         │                           │                         │
         │ Loaded to DB              │ USED!                   │ Ignored
         │ once on startup           │                         │
         ▼                           ▼                         ▼
  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │   Database       │      │   API Request    │      │   Backend uses   │
  │   ConfigProfile  │      │   steps=[...]    │      │   frontend's list│
  │   (stale copy)   │      │   (hardcoded)    │      │   not its own    │
  └──────────────────┘      └──────────────────┘      └──────────────────┘
```

### Problems

1. **Frontend hardcodes step list** - ignores YAML and DB
2. **Backend hardcodes step list** - only used if frontend sends `steps=None`
3. **YAML config is outdated** - still shows MediaPipe as default
4. **DB config is stale** - only loaded once, never refreshed
5. **No user settings persistence** - UI customizations lost on refresh

---

## Target State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TARGET: YAML → DB → FRONTEND                              │
└─────────────────────────────────────────────────────────────────────────────┘

  configs/pipeline.yaml                    Database
  ┌──────────────────────┐          ┌─────────────────────────────┐
  │ SINGLE SOURCE OF     │          │ ConfigProfile table         │
  │ TRUTH                │          │                             │
  │                      │  Sync    │ ┌─────────────────────────┐ │
  │ default_pipeline:    │ ──────►  │ │ "default" profile       │ │
  │   - discover_images  │          │ │ (from YAML, read-only)  │ │
  │   - detect_persons   │          │ └─────────────────────────┘ │
  │   - insightface_*    │          │                             │
  │   ...                │          │ ┌─────────────────────────┐ │
  │                      │          │ │ "user_abc123" profile   │ │
  │ Step configs:        │          │ │ (user customizations)   │ │
  │   select_best:       │          │ │ inherits from default   │ │
  │     max_per_cluster  │          │ └─────────────────────────┘ │
  └──────────────────────┘          └──────────────┬──────────────┘
                                                   │
                                                   │ API: GET /config
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │ Frontend                    │
                                    │                             │
                                    │ - Fetches config from API   │
                                    │ - No hardcoded steps        │
                                    │ - Saves user prefs to DB    │
                                    └─────────────────────────────┘
```

---

## Data Model

### ConfigProfile Table (Already Exists)

```python
class ConfigProfile(Base):
    id: str              # UUID
    name: str            # "default", "user_abc123", "my_fast_pipeline"
    description: str     # Human-readable description
    config: JSON         # Full config dict (pipeline + step configs)
    is_default: bool     # True for the system default profile
    is_system: bool      # NEW: True = managed by system, False = user-created
    user_id: str         # NEW: Optional user identifier (for user profiles)
    created_at: datetime
    updated_at: datetime
```

### Config Structure

```yaml
# What gets stored in ConfigProfile.config JSON
{
  # Pipeline definition
  "default_pipeline": [
    "discover_images",
    "detect_persons",
    "insightface_detect_faces",
    ...
  ],

  # Alternative pipelines (optional)
  "insightface_pipeline": [...],
  "minimal_pipeline": [...],

  # Step configurations
  "detect_persons": {
    "model_size": "small",
    "confidence_threshold": 0.25
  },
  "select_best": {
    "max_images_per_cluster": 2,
    "min_score_threshold": 0.4
  },
  ...
}
```

---

## Implementation Plan

### Phase 1: Update YAML to Current State

**Files:** `configs/pipeline.yaml`

Update the YAML to reflect the current desired state (InsightFace pipeline):

```yaml
# configs/pipeline.yaml

# The default pipeline - this is THE source of truth
default_pipeline:
  - discover_images
  - score_iqa
  - score_ava
  - detect_persons
  - insightface_detect_faces
  - insightface_score_expression
  - insightface_score_eyes
  - insightface_score_pose
  - filter_quality
  - extract_scene_embedding
  - cluster_scenes
  - extract_face_embeddings
  - cluster_people
  - cluster_by_identity
  - select_best

# Alternative: Minimal pipeline (no face detection)
minimal_pipeline:
  - discover_images
  - score_iqa
  - filter_quality
  - extract_scene_embedding
  - cluster_scenes
  - select_best

# Alternative: Legacy MediaPipe pipeline (if needed)
mediapipe_pipeline:
  - discover_images
  - detect_faces
  - score_face_pose
  - score_face_eyes
  - score_face_smile
  ...

# Step configurations (unchanged)
detect_persons:
  model_size: small
  confidence_threshold: 0.25
  ...
```

---

### Phase 2: Improve Config Sync (Backend)

**Files:**
- `sim_bench/api/services/config_service.py`
- `sim_bench/api/database/models.py`

#### 2.1 Add System Profile Flag

```python
# models.py
class ConfigProfile(Base):
    ...
    is_system = Column(Boolean, default=False)  # NEW: system-managed profile
    user_id = Column(String, nullable=True)      # NEW: for user profiles
    parent_profile_id = Column(String, nullable=True)  # NEW: inheritance
```

#### 2.2 Sync YAML to DB on Every Startup

```python
# config_service.py

def sync_default_profile(self) -> ConfigProfile:
    """
    Sync the default profile with pipeline.yaml.

    Called on every API startup to ensure DB matches YAML.
    User profiles are NOT affected - they inherit from default.
    """
    yaml_config = load_yaml_config(DEFAULT_CONFIG_PATH)

    profile = self._session.query(ConfigProfile).filter(
        ConfigProfile.name == "default",
        ConfigProfile.is_system == True
    ).first()

    if profile is None:
        # First run - create default profile
        profile = ConfigProfile(
            id=str(uuid.uuid4()),
            name="default",
            description="System default (from pipeline.yaml)",
            config=yaml_config,
            is_default=True,
            is_system=True,
        )
        self._session.add(profile)
    else:
        # Update existing - YAML takes precedence
        profile.config = yaml_config
        profile.updated_at = datetime.utcnow()

    self._session.commit()
    return profile
```

#### 2.3 Update API Startup

```python
# main.py

@app.on_event("startup")
def startup():
    ...
    config_service = ConfigService(session)
    config_service.sync_default_profile()  # Always sync from YAML
    ...
```

---

### Phase 3: Remove Hardcoded Pipelines

**Files:**
- `sim_bench/api/services/pipeline_service.py`
- `app/streamlit/components/pipeline_runner.py`

#### 3.1 Backend: Remove Hardcoded DEFAULT_PIPELINE

```python
# pipeline_service.py

# REMOVE THIS:
# DEFAULT_PIPELINE = [
#     "discover_images",
#     ...
# ]

def start_pipeline(
    self,
    album_id: str,
    steps: list[str] = None,
    pipeline_name: str = "default_pipeline",  # NEW: which pipeline to use
    step_configs: dict[str, dict] = None,
    profile: str = "default",
    ...
) -> str:
    """Start a pipeline run."""

    if steps is None:
        # Load from config profile
        config_service = ConfigService(self._session)
        config = config_service.get_merged_config(profile_name=profile)
        steps = config.get(pipeline_name, config.get("default_pipeline", []))

    ...
```

#### 3.2 Frontend: Fetch Config from API

```python
# pipeline_runner.py

# REMOVE THIS:
# DEFAULT_PIPELINE = [
#     "discover_images",
#     ...
# ]

def get_pipeline_options() -> dict:
    """Fetch available pipelines from API."""
    client = get_client()
    config = client.get_default_config()

    return {
        "default": config.get("default_pipeline", []),
        "minimal": config.get("minimal_pipeline", []),
        "mediapipe": config.get("mediapipe_pipeline", []),
    }

def render_pipeline_runner(album: Album) -> Optional[str]:
    """Render pipeline configuration and run button."""

    # Fetch pipelines from API (cached)
    pipelines = get_pipeline_options()

    pipeline_type = st.radio(
        "Pipeline Type",
        options=list(pipelines.keys()),
        format_func=lambda x: {
            "default": "Full (InsightFace)",
            "minimal": "Minimal (no faces)",
            "mediapipe": "Legacy (MediaPipe)",
        }.get(x, x),
        horizontal=True,
    )

    steps = pipelines[pipeline_type]
    ...
```

---

### Phase 4: User Settings Persistence

**Files:**
- `sim_bench/api/routers/config.py`
- `sim_bench/api/services/config_service.py`
- `app/streamlit/components/pipeline_runner.py`
- `app/streamlit/session.py`

#### 4.1 User Profile API Endpoints

```python
# config.py (router)

@router.get("/user/{user_id}")
def get_user_config(user_id: str, session: Session = Depends(get_session)):
    """Get user's saved config, or default if none exists."""
    service = ConfigService(session)
    profile = service.get_user_profile(user_id)

    if profile is None:
        profile = service.get_default_profile()

    return {"profile": profile.name, "config": profile.config}


@router.post("/user/{user_id}")
def save_user_config(
    user_id: str,
    request: SaveConfigRequest,
    session: Session = Depends(get_session)
):
    """Save user's config preferences."""
    service = ConfigService(session)
    profile = service.save_user_profile(
        user_id=user_id,
        config_overrides=request.config,
        selected_pipeline=request.pipeline,
    )
    return {"profile": profile.name, "saved": True}
```

#### 4.2 Config Service: User Profiles

```python
# config_service.py

def get_user_profile(self, user_id: str) -> Optional[ConfigProfile]:
    """Get user's saved profile."""
    return self._session.query(ConfigProfile).filter(
        ConfigProfile.user_id == user_id
    ).first()


def save_user_profile(
    self,
    user_id: str,
    config_overrides: dict,
    selected_pipeline: str = "default_pipeline",
) -> ConfigProfile:
    """
    Save user's config preferences.

    Creates a new profile or updates existing one.
    Stores only the OVERRIDES, not the full config.
    """
    profile = self.get_user_profile(user_id)

    if profile is None:
        profile = ConfigProfile(
            id=str(uuid.uuid4()),
            name=f"user_{user_id}",
            description=f"User preferences",
            user_id=user_id,
            parent_profile_id=self.get_default_profile().id,
            is_system=False,
            config={
                "_selected_pipeline": selected_pipeline,
                "_overrides": config_overrides,
            },
        )
        self._session.add(profile)
    else:
        profile.config = {
            "_selected_pipeline": selected_pipeline,
            "_overrides": config_overrides,
        }
        profile.updated_at = datetime.utcnow()

    self._session.commit()
    return profile


def get_merged_user_config(self, user_id: str) -> dict:
    """
    Get user's effective config (default + user overrides).
    """
    default = self.get_default_profile().config
    user_profile = self.get_user_profile(user_id)

    if user_profile is None:
        return default

    overrides = user_profile.config.get("_overrides", {})
    return self._deep_merge(default, overrides)
```

#### 4.3 Frontend: Load and Save User Settings

```python
# pipeline_runner.py

def get_user_id() -> str:
    """Get or create a persistent user ID."""
    if "user_id" not in st.session_state:
        # Generate a unique ID for this browser session
        # In production, this could come from authentication
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id


def load_user_settings() -> dict:
    """Load user's saved settings from API."""
    client = get_client()
    user_id = get_user_id()

    try:
        response = client.get_user_config(user_id)
        return response.get("config", {})
    except:
        return {}


def save_user_settings(pipeline: str, config: dict) -> None:
    """Save user's settings to API."""
    client = get_client()
    user_id = get_user_id()

    client.save_user_config(user_id, pipeline=pipeline, config=config)
    st.success("Settings saved!")


def render_pipeline_runner(album: Album) -> Optional[str]:
    """Render pipeline configuration with persistent user settings."""

    # Load user's saved settings
    user_settings = load_user_settings()

    # Get saved pipeline selection or default
    saved_pipeline = user_settings.get("_selected_pipeline", "default_pipeline")
    saved_overrides = user_settings.get("_overrides", {})

    # Pipeline selection (with saved default)
    pipelines = get_pipeline_options()
    pipeline_type = st.radio(
        "Pipeline Type",
        options=list(pipelines.keys()),
        index=list(pipelines.keys()).index(saved_pipeline) if saved_pipeline in pipelines else 0,
        ...
    )

    # Config sliders (with saved defaults)
    saved_select_best = saved_overrides.get("select_best", {})
    max_per_cluster = st.number_input(
        "Max Images per Cluster",
        value=saved_select_best.get("max_images_per_cluster", 2),
        ...
    )

    ...

    # Save button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Pipeline", ...):
            return _start_pipeline(...)

    with col2:
        if st.button("Save Settings", ...):
            save_user_settings(pipeline_type, config)
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `configs/pipeline.yaml` | Update `default_pipeline` to InsightFace steps |
| `sim_bench/api/database/models.py` | Add `is_system`, `user_id`, `parent_profile_id` columns |
| `sim_bench/api/services/config_service.py` | Add `sync_default_profile()`, `get_user_profile()`, `save_user_profile()` |
| `sim_bench/api/routers/config.py` | Add `/user/{user_id}` GET/POST endpoints |
| `sim_bench/api/main.py` | Call `sync_default_profile()` on startup |
| `sim_bench/api/services/pipeline_service.py` | Remove `DEFAULT_PIPELINE`, load from config service |
| `app/streamlit/components/pipeline_runner.py` | Remove `DEFAULT_PIPELINE`, fetch from API, add save/load |
| `app/streamlit/api_client.py` | Add `get_user_config()`, `save_user_config()` methods |

---

## Data Flow After Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEW DATA FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────┘

1. API STARTUP
   ┌──────────────────┐      ┌──────────────────┐
   │ pipeline.yaml    │ ───► │ Database         │
   │ (source of truth)│ sync │ default profile  │
   └──────────────────┘      └──────────────────┘

2. USER OPENS APP
   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
   │ Frontend         │ ───► │ API              │ ───► │ Database         │
   │ GET /config/user │      │ get_user_config  │      │ user profile     │
   │                  │ ◄─── │                  │ ◄─── │ (or default)     │
   │ Populate UI with │      │ merge default +  │      │                  │
   │ saved settings   │      │ user overrides   │      │                  │
   └──────────────────┘      └──────────────────┘      └──────────────────┘

3. USER CHANGES SETTINGS
   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
   │ Frontend         │ ───► │ API              │ ───► │ Database         │
   │ POST /config/user│      │ save_user_profile│      │ user profile     │
   │ {overrides}      │      │                  │      │ {_overrides: {}} │
   └──────────────────┘      └──────────────────┘      └──────────────────┘

4. USER RUNS PIPELINE
   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
   │ Frontend         │ ───► │ API              │ ───► │ Pipeline         │
   │ POST /pipeline   │      │ get_merged_config│      │ Executor         │
   │ {profile: user}  │      │ default+overrides│      │ runs steps       │
   └──────────────────┘      └──────────────────┘      └──────────────────┘

5. ADMIN UPDATES YAML
   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
   │ Edit pipeline.   │ ───► │ Restart API      │ ───► │ Database         │
   │ yaml             │      │ sync_default_    │      │ default profile  │
   │                  │      │ profile()        │      │ updated          │
   └──────────────────┘      └──────────────────┘      └──────────────────┘
                                                              │
                                    User profiles automatically get new defaults
                                    (only overrides are stored, base is merged)
```

---

## Testing Checklist

- [ ] Update `pipeline.yaml` to InsightFace steps
- [ ] Delete `sim_bench.db` (fresh start)
- [ ] Start API - verify default profile is created from YAML
- [ ] Open frontend - verify InsightFace steps are shown
- [ ] Change settings in UI - verify they apply
- [ ] Click "Save Settings" - verify saved to DB
- [ ] Refresh page - verify settings are restored
- [ ] Update `pipeline.yaml` again - restart API
- [ ] Verify user settings are preserved but get new defaults
- [ ] Run pipeline - verify correct steps execute

---

## Rollback Plan

If issues occur:

1. Restore old `pipeline.yaml` from git
2. Delete `sim_bench.db`
3. Restart API (will recreate from YAML)

---

## Future Enhancements

1. **Named Presets**: Users can save multiple named presets ("Wedding", "Landscape", etc.)
2. **Sharing**: Share preset configs with other users
3. **Version History**: Track config changes over time
4. **Validation**: Validate step dependencies before saving
5. **UI Config Editor**: Visual pipeline builder in the frontend
