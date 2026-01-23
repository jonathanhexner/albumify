# Album Organization UI

Streamlit UI for photo album organization workflow.

## Architecture

```
app/album/
├── main.py              # Entry point
├── session.py           # Session state management
└── components/          # UI components (pure rendering)
    ├── config_panel.py  # Configuration UI
    ├── workflow_runner.py # Workflow execution
    ├── gallery.py       # Image gallery
    ├── metrics.py       # Metrics display
    └── results.py       # Results orchestrator
```

## Running the App

```bash
streamlit run app/album/main.py
```

## Usage

### Session Management

```python
from app.album.session import AlbumSession

AlbumSession.initialize()  # Initialize state
service = AlbumSession.get_service(config)  # Get service
result = AlbumSession.get_result()  # Get result
AlbumSession.set_result(result)  # Store result
```

### Components

```python
from app.album.components import (
    render_config_panel,
    render_workflow_form,
    render_workflow_runner,
    render_results,
    render_gallery,
    render_metrics
)

# All components are pure rendering functions
# Business logic goes through AlbumSession -> AlbumService
```

## Design Principles

1. **Thin UI Layer**: Components only render, no business logic
2. **Service Delegation**: All logic goes through AlbumService
3. **Session Isolation**: State managed centrally in AlbumSession

## Workflow

1. User configures settings → `render_config_panel()`
2. User enters paths → `render_workflow_form()`
3. User runs workflow → `render_workflow_runner()`
4. Results displayed → `render_results()`

## Future: Swapping UI

The thin UI layer means you can replace Streamlit with:

- **NiceGUI**: Similar Python API
- **FastAPI + React**: REST endpoints calling AlbumService
- **Desktop (Tauri)**: Same service layer
