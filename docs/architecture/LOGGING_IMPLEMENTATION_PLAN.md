# Logging Implementation Plan

## Summary

Add local logging with:
- Timestamped folders per app run (e.g., `logs/2024-01-30_10-30-00/api.log`)
- Optional logger injection in services for testing and configuration flexibility

---

## Files Impacted

| File | Action | Description |
|------|--------|-------------|
| `sim_bench/api/logging.py` | **CREATE** | New centralized logging setup module |
| `sim_bench/api/main.py` | **MODIFY** | Add logging initialization on startup |
| `sim_bench/api/services/album_service.py` | **MODIFY** | Add optional logger injection |
| `sim_bench/api/services/pipeline_service.py` | **MODIFY** | Add optional logger injection |

---

## File 1: `sim_bench/api/logging.py` (NEW)

This is a new file that provides centralized logging configuration.

```python
"""Logging configuration for the API."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Module-level reference to current log directory
_current_log_dir: Optional[Path] = None


def setup_logging(
    base_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True
) -> Path:
    """
    Configure logging with timestamped folder.

    Creates a new folder for each app run:
        logs/2024-01-30_10-30-00/api.log

    Args:
        base_dir: Base directory for logs (default: "logs")
        level: Logging level (default: INFO)
        console: Whether to also log to console (default: True)

    Returns:
        Path to the log directory for this run
    """
    global _current_log_dir

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(base_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    _current_log_dir = log_dir

    log_file = log_dir / "api.log"

    # Configure root logger
    handlers = [logging.FileHandler(log_file, encoding='utf-8')]
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # Override any existing config
    )

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return log_dir


def get_log_dir() -> Optional[Path]:
    """Get the current log directory for this run."""
    return _current_log_dir


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.

    Convenience wrapper for logging.getLogger().

    Usage:
        from sim_bench.api.logging import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
```

---

## File 2: `sim_bench/api/main.py` (MODIFY)

### Current Code

```python
"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sim_bench.api.database.session import init_db
from sim_bench.api.routers import albums, pipeline, steps, websocket

app = FastAPI(
    title="Album Organization API",
    description="API for organizing photo albums using ML-powered pipelines",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(albums.router)
app.include_router(pipeline.router)
app.include_router(steps.router)
app.include_router(websocket.router)


@app.on_event("startup")
def startup():
    """Initialize database on startup."""
    init_db()

    import sim_bench.pipeline.steps.all_steps


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### New Code (changes highlighted with comments)

```python
"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sim_bench.api.database.session import init_db
from sim_bench.api.routers import albums, pipeline, steps, websocket
from sim_bench.api.logging import setup_logging, get_logger  # NEW

# Setup logging - creates timestamped folder                   # NEW
log_dir = setup_logging()                                      # NEW
logger = get_logger(__name__)                                  # NEW

app = FastAPI(
    title="Album Organization API",
    description="API for organizing photo albums using ML-powered pipelines",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(albums.router)
app.include_router(pipeline.router)
app.include_router(steps.router)
app.include_router(websocket.router)


@app.on_event("startup")
def startup():
    """Initialize database on startup."""
    logger.info(f"Starting Album Organization API")            # NEW
    logger.info(f"Logs directory: {log_dir}")                  # NEW
    init_db()
    logger.info("Database initialized")                        # NEW

    import sim_bench.pipeline.steps.all_steps
    logger.info("Pipeline steps registered")                   # NEW


@app.on_event("shutdown")                                      # NEW
def shutdown():                                                # NEW
    """Log shutdown."""                                        # NEW
    logger.info("Shutting down Album Organization API")        # NEW


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## File 3: `sim_bench/api/services/album_service.py` (MODIFY)

### Current Code

```python
"""Album service - business logic for album management."""

import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from sim_bench.api.database.models import Album


class AlbumService:
    """Service for managing albums."""

    def __init__(self, session: Session):
        self._session = session

    def create(self, name: str, source_path: str) -> Album:
        """Create a new album."""
        path = Path(source_path)
        if not path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        if not path.is_dir():
            raise ValueError(f"Source path is not a directory: {source_path}")

        image_extensions = {".jpg", ".jpeg", ".png", ".heic", ".raw"}
        image_count = sum(
            1 for f in path.rglob("*")
            if f.suffix.lower() in image_extensions
        )

        album = Album(
            id=str(uuid.uuid4()),
            name=name,
            source_path=str(path.resolve()),
            image_count=image_count
        )

        self._session.add(album)
        self._session.commit()
        self._session.refresh(album)

        return album

    def get(self, album_id: str) -> Optional[Album]:
        """Get an album by ID."""
        return self._session.query(Album).filter(Album.id == album_id).first()

    def list_all(self) -> list[Album]:
        """List all albums."""
        return self._session.query(Album).order_by(Album.created_at.desc()).all()

    def delete(self, album_id: str) -> bool:
        """Delete an album by ID."""
        album = self.get(album_id)
        if album is None:
            return False

        self._session.delete(album)
        self._session.commit()
        return True
```

### New Code (changes highlighted with comments)

```python
"""Album service - business logic for album management."""

import logging                                                  # NEW
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from sim_bench.api.database.models import Album


class AlbumService:
    """Service for managing albums."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None                 # NEW - optional injection
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)    # NEW - default if not provided

    def create(self, name: str, source_path: str) -> Album:
        """Create a new album."""
        self._logger.info(f"Creating album '{name}' from {source_path}")  # NEW

        path = Path(source_path)
        if not path.exists():
            self._logger.error(f"Source path does not exist: {source_path}")  # NEW
            raise ValueError(f"Source path does not exist: {source_path}")
        if not path.is_dir():
            self._logger.error(f"Source path is not a directory: {source_path}")  # NEW
            raise ValueError(f"Source path is not a directory: {source_path}")

        image_extensions = {".jpg", ".jpeg", ".png", ".heic", ".raw"}
        image_count = sum(
            1 for f in path.rglob("*")
            if f.suffix.lower() in image_extensions
        )

        album = Album(
            id=str(uuid.uuid4()),
            name=name,
            source_path=str(path.resolve()),
            image_count=image_count
        )

        self._session.add(album)
        self._session.commit()
        self._session.refresh(album)

        self._logger.info(f"Created album {album.id} with {image_count} images")  # NEW

        return album

    def get(self, album_id: str) -> Optional[Album]:
        """Get an album by ID."""
        return self._session.query(Album).filter(Album.id == album_id).first()

    def list_all(self) -> list[Album]:
        """List all albums."""
        return self._session.query(Album).order_by(Album.created_at.desc()).all()

    def delete(self, album_id: str) -> bool:
        """Delete an album by ID."""
        album = self.get(album_id)
        if album is None:
            self._logger.warning(f"Attempted to delete non-existent album: {album_id}")  # NEW
            return False

        self._session.delete(album)
        self._session.commit()
        self._logger.info(f"Deleted album {album_id}")          # NEW
        return True
```

---

## File 4: `sim_bench/api/services/pipeline_service.py` (MODIFY)

Same pattern as AlbumService:

1. Add `import logging` at top
2. Add optional `logger` parameter to `__init__`
3. Store as `self._logger = logger or logging.getLogger(__name__)`
4. Add logging calls at key points:
   - `start_pipeline()`: Log when pipeline starts
   - `execute_pipeline()`: Log when execution begins, completes, or fails
   - `subscribe()`/`unsubscribe()`: Log WebSocket subscriptions

---

## Result: Log Directory Structure

After running the app multiple times:

```
logs/
    2024-01-30_10-30-00/
        api.log              # First run
    2024-01-30_14-22-15/
        api.log              # Second run
    2024-01-31_09-00-00/
        api.log              # Third run
```

Each `api.log` contains entries like:

```
2024-01-30 10:30:00 - sim_bench.api.main - INFO - Starting Album Organization API
2024-01-30 10:30:00 - sim_bench.api.main - INFO - Logs directory: logs/2024-01-30_10-30-00
2024-01-30 10:30:00 - sim_bench.api.main - INFO - Database initialized
2024-01-30 10:30:00 - sim_bench.api.main - INFO - Pipeline steps registered
2024-01-30 10:30:05 - sim_bench.api.services.album_service - INFO - Creating album 'Vacation' from D:/photos/vacation
2024-01-30 10:30:06 - sim_bench.api.services.album_service - INFO - Created album abc-123 with 150 images
```

---

## How Logger Injection Works

### Normal Usage (no injection needed)

```python
# In a router - service uses default logger
service = AlbumService(session)
album = service.create("Vacation", "/path/to/photos")
# Logs go to: sim_bench.api.services.album_service
```

### Testing (inject mock logger)

```python
from unittest.mock import MagicMock

def test_create_album():
    mock_logger = MagicMock()
    service = AlbumService(session, logger=mock_logger)

    album = service.create("Test", "/tmp/photos")

    # Verify logging was called
    mock_logger.info.assert_called()
```

### Custom Configuration (inject custom logger)

```python
# Create a logger with different settings
debug_logger = logging.getLogger("album.debug")
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler("logs/album_debug.log")
debug_logger.addHandler(debug_handler)

# Inject it into the service
service = AlbumService(session, logger=debug_logger)
# Now this service logs to a separate file at DEBUG level
```

---

## Verification Steps

1. **Start the API**
   ```bash
   .venv\Scripts\python -m uvicorn sim_bench.api.main:app --reload --port 8000
   ```

2. **Check log folder created**
   ```
   logs/2024-01-30_XX-XX-XX/api.log
   ```

3. **Check startup logs**
   ```
   Starting Album Organization API
   Logs directory: logs/2024-01-30_XX-XX-XX
   Database initialized
   Pipeline steps registered
   ```

4. **Make an API call**
   ```bash
   curl -X POST http://localhost:8000/api/v1/albums/ \
     -H "Content-Type: application/json" \
     -d '{"name": "Test", "source_path": "D:/photos/test"}'
   ```

5. **Check operation logged**
   ```
   Creating album 'Test' from D:/photos/test
   Created album xyz-789 with 42 images
   ```

6. **Restart the app, verify new folder created**
