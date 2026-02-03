# Running Instructions - Phase 1

## Prerequisites

Activate the virtual environment and install dependencies:

```bash
cd D:\sim-bench

# Activate venv (Windows)
.venv\Scripts\activate

# Or on Linux/Mac
# source .venv/bin/activate

# Install all dependencies (includes new API/frontend packages)
pip install -r requirements.txt
```

## Run Tests

```bash
# Run all pipeline tests
.venv\Scripts\python -m pytest tests/pipeline/ -v
```

## Quick Test - Pipeline Engine Only

Test the pipeline engine without the API:

```bash
.venv\Scripts\python scripts/test_pipeline.py ./samples/ukbench
```

This runs the walking skeleton pipeline:
- discover_images → score_iqa → filter_quality → extract_scene_embedding → cluster_scenes → select_best

## Running the Full System

### 1. Start the FastAPI Backend

```bash
cd D:\sim-bench
.venv\Scripts\python -m uvicorn sim_bench.api.main:app --reload --port 8000
```

The API will be available at:
- http://localhost:8000
- Docs: http://localhost:8000/docs (Swagger UI)
- Health: http://localhost:8000/health

### 2. Start the NiceGUI Frontend

In a **separate terminal**:
```bash
cd D:\sim-bench
.venv\Scripts\python -m app.nicegui.main
```

The frontend will be available at:
- http://localhost:8080

## API Endpoints

### Albums
- `POST /api/v1/albums/` - Create album
- `GET /api/v1/albums/` - List albums
- `GET /api/v1/albums/{id}` - Get album
- `DELETE /api/v1/albums/{id}` - Delete album

### Steps
- `GET /api/v1/steps/` - List all steps
- `GET /api/v1/steps/{name}` - Get step details
- `GET /api/v1/steps/{name}/schema` - Get step config schema

### Pipeline
- `POST /api/v1/pipeline/run` - Start pipeline
- `GET /api/v1/pipeline/{job_id}` - Get status
- `GET /api/v1/pipeline/{job_id}/result` - Get result

### WebSocket
- `ws://localhost:8000/ws/progress/{job_id}` - Real-time progress

## Example API Usage

```bash
# Create an album
curl -X POST http://localhost:8000/api/v1/albums/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Album", "source_path": "D:/photos/test"}'

# List steps
curl http://localhost:8000/api/v1/steps/

# Run pipeline
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"album_id": "YOUR_ALBUM_ID"}'

# Check status
curl http://localhost:8000/api/v1/pipeline/YOUR_JOB_ID

# Get results
curl http://localhost:8000/api/v1/pipeline/YOUR_JOB_ID/result
```

## Files Created in Phase 1

### Pipeline Engine (`sim_bench/pipeline/`)
- `base.py` - PipelineStep protocol, StepMetadata, BaseStep
- `context.py` - PipelineContext (shared state)
- `registry.py` - StepRegistry (step discovery)
- `builder.py` - PipelineBuilder (dependency resolution)
- `executor.py` - PipelineExecutor (runs pipeline)
- `config.py` - PipelineConfig

### Steps (`sim_bench/pipeline/steps/`)
- `discover_images.py` - Scan directory
- `score_iqa.py` - Technical quality scoring
- `filter_quality.py` - Quality threshold filtering
- `extract_scene_embedding.py` - DINOv2 features
- `cluster_scenes.py` - HDBSCAN clustering
- `select_best.py` - Best image per cluster

### API (`sim_bench/api/`)
- `main.py` - FastAPI app
- `database/models.py` - SQLAlchemy models
- `database/session.py` - Session management
- `schemas/*.py` - Pydantic schemas
- `routers/*.py` - API endpoints
- `services/*.py` - Business logic

### Frontend (`app/nicegui/`)
- `main.py` - NiceGUI app
- `api_client.py` - HTTP/WebSocket client
- `state/app_state.py` - State management

## Next Steps (Phase 2+)

1. Add face analysis steps (detect_faces, score_face_pose, etc.)
2. Add People feature (cluster_people, select_best_per_person)
3. Polish the NiceGUI frontend
4. Add more configuration options
5. Add result export functionality
