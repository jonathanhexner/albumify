# System Architecture

This document describes the current system architecture of sim-bench.

## High-Level Overview

```
┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │
│   Frontend      │◀────│    Backend      │
│ (app/streamlit/)│ WS  │ (sim_bench/api/)│
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │  Pipeline │ │  Services │ │  Database │
            │  Engine   │ │   Layer   │ │  (SQLite) │
            └───────────┘ └───────────┘ └───────────┘
```

## Components

### Frontend (Streamlit)
- Location: `app/streamlit/`
- Purpose: Web UI for album viewing, pipeline execution, results visualization

### Backend (FastAPI)
- Location: `sim_bench/api/`
- Routers: albums, pipeline, steps, websocket, people, results, config
- Services: PipelineService, AlbumService, PeopleService, ConfigService, ResultService
- Database: SQLite via SQLAlchemy ORM

### Pipeline Engine
- Location: `sim_bench/pipeline/`
- Components: BaseStep, PipelineContext, PipelineExecutor, PipelineBuilder
- Step registration via `@register_step` decorator
- Dependency resolution via topological sort

### Benchmarking System
- CLI: `sim_bench/cli.py`
- Factory patterns for: Methods, Datasets, Metrics, Distances, Clustering
- See README.md for available methods and datasets

## Data Flow

1. User requests pipeline via API
2. PipelineService creates PipelineRun record
3. PipelineExecutor resolves step dependencies
4. Each step processes context and writes results
5. Progress updates via WebSocket
6. Final results stored in database

## Key Design Decisions

- **Caching**: UniversalCache with mtime tracking for invalidation
- **Image handling**: Global image cache with EXIF normalization
- **Configuration**: YAML-driven behavior (configs/)
- **Extensibility**: Factory pattern for all major components

---
*Last updated: 2026-02-18*
