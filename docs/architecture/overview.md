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

## Standalone Face Clustering Pipeline (`face_cluster/`)

Separate from the main `sim_bench/pipeline/` framework. Designed to run standalone (no FastAPI, no YAML plugin system).

### Design principle: config-driven stages

The pipeline has one public method: `run(config)`. Everything — which stages to execute, where to read input, where to write output, all algorithm parameters — lives in `PipelineConfig`. There is no `mode` flag and no special-case methods.

```python
pipeline = FaceClusteringPipeline()
result = pipeline.run(config)
```

**Stage list** is a field on `PipelineConfig`. Named presets (factory functions or YAML files) cover the standard use cases:

| Preset | Stages | Source input |
|---|---|---|
| `full_run` | discover → embed → quality → crops → cluster → exemplars → merge → export | Raw image folder |
| `recluster` | cluster → exemplars → merge → export | Previous run output dir (reuses crops + embeddings) |
| `remerge` | merge → export | Cluster snapshot dir (reuses cluster assignments) |

### Source loading

Each preset has a corresponding **source loader** that populates `_RunContext` before stage execution begins. The loader is determined by which stages are present in the config, not by a mode flag.

### Two pipelines remain separate

`sim_bench/pipeline/` is a YAML-driven plugin system with `@register_step` registration, topological sort, and FastAPI/WebSocket integration. `face_cluster/pipeline.py` is a simpler fixed-sequence executor for standalone use. They serve different purposes and are intentionally kept separate.

---
*Last updated: 2026-04-17*
