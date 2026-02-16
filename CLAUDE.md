# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General
- Always aim to learn from your failures. For any significant bug, please be sure to prodce a failure reports in a few lines and summarize a recommendation to docs\LEARNINGS.md.
Log learnings in up to 3-5 lines with date according to order.
- For every significant implemenation and plannding step always review learnings, and ensure we're not reapeating errors from the past.
- Unless otherwise instructed, produce plans for approval prior to implementing code changes.
- Consider how to test newly added features. Use tests/ folder.
- In any rejection or replan, consider what needs to be changed in CLAUDE.md for future improvement, and propose making changes.
- Break down plans into small individual work items. Create a TODO list from them.
- Always update the TODOs plan after each step.

## Coding
- `__init__.py` files should be kept empty.
- Never use local or relative imports. Always use full imports (e.g., `from sim_bench.pipeline.base import BaseStep`).
- Avoid excessive Try/Except. Keep it only for extreme cases where output is unpredictable.
- Avoid excessive If statements. Prefer using strategy or factory pattern.
- Avoid usage of prints, prefer usage of proper logging. Make sure we support logging injection for centralized logging.


## Verification
- Avoid half baked code. Always verify you understand what you're being asked and that the code comlies with the request. 
- When in doubt always ask questions to verify you understand the request.


## ⚠️ IMPORTANT: Change Tracking

**After EVERY code change you make**, append an entry to `CHANGES_LOG.md` with:
- Date and time (ISO 8601 format)
- Files modified
- Brief description of what was changed
- Why it was changed

Example entry:
```markdown
### 2026-02-03 14:30:00
**Files**: `app/streamlit/components/gallery.py`
**Change**: Added "Final Score" column to cluster debug table
**Reason**: User requested visibility of composite selection score for debugging
```

If `CHANGES_LOG.md` doesn't exist, create it with header "# Change Log".

---

## Project Overview

sim-bench is a Python 3.10+ framework for image similarity benchmarking and image quality assessment:
- **Image similarity/retrieval**: Classical (HSV histograms) and deep learning (ResNet50, DINOv2, OpenCLIP)
- **Quality assessment**: Siamese networks and AVA aesthetic models
- **Face recognition**: ArcFace embeddings, pose estimation, expression scoring
- **Album organization**: Multi-step pipeline with Streamlit frontend + FastAPI backend

## Common Commands

```bash
# Backend API (FastAPI) - Terminal 1
python -m uvicorn sim_bench.api.main:app --reload --port 8000

# Frontend (Streamlit) - Terminal 2
streamlit run app/streamlit/main.py

# Run benchmarks
python -m sim_bench.cli --methods chi_square,deep,dinov2 --datasets ukbench,holidays
python -m sim_bench.cli --quick --methods chi_square --datasets ukbench  # Fast test

# Train models
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
python -m sim_bench.training.train_ava_resnet --config configs/ava/resnet50_cpu.yaml
python -m sim_bench.face_recognition.train --config configs/face/resnet50_arcface.yaml

# Run tests (use venv python on Windows)
python -m pytest tests/                                              # All tests
python -m pytest tests/test_model_hub.py                             # Single file
python -m pytest tests/test_model_hub.py -k "test_hub_initialization" # Single test
python -m pytest tests/pipeline/test_face_pipeline_e2e.py -v -s      # Face pipeline E2E

# Verify syntax
python -m py_compile <file>
```

## Architecture

### Frontend + Backend Stack
- **Frontend**: Streamlit (`app/streamlit/`) - Web UI for album viewing, pipeline execution, results
- **Backend**: FastAPI (`sim_bench/api/`) - REST API with SQLAlchemy ORM, SQLite database (`sim_bench.db`)
- **Communication**: HTTP REST + WebSocket (`/ws`) for real-time progress updates
- **Routers**: `albums`, `pipeline`, `steps`, `websocket`, `people`, `results`, `config` (see `sim_bench/api/routers/`)
- **Services**: Business logic layer in `sim_bench/api/services/` - `PipelineService`, `AlbumService`, `PeopleService`, `ConfigService`, `ResultService`

### Pipeline Engine (`sim_bench/pipeline/`)
The pipeline processes photos through configurable steps. Two pipelines are available:
- **default_pipeline**: Uses MediaPipe for face detection
- **insightface_pipeline**: Uses YOLOv8-Pose + InsightFace (SCRFD) for better accuracy

**Key Components:**
- `base.py`: `BaseStep` class with template method for automatic caching
- `context.py`: `PipelineContext` - shared mutable state passed through all steps
- `registry.py`: Global step registry with `@register_step` decorator
- `steps/`: Individual step implementations (24+ steps)

**Adding a New Pipeline Step:**
```python
# sim_bench/pipeline/steps/your_step.py
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.registry import register_step

@register_step
class YourStep(BaseStep):
    _metadata = StepMetadata(
        name="your_step",
        display_name="Your Step",
        description="What this step does",
        category="analysis",  # analysis, filtering, embedding, clustering, selection
        requires={"image_paths"},  # Context keys this step reads
        produces={"your_output"},  # Context keys this step writes
        depends_on=["discover_images"],  # Steps that must run first
    )

    def process(self, context, config):
        # Read from context, process, write back to context
        for path in context.image_paths:
            # ... your logic
            pass
```

Then import in `sim_bench/pipeline/steps/all_steps.py` and add to `configs/pipeline.yaml`.

### Pipeline Data Flow
1. User requests pipeline via API → `PipelineService` creates `PipelineRun` record
2. `PipelineExecutor` resolves dependencies via `PipelineBuilder` (topological sort)
3. Each step: validates context requirements → executes `process(context, config)` → writes results back to context
4. Progress updates sent via WebSocket in real-time
5. Final results stored in `PipelineResult` and `people` tables

### Face Recognition Pipeline
Key steps for face clustering:
1. `insightface_detect_faces` - Detect faces, store bbox in `context.insightface_faces`
2. `extract_face_embeddings` - Crop faces using bbox, extract 512-dim embeddings via InsightFace
3. `cluster_people` - HDBSCAN clustering on normalized embeddings → `context.people_clusters`

**Important**: Embeddings are cached by `(image_path, face_index)`. If embeddings become corrupted (zero vectors), clear the cache and re-run.

### Caching System
Steps can cache computed features to SQLite (`UniversalCache` table) with mtime tracking:
- Override `_get_cache_config()`, `_serialize_for_cache()`, `_deserialize_from_cache()` in your step
- Cache automatically invalidates when source file is modified
- Use `UniversalCacheHandler` for direct cache access

### Factory Pattern (Benchmarking Components)
- **Methods**: `sim_bench.feature_extraction.base.load_method(name, config)` - maps to classes via registry
- **Datasets**: `sim_bench.datasets.base.load_dataset(name, config)` - ukbench, holidays, phototriage, flatdir
- **Metrics**: `sim_bench.metrics.factory.MetricFactory` - auto-discovers BaseMetric subclasses
- **Distances**: `sim_bench.distances.base.create_distance_strategy(config)` - cosine, euclidean, chi_square
- **Clustering**: `sim_bench.clustering.base.load_clustering_method(config)` - HDBSCAN, KMeans, hierarchical
- **Face Embeddings**: `sim_bench.pipeline.face_embedding.factory.FaceEmbeddingExtractorFactory` - CustomArcFace or InsightFaceNative

### Configuration
All behavior is YAML-configured in `configs/`:
- `pipeline.yaml` - Pipeline steps and their parameters (most frequently edited)
- `global_config.yaml` - Model checkpoints and global settings
- `dataset.*.yaml` - Dataset paths
- `methods/*.yaml` - Feature extraction configs
- `run.yaml` - Metrics, sampling, output settings

### Key Entry Points
| Entry Point | Purpose |
|-------------|---------|
| `sim_bench/cli.py` | CLI for benchmarking |
| `sim_bench/api/main.py` | FastAPI server |
| `app/streamlit/main.py` | Streamlit frontend |
| `sim_bench/model_hub/hub.py` | Unified model interface with lazy loading |

### Model Weights
Trained models are stored in `models/album_app/`:
- `ava_resnet50.pt` - AVA aesthetic model
- `siamese_comparison_model.pt` - Siamese comparison model
- `arcface_resnet50.pt` - ArcFace face recognition

Referenced in `configs/pipeline.yaml` and `configs/global_config.yaml`.

## Debugging Tips

### Common Issues
1. **Images not rotating**: Use `ImageOps.exif_transpose()` before display
2. **Model not loading**: Check nested config structure (e.g., `config["siamese"]["checkpoint_path"]`)
3. **Clustering parameters ignored**: Verify all params passed to HDBSCAN constructor
4. **Mixed type errors in DataFrames**: Convert all column values to consistent types
5. **Cache not working**: Check `UniversalCache` table exists, verify mtime tracking
6. **All faces clustering to one person**: Check for zero-vector embeddings in cache (stale data from previous bugs). Clear with: `DELETE FROM universal_cache WHERE feature_type = 'face_embedding'`

### Database
SQLite database location: `~/.sim_bench/sim_bench.db` (NOT in project root)
Tables: `albums`, `pipeline_runs`, `pipeline_results`, `universal_cache`, `people`, `config_profiles`

### Useful Debug Scripts
- `scripts/check_zero_vectors.py` - Analyze face embeddings for zero vectors
- `scripts/clear_face_embedding_cache.py` - Clear stale face embedding cache

---

## Change Log Location

**Always maintain**: `CHANGES_LOG.md` at project root
