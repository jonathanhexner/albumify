# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

sim-bench is a Python framework for image similarity benchmarking and image quality assessment. Key capabilities:
- Image similarity/retrieval using classical (HSV histograms) or deep learning methods (ResNet50, DINOv2, OpenCLIP)
- Clustering algorithms for grouping similar images
- Quality assessment via Siamese networks and AVA aesthetic models
- Face recognition with ArcFace embeddings
- **Album organization pipeline with Streamlit frontend + FastAPI backend**

## Common Commands

### Run image similarity benchmarks
```bash
python -m sim_bench.cli --methods chi_square,deep,dinov2 --datasets ukbench,holidays
python -m sim_bench.cli --quick --methods chi_square --datasets ukbench  # Fast testing
python -m sim_bench.cli  # All methods, all datasets
```

### Train models
```bash
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
python -m sim_bench.training.train_ava_resnet --config configs/ava/resnet50_cpu.yaml
python -m sim_bench.face_recognition.train --config configs/face/resnet50_arcface.yaml
```

### Launch apps
```bash
# Backend API (FastAPI) - Terminal 1
python -m uvicorn sim_bench.api.main:app --reload --port 8000

# Frontend (Streamlit) - Terminal 2
streamlit run app/streamlit/main.py

# Legacy apps (if needed)
streamlit run app/photo_organization/main.py  # AI agent for photo organization
```

### Run tests
```bash
python -m pytest tests/                        # All tests
python -m pytest tests/test_model_hub.py       # Single test file
python -m pytest tests/test_model_hub.py -k "test_hub_initialization"  # Single test
```

### Install dependencies
```bash
pip install -r requirements.txt          # Full install
pip install -r requirements-minimal.txt  # Minimal (no visualization)
pip install -r requirements-dev.txt      # Development (includes testing tools)
```

## Architecture

### Current Stack (Album Organization)
- **Frontend**: Streamlit (`app/streamlit/`) - Web UI for album viewing, pipeline execution, results
- **Backend**: FastAPI (`sim_bench/api/`) - REST API with SQLAlchemy ORM, SQLite database
- **Pipeline**: 18-step processing pipeline (`sim_bench/pipeline/steps/`) - Image analysis, clustering, selection
- **Communication**: HTTP REST + WebSocket for real-time progress updates

### Recent Updates (Feb 2026)
- ✅ Fixed Siamese model config loading (nested config structure)
- ✅ Fixed HDBSCAN clustering parameters (all params now passed through)
- ✅ Fixed image EXIF rotation (ImageOps.exif_transpose)
- ✅ Added per-image score display (IQA, AVA, sharpness, face metrics)
- ✅ Added cluster debug view with image thumbnails and all scores
- ✅ Added "Final Score" column showing composite selection score

### Factory Pattern
All major components use factories for extensibility - no CLI changes needed when adding new methods, datasets, or metrics:

- **Methods**: `sim_bench.feature_extraction.base.load_method(name, config)` - registry in same file maps names to classes
- **Datasets**: `sim_bench.datasets.base.load_dataset(name, config)` - supports ukbench, holidays, phototriage, flatdir
- **Metrics**: `sim_bench.metrics.factory.MetricFactory` - auto-discovers BaseMetric subclasses
- **Distances**: `sim_bench.distances.base.create_distance_strategy(config)` - cosine, euclidean, chi_square, wasserstein
- **Clustering**: `sim_bench.clustering.base.load_clustering_method(config)` - HDBSCAN, KMeans, hierarchical

### Configuration-Driven Design
Everything is YAML-configured in `configs/`:
- `configs/dataset.*.yaml` - Dataset paths and settings
- `configs/methods/*.yaml` - Feature extraction parameters and distance strategies
- `configs/run.yaml` - Metrics, sampling, output settings
- `configs/siamese_e2e/`, `configs/ava/`, `configs/face/` - Training configs
- `configs/global_config.yaml` - Global settings for ModelHub

### Key Entry Points
- `sim_bench/cli.py` - Main CLI interface with comma-separated method/dataset lists
- `sim_bench/experiment_runner.py` - ExperimentRunner for single experiments, BenchmarkRunner for multi-dataset
- `sim_bench/model_hub/hub.py` - ModelHub: unified interface with lazy-loaded models and feature caching

### Data Flow
1. CLI parses args → loads run.yaml + dataset.*.yaml + methods/*.yaml
2. ExperimentRunner creates dataset (BaseDataset subclass) and method (BaseMethod subclass)
3. Method extracts features → distance strategy computes distances → metrics evaluate rankings
4. ResultManager saves to `outputs/<timestamp>/<method>/`

### Module Structure
- `sim_bench/api/` - **FastAPI backend** (routers, services, database models, schemas)
- `sim_bench/pipeline/` - **18-step pipeline engine** (steps, context, registry, executor, cache)
- `sim_bench/feature_extraction/` - Feature extractors inheriting BaseMethod
- `sim_bench/datasets/` - Dataset loaders inheriting BaseDataset
- `sim_bench/metrics/` - Evaluation metrics inheriting BaseMetric
- `sim_bench/distances/` - Distance computation strategies
- `sim_bench/training/` - Model training scripts (Siamese, AVA, face recognition)
- `sim_bench/model_hub/` - Unified ModelHub interface + ImageMetrics dataclass
- `sim_bench/face_pipeline/` - Face detection, cropping, pose estimation
- `sim_bench/face_recognition/` - ArcFace training and inference
- `sim_bench/portrait_analysis/` - MediaPipe-based portrait analysis
- `sim_bench/clustering/` - Clustering algorithms (HDBSCAN, hierarchical)
- `sim_bench/album/` - Album organization services and domain models
- `app/streamlit/` - **Streamlit frontend** (pages, components, API client, models)

### Adding New Components
1. **New pipeline step**: Create class in `sim_bench/pipeline/steps/your_step.py` inheriting `BaseStep`, decorate with `@register_step`, define metadata (requires, produces, depends_on)
2. **New method**: Create class in `sim_bench/feature_extraction/your_method.py` inheriting `BaseMethod`, add YAML config in `configs/methods/`, add entry to `method_registry` dict in `base.py:load_method()`
3. **New dataset**: Create class in `sim_bench/datasets/your_dataset.py` inheriting `BaseDataset`, add YAML config `configs/dataset.*.yaml`, add to `load_dataset()` in `base.py`
4. **New metric**: Create class in `sim_bench/metrics/your_metric.py` inheriting `BaseMetric` - auto-discovered by factory
5. **New API endpoint**: Add router in `sim_bench/api/routers/`, register in `main.py`, add schemas in `schemas/`, implement service logic in `services/`

## Testing

- Use `--quick` flag for fast iteration with small subsets
- Sample images in `samples/` folder for validation
- Tests in `tests/` run with pytest
- Manual testing: Start backend + Streamlit, test full workflow
- File compilation: `python -m py_compile <file>` to verify syntax
- `sim_bench/face_recognition/verify_split.py` for face model validation

## Debugging Tips

### Common Issues
1. **Images not rotating**: Ensure `ImageOps.exif_transpose()` is used before display
2. **Model not loading**: Check nested config structure (e.g., `config["siamese"]["checkpoint_path"]`)
3. **Clustering parameters ignored**: Verify all params passed to HDBSCAN constructor
4. **Mixed type errors in DataFrames**: Convert all column values to consistent types (e.g., all strings)
5. **Cache not working**: Check `UniversalCache` table exists, verify mtime tracking

### Key Config Files
- `configs/pipeline.yaml` - Pipeline step configuration
- `configs/global_config.yaml` - Model checkpoints and global settings
- Database: `sim_bench.db` (SQLite) - Albums, results, cache, people

## Output Structure

### Benchmarking Results
Results saved to `outputs/<timestamp>/<method>/`:
- `metrics.csv` - Overall performance (Accuracy, Recall@k, Precision@k, mAP@k, N-S Score)
- `per_query.csv` - Per-query detailed metrics
- `rankings.csv` - Full ranking lists
- `summary.csv` - Cross-method comparison

### Trained Models
Model checkpoints saved to:
- `outputs/ava/gpu_run_regression_18_01/best_model.pt` - AVA aesthetic model (96 MB)
- `outputs/siamese_e2e/20260113_073023/best_model.pt` - Siamese comparison model (94 MB)
- Referenced in `configs/global_config.yaml` and `configs/pipeline.yaml`

### Database
- `sim_bench.db` - SQLite database with tables: Album, PipelineRun, PipelineResult, UniversalCache, Person

### Logs
- `logs/` - Application logs with timestamps and structured output

---

## Change Log Location

**Always maintain**: `CHANGES_LOG.md` at project root
- Append after every code modification
- Include timestamp, files, change description, reason
- This helps track development history and debug issues
