# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## On Init Checklist
Upon starting a new session:
1. Scan `TODO.md` for open tasks (status `[ ]` or `[>]`)
   - Prioritize in-progress tasks (`[>]`)
   - Check blocked tasks (`[!]`) to see if unblocked
2. Scan `docs/FEATURE_REQUESTS.md` for open requests
3. Check `docs/SIGHTINGS.md` for open/in-progress issues
   - **SIGHTING-007** (Critical, OPEN): Ground truth face crop mapping mismatch — full pipeline E2E test is broken. See `docs/SIGHTINGS.md` for resolution options.
   - **SIGHTING-005** (Critical, OPEN): Pre-clusters contain mixed people (transitive closure problem in kNN graph).
   - **SIGHTING-001** (Critical, IN PROGRESS): Face alignment not working for upside-down faces.
4. Review recent entries in `docs/LEARNINGS.md` (last 5-10 entries) and skim `CHANGES_LOG.md` (last 2 weeks)
   - Look for patterns: multiple related learnings indicate systemic issues
   - Check if current task relates to previous learnings
   - Identify recurring patterns that could become a reusable skill or automated check — propose to user if found

## General
- When asked a question don't necessarily turn to implement new code. Particularly when question suggests there might be a problem, perform first level of debug.
If solution is obvious and can be done quickly. Go for it. Else- please open a sighting with the relevant details and file it under docs/SIGHTINGS.md.
- Always aim to learn from your failures. For any significant bug, missed target, wrong assumption, or unexpected error, produce a failure report in a few lines and append it to `docs/LEARNINGS.md`. Log learnings in up to 3-5 lines with date, newest first.
- After logging a learning, scan both `docs/LEARNINGS.md` and `CHANGES_LOG.md` for recurring patterns that could be codified into a reusable skill or automated check (e.g., a validation script, a test fixture, a CLAUDE.md rule). If a clear skill candidate emerges, propose it to the user.
- For every significant implementation and planning step always review learnings, and ensure we're not repeating errors from the past.
- Unless otherwise instructed, produce plans for approval prior to implementing code changes that are non-trivial (affecting architecture, data flow, configuration, or cross-module behavior).
Small isolated fixes may be implemented directly.
- No feature is considered complete without:
	- Unit test (if applicable)
	- Or explicit justification why test is not required. Use `tests/` folder.
	- **Tests must use production-default config**, not relaxed/special-case values. Relaxed config (e.g. `blur_min=10`) masks bugs that only appear under normal operating conditions. If relaxed config is genuinely needed for a test to run, add a comment explaining why and also add a separate test asserting the correct behavior under default config.
	- **Pipeline integration tests must exercise the holdout path**: at least one test must include faces that fail quality gating (i.e. `core_indices` is a proper subset of all face indices). This catches index-mapping bugs that only appear when quality gating actually rejects faces.
	- **Tests must be run on Windows before declaring them passing.** Unicode characters in print/log output (e.g. `█`, `░`, `✓`) cause `charmap` codec errors on Windows console. Use ASCII-only characters in CLI output.
- **Test naming convention:**
	- Unit test classes are named `ut_<Subject>` (e.g., `ut_QualityGater`, `ut_KNNGraph`). Methods within them are named `test_<scenario>` (e.g., `test_blur_filter_rejects_blurry_face`).
	- E2E and integration tests that are not class-based keep `test_*` module-level functions.
	- pytest is configured in `pyproject.toml` to discover `ut_*` classes and `test_*` functions.
- **Test anti-patterns — never acceptable:**
	- **No `skipif` on missing fixtures.** `pytest.mark.skipif` is for platform/dependency conditions (no GPU, Windows-only). Never use it to guard against missing test data — fixtures must exist in the repo unconditionally. If a fixture is unexpectedly absent at runtime, use `pytest.fail()` explicitly so the failure is visible, not silently skipped.
	- **No `sys.path` manipulation in test files.** `sys.path.insert(...)` in a test is always a sign the package isn't installed. Fix it with `pip install -e .` in `.venv`. If imports fail without path hacking, that is a setup bug to fix, not to work around.
	- **Clustering/grouping tests must assert both purity AND completeness.** Purity alone (no mixed-identity clusters) passes even when output is wrong — e.g., every face in its own cluster. Completeness must also be checked: all faces belonging to the same identity must end up in the same cluster.
	- **No vacuously true assertions.** Never write `assert not some_list` or `assert len(df) > 0` when the list/dataframe could be empty because data failed to load. Always assert on concrete expected values, and separately assert that input data was actually loaded.
	- **File format contracts must be tested end-to-end.** When module A writes a file and module B reads it, there must be a test that calls A's writer then B's reader and validates the data types of each field — not just that the file exists. Never assume the format of a file written by another module without reading its writer first. (SIGHTING-013)
- In any rejection or replan, consider what needs to be changed in CLAUDE.md for future improvement, and propose making changes.
- Break down plans into small individual work items. Add them to `TODO.md` with format: `[STATUS] Task | Date | Owner`
- Always update `TODO.md` after completing tasks (change `[ ]` to `[x]`) or when starting work (change `[ ]` to `[>]`).
- When in doubt ask clarifying questions.
- For any new request from the user always log it as a feature request in docs/FEATURE_REQUESTS.md. After completing the feature you can mark it done, and
after user provided feedback you can mark as verified. If user feedback was negative verify you understood the feedback and check if there is a possible learning.
- README.md: All the high level, getting started, starting apps, etc. goes into the README.md. Note that we have multiple apps in the repo - explain all existing ones and how to start them up.

## ML Development Best Practices
See `docs/ML_DEVELOPER_SKILLS.md` for comprehensive guidelines. Key principles:
- **Benchmark everything** - Save inputs, outputs, configs, metrics for every experiment
- **Build debug panels** - For ML pipelines, show all stages side-by-side (original → intermediate → final)
- **Document coordinate systems** - Use type hints like `landmarks_px` vs `landmarks_norm` to avoid transform mismatches
- **Compare baselines** - Use `scripts/compare_benchmarks.py` to compare runs before/after changes

## Sightings
- Filing - Present briefly: problem description, symptoms, suspicions if any, steps for reproduction, and who is the persona responsible for dealing with it. Status is OPEN.
- Resolution - Verify solution is tested before closing. Also be sure to write to docs/LEARNINGS.md what you learnt from this sighting and how this could have been overcome (e.g., better testing, new understandings, etc.)

## Face Clustering Architecture Rules

⚠️ **SIGHTING-008**: The face clustering subsystem was reworked into `face_cluster/` with `FaceClusteringPipeline` as the single entry point. See `RECOVERY_PLAN.md` for the responsibility table.

See full spec in `RECOVERY_PLAN.md`. Non-negotiable rules:

- **Single public API**: `FaceClusteringPipeline.run(image_dir, output_dir, on_progress=None)` in `face_cluster/pipeline.py` is the ONLY entrypoint. Do not add new orchestration scripts or methods.
- **Algorithms only** in `face_cluster/` (except `crops.py`, `export.py`, and `loader.py` which own file I/O for their stage). No paths, no CLI anywhere else in `face_cluster/`.
- **One production entry point**: `app/face_clustering.py` calls `FaceClusteringPipeline`. No new orchestration scripts.
- **No algorithm code** in `scripts/` or `app/`. They call `face_cluster/`, never implement logic.
- **No null `image_path`** on FaceRecord. Stage 1 validates before writing output.
- **No stage reads from memory**. Every stage reads its input from the previous stage's output file.
- **No test uses a real album path** (except `tests/face_clustering/test_pipeline_e2e.py` which is explicitly designated to use `test_data/face_clustering/source_images/`).
- **Every stage must have a test** before it is considered implemented. Untested stages are incomplete.
- **Notebook insights** that prove useful → promote to `face_cluster/` with a test.
- Before adding any new file to the face clustering subsystem: check the responsibility table in `RECOVERY_PLAN.md`.

## Architecture Discipline Rule

If architectural changes are approved and implemented, update docs/architecture.md to reflect the new state.
Architecture.md must represent the current true system architecture.

For any feature that affects:
- System architecture - compare against docs/architecture.md to know if the architecture has changed.
- Data flow
- Model behavior
- Configuration schema
- Cross-module interfaces
- Memory strategy
- Training/validation logic

Claude must:

1. Provide a structured design breakdown including:
   - Requirement. If a feature introduces a new functional or non-functional requirement, append it to docs/requirements.md with date and short description.
   - Objective
   - Constraints
   - Integration points
   - Data flow
   - Edge cases
   - Risks and trade-offs

2. Wait for explicit approval before writing implementation code.

If ambiguity exists, ask clarifying questions instead of assuming.


## SW Design: Non-Blocking UI — Async Compute Pattern

Any computation that takes more than ~0.2s must **never run on the Streamlit render thread**. Running heavy work on the render thread freezes the entire UI — the user cannot switch tabs, click buttons, or see any feedback.

**Rule**: Every slow operation (pipeline execution, UMAP, embedding distance matrices, model loading) must be dispatched to a background thread and the UI must poll via `st.rerun()`.

**Pattern** (used in `app/face_clustering.py`):
```python
# 1. Start work in a background daemon thread
worker = _AsyncState()
worker.start(heavy_fn, *args)
st.session_state.my_worker = worker
st.rerun()

# 2. On next render: check state, rerun if still running
if worker.is_running:
    st.info("Computing...")
    time.sleep(0.4)
    st.rerun()
elif worker.has_error:
    st.error(worker.error)
else:
    render(worker.result)
```

**Logging**: Route library loggers to a `queue.Queue` via `_QueueHandler`, drain on each rerun, display in `st.expander("Live log")`. This gives users visibility into what is happening without polling files.

**Applies to**: both `app/face_clustering.py` and the main Streamlit app.

## SW Design: Writer-Reader Contracts

Any time module A writes a file that module B reads, the schema is a **contract** — treat it the same way you would treat an API interface.

**Before writing either the writer or the reader:**
1. Define the schema explicitly — field names, types, and nesting — as a typed dataclass or a docstring constant in the writer module. Example: `crop_manifest.json` schema is `Dict[str, str]` mapping `face_id -> relative_crop_path`.
2. Add a `load_<format>(path) -> TypedResult` helper in the writer module so readers never parse raw JSON/CSV themselves.
3. If no helper exists, read the writer source before implementing the reader. Never assume the format.

**Rule**: The schema definition lives in the writer module. The reader imports the loader or references the documented schema. There is no implicit contract.

**Testing requirement** (see also Test anti-patterns below): every writer-reader pair must have a contract test — runs the writer in a `tmp_path`, calls the reader, asserts field types and values. This catches schema drift before it reaches the app.

## Coding
- Python 3.10+ required. Use type hints consistently.
- `__init__.py` files should be kept empty, except for `face_cluster/__init__.py` which exports the public API (`FaceClusteringPipeline`, `FaceRecord`, etc.).
- Never use local or relative imports. Always use full imports (e.g., `from sim_bench.pipeline.base import BaseStep`).
- Avoid excessive Try/Except. Keep it only for extreme cases where output is unpredictable.
- Avoid excessive If statements. Prefer using strategy or factory pattern.
- Avoid usage of prints, prefer usage of proper logging. Make sure we support logging injection for centralized logging.
- **Protobuf compatibility**: Use `protobuf>=3.20,<4` (MediaPipe requires this version range).
- **Metadata Storage**: When exporting/transforming data that references external files, always save source path, timestamp, and version in metadata JSON. Never reconstruct paths via heuristics.


## Windows Development Notes
- **Python environment**: Use the `.venv` folder in the project root. Activate with `.venv\Scripts\activate` or invoke directly: `.venv\Scripts\python -m pytest ...`
- **Always use venv executables directly**: `.venv\Scripts\streamlit run app/face_clustering.py`, `.venv\Scripts\python -m pytest`. Never use bare `streamlit` or `python` — they resolve to system installs that don't have project packages. If you see `ModuleNotFoundError: No module named 'face_cluster'` or `'sim_bench'`, this is always the cause.
- **After adding a new top-level package directory** (e.g. `face_cluster/`, `my_lib/`), immediately re-run `.venv\Scripts\pip install -e .`. The editable install finder (`__editable___sim_bench_*_finder.py`) only contains packages that existed at install time — it will NOT auto-discover new packages. Verify with: `cd C:/Windows/Temp && D:/sim-bench/.venv/Scripts/python -c "import <new_package>"` (non-CWD to rule out path fallback).
- Use forward slashes in code paths (`docs/LEARNINGS.md`) even on Windows
- Run tests via `.venv\Scripts\python -m pytest` (NOT system `python`) to ensure the correct environment and proper module resolution
- Database path `~/.sim_bench/` resolves to `%USERPROFILE%\.sim_bench\`


## Verification
- Avoid half baked code. Always verify you understand what you're being asked and that the code complies with the request.
- When in doubt always ask questions to verify you understand the request.

Bug Discipline - For every non-trivial bug 
1. Identify: Root cause - Why it wasn’t caught earlier? 
2. Add a prevention mechanism: Test, Validation, Assertion, Architectural constraint
3. Log concise learning in docs/LEARNINGS.md.
Never fix symptoms without addressing systemic cause.

## ⚠️ IMPORTANT: Change Tracking

**After EVERY code change you make**, append an entry to `CHANGES_LOG.md` with:
- Date and time (ISO 8601 format)
- **Category tag**: [FEATURE], [BUGFIX], [REFACTOR], [DOCS], [TEST], [CONFIG], [PERF]
- Files modified
- Brief description of what was changed
- Why it was changed

### Detail Level Guidelines
- **Simple fixes**: 3-5 lines (timestamp, category, files, change, reason)
- **Complex changes**: Add Details section (under 20 lines preferred)
- **Critical bugs**: Include Root Cause, Details, Verification, Lesson sections

### Category Tags
- **[FEATURE]**: New functionality, tools, or capabilities
- **[BUGFIX]**: Fixing incorrect behavior
- **[REFACTOR]**: Code restructuring without changing behavior
- **[DOCS]**: Documentation updates (including SIGHTINGS, LEARNINGS)
- **[TEST]**: Test additions or modifications
- **[CONFIG]**: Configuration file changes
- **[PERF]**: Performance improvements

Example entry:
```markdown
### 2026-02-03 14:30:00 [FEATURE]
**Files**: `app/streamlit/components/gallery.py`
**Change**: Added "Final Score" column to cluster debug table
**Reason**: User requested visibility of composite selection score for debugging
```

### Archiving Policy
- Entries older than 3 months are moved to `archive/CHANGES_YYYY-MM.md`
- Main log stays focused on recent work (last 3 months)
- All history preserved in archive directory
- Archive files organized by month: `archive/CHANGES_2026-01.md`, `archive/CHANGES_2026-02.md`, etc.

If `CHANGES_LOG.md` doesn't exist, create it with the header template from the file.

---

## Project Overview

sim-bench is a Python 3.10+ framework for image similarity benchmarking and image quality assessment:
- **Image similarity/retrieval**: Classical (HSV histograms) and deep learning (ResNet50, DINOv2, OpenCLIP)
- **Quality assessment**: Siamese networks and AVA aesthetic models
- **Face recognition**: ArcFace embeddings, pose estimation, expression scoring
- **Album organization**: Multi-step pipeline with Streamlit frontend + FastAPI backend

For detailed usage, benchmarking results, and dataset configuration, see [README.md](README.md).

## Common Commands

```bash
# See README.md "Applications" section for all app startup commands
# Quick reference for main app:
python -m uvicorn sim_bench.api.main:app --reload --port 8000  # Backend (Terminal 1)
streamlit run app/streamlit/main.py                             # Frontend (Terminal 2)

# Run benchmarks
python -m sim_bench.cli --methods chi_square,deep,dinov2 --datasets ukbench,holidays
python -m sim_bench.cli --quick --methods chi_square --datasets ukbench  # Fast test

# Train models
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
python -m sim_bench.training.train_ava_resnet --config configs/ava/resnet50_cpu.yaml
python -m sim_bench.face_recognition.train --config configs/face/resnet50_arcface.yaml

# Run tests — always use .venv Python on Windows (.venv/ in project root)
.venv/Scripts/python -m pytest tests/                                              # All tests
.venv/Scripts/python -m pytest tests/test_model_hub.py                             # Single file
.venv/Scripts/python -m pytest tests/test_model_hub.py -k "test_hub_initialization" # Single test
.venv/Scripts/python -m pytest tests/pipeline/ -v                                  # Pipeline tests
.venv/Scripts/python -m pytest tests/clustering/ -v                                # Clustering tests
.venv/Scripts/python -m pytest tests/pipeline/test_face_pipeline_e2e.py -v -s      # Face pipeline E2E

# Database maintenance
sqlite3 ~/.sim_bench/sim_bench.db "DELETE FROM universal_cache WHERE feature_type = 'face_embedding'"  # Clear face embeddings
sqlite3 ~/.sim_bench/sim_bench.db "DELETE FROM universal_cache"      # Clear all cached features
rm -rf ~/.sim_bench/image_cache/                                      # Clear image cache

# ML Clustering workflow
python scripts/export_clustering_data.py --embeddings <path>          # Export clustering data
streamlit run app/face_clustering_labeling.py                        # Label clusters
python scripts/train_merge_classifier.py --data <path>                # Train merge model

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
2. `filter_faces` - Remove small/low-confidence faces (marks `filter_passed`)
3. `score_face_frontal` - Compute frontal score, marks `is_clusterable`
4. `extract_face_embeddings` - 5-point align faces using ArcFace template, extract 512-dim embeddings
5. `cluster_people` - HDBSCAN clustering on normalized embeddings → `context.people_clusters`
6. `identity_refinement` (optional) - Attach noise faces to clusters, apply user overrides

**Pose estimation**: InsightFace `buffalo_l` includes the `1k3d68` model which provides `face.pose = [pitch, yaw, roll]` on every detected face — no external pose estimator needed. `face_cluster/embedding.py` reorders to `(yaw, pitch, roll)` to match `FaceRecord` convention. Quality gating in `face_cluster/quality.py` uses these angles directly (always active).

**Face Embedding Backend**: Configured in `pipeline.yaml` under `extract_face_embeddings.backend`:
- `insightface` (default): Uses InsightFace's w600k_r50 model (better rotation invariance, recommended for production)
- `custom`: Uses trained `arcface_resnet50.pt` model (requires face orientation detection step first)

**Backend Selection Guide**:
- Use `insightface` for production: handles rotated faces automatically, more robust
- Use `custom` only if you need specific fine-tuned model weights

**Important**: Embeddings are cached by `(image_path, face_index)`. If embeddings become corrupted (zero vectors), clear the cache:
```bash
sqlite3 ~/.sim_bench/sim_bench.db "DELETE FROM universal_cache WHERE feature_type = 'face_embedding'"
```

### Image Cache System
All pipeline steps use a global image cache (`sim_bench/pipeline/utils/image_cache.py`) that:
- Normalizes EXIF rotation once (prevents bbox coordinate mismatches)
- Stores images in `~/.sim_bench/image_cache/`
- Uses content-based cache keys (EXIF datetime + device info, or file hash fallback)

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
- **Clustering**: `sim_bench.clustering.base.load_clustering_method(config)` - HDBSCAN, KMeans, hierarchical, hybrid_hdbscan_knn, hybrid_closest_face
- **Face Embeddings**: `sim_bench.pipeline.face_embedding.factory.FaceEmbeddingExtractorFactory` - CustomArcFace or InsightFaceNative

### Hybrid Face Clustering
For better clustering results, use `hybrid_hdbscan_knn` algorithm:
1. HDBSCAN creates initial dense clusters
2. Computes per-cluster threshold: T = median(K-NN distances) + 2×IQR
3. Iteratively merges clusters where closest pair ≤ min(T_a, T_b)
4. Attaches noise points to nearest cluster if ≤ threshold
5. Key params: `knn_k=3`, `iqr_multiplier=2.0`, `threshold_floor=0.3`

### Standalone Face Clustering Module
`face_cluster/` - Modular face clustering library (separate from sim_bench pipeline):
- **Purpose**: ML-based merge training, analysis, and experimentation
- **Components**:
  - `pipeline.py` - `FaceClusteringPipeline` — single public API (`run()` method)
  - `embedding.py` - InsightFace wrapper for detection + embedding extraction (uses `face.pose` from 1k3d68)
  - `quality.py` - Quality gating: blur, pose angles, area, top-K per image
  - `knn_graph.py` - Build mutual kNN graphs with distance thresholds
  - `clustering.py` - Connected components clustering on kNN graph
  - `merge.py` - Cluster merging with adaptive thresholds (ConservativeMerger, MLMerger)
  - `exemplars.py` - Select representative faces per cluster (d10-based)
  - `attach.py` - Attach noise faces to clusters (vote+margin strategy)
  - `analysis.py` - Cluster distance matrices, merge decisions, statistics
  - `features.py` - `FeatureComputer` / `ClusterPairFeatures` for ML merge training
  - `export.py` - Writes faces.csv, clusters.csv, embeddings.npy, crop_manifest.json
  - `loader.py` - `load_pipeline_result()` — loads all export artifacts from disk (no model execution)
- **Key Features**:
  - Independent kNN graph clustering (no FAISS, full interpretability)
  - Step-by-step debugging in notebooks
  - Merge analysis with cluster-to-cluster distance matrices
  - Adaptive per-cluster thresholds (median + IQR)
- **When to Use**:
  - Use `face_cluster/` for ML training data generation and experimentation
  - Use `sim_bench/pipeline/` for production face clustering in albums
- **Usage**:
  - Interactive: `notebooks/debug_knn_graph_clustering.ipynb`
  - Batch: `scripts/export_clustering_data.py`
  - Labeling: `app/face_clustering_labeling.py`
- **Integration**: Labeling app uses face_cluster module to generate training data for ML merge classifier

### ML Training Workflow (Face Cluster Merging)
Pipeline for training ML-based cluster merging classifier:

**Phase 1: Export Clustering Data**
```bash
python scripts/export_clustering_data.py --embeddings results/face_clustering_benchmark/embeddings_*.npy
```
Runs kNN graph clustering and exports:
- `faces.csv` - Face metadata (embeddings, quality scores, image paths)
- `clusters.csv` - Cluster statistics (size, diameter, exemplar distances)
- `candidate_pairs.csv` - Cluster pair features for merge candidates
- `export_summary.json` - Metadata (source paths, config, statistics)

**Phase 2: Manual Labeling**
```bash
streamlit run app/face_clustering_labeling.py
```
Labeling interface to assign corrected_identity to faces. Saves corrections to `corrected_identities.json`.

**Phase 3: Training** (In Progress)
```bash
python scripts/train_merge_classifier.py --data results/face_clustering_training/
```
Generates training data from corrected labels, trains logistic regression, evaluates model.

**Key Design Decisions**:
- Keep both heuristic (ConservativeMerger) and ML (MLMerger) for comparison
- Features: min_exemplar_dist, p10_cross_dist, support_fraction, diameter_ratio, adaptive thresholds
- Store all metadata in export_summary.json (never reconstruct paths via heuristics)

### Regenerating Corrupted Face Embeddings

If face embeddings become corrupted or mismatched with face crops:

**1. Verify the Issue**
Use the verification notebook to check if stored embeddings match actual face images:
```bash
# notebooks/verify_face_embeddings.ipynb
# Compare fresh embeddings from crops vs stored embeddings
```

**2. Regenerate from Existing Crops**
```bash
python scripts/regenerate_embeddings_from_crops.py \
    --face-crops results/Google_Germany/face_crops \
    --output results/Google_Germany \
    --metadata results/Google_Germany/benchmark_*.json
```

This script:
- Reads existing aligned face crops (face_XXXX_aligned.jpg)
- Extracts fresh embeddings using shared `face_cluster.InsightFaceEmbedder`
- Saves embeddings with metadata mapping face_id to source image paths
- Output: `embeddings_FRESH_*.npy` and `embeddings_metadata_FRESH_*.json`

**3. Clean Up Old Embeddings**
```bash
# Delete old corrupted embeddings to avoid confusion
rm results/Google_Germany/embeddings_2026-*.npy
```

**4. Re-export Clustering Data**
```bash
python scripts/export_clustering_data.py \
    --embeddings results/Google_Germany/embeddings_FRESH_*.npy \
    --output results/Google_Germany/clustering_export_fresh
```

**Common Causes of Embedding Corruption**:
- Face IDs shuffled between embedding extraction and crop saving
- Different face detection runs producing different face orderings
- Manual editing of face_crops directory without regenerating embeddings
- Stale cache with zero-vector embeddings from previous bugs

**Prevention**:
- Always use `export_summary.json` metadata to track source paths (never reconstruct via heuristics)
- Verify embeddings match crops before running clustering analysis
- Store face_id explicitly in metadata to maintain correspondence

### Configuration
All behavior is YAML-configured in `configs/`:
- `pipeline.yaml` - Pipeline steps and their parameters (most frequently edited)
- `global_config.yaml` - Model checkpoints and global settings
- `dataset.*.yaml` - Dataset paths
- `methods/*.yaml` - Feature extraction configs
- `run.yaml` - Metrics, sampling, output settings
- `clustering_benchmark.yaml` - Clustering algorithm configs for benchmarking

### Key Entry Points
| Entry Point | Purpose |
|-------------|---------|
| `sim_bench/cli.py` | CLI for benchmarking |
| `sim_bench/api/main.py` | FastAPI server |
| `app/streamlit/main.py` | Streamlit frontend (album app) |
| `app/face_clustering.py` | Streamlit app for face clustering (run pipeline, history, analysis) |
| `app/face_clustering_labeling.py` | Streamlit app for ML merge labeling |
| `app/face_clustering_debug/main.py` | Streamlit app for clustering debug views |
| `face_cluster/pipeline.py` | `FaceClusteringPipeline` — standalone clustering API |
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
6. **All faces clustering to one person**: Check for zero-vector embeddings in cache (stale data from previous bugs). Clear face embedding cache (see Database Maintenance commands)
7. **Upside-down faces not correcting**: Ensure `detect_face_orientation` and `align_faces` steps are in pipeline before `extract_face_embeddings`
8. **Notebook vs script produces different results**: Always use EXACT same input files (explicitly specify paths, don't rely on "most recent" logic)
9. **Heuristic path reconstruction fails**: Check if metadata JSON exists with source paths; re-export data if metadata missing

### Database
SQLite database location: `~/.sim_bench/sim_bench.db` (NOT in project root)

**Tables:**
- `albums` - Album metadata
- `pipeline_runs`, `pipeline_results` - Pipeline execution tracking
- `universal_cache` - Step-level feature caching (mtime-based invalidation)
- `people` - Face clusters with identity labels
- `config_profiles` - Saved pipeline configurations
- `faces` - Individual face detections with bbox, embeddings, quality scores

**Clear corrupted face embeddings:**
```bash
sqlite3 ~/.sim_bench/sim_bench.db "DELETE FROM universal_cache WHERE feature_type = 'face_embedding'"
```

### Useful Debug Scripts
- `scripts/benchmark_face_clustering.py` - Benchmark HDBSCAN vs Hybrid clustering methods on saved embeddings
- `scripts/compare_benchmarks.py` - Compare results across different benchmark runs (before/after changes)
- `scripts/compare_clustering_results.py` - Compare two clustering outputs (e.g., notebook vs script)
- `scripts/export_clustering_data.py` - Export face embeddings and cluster data for ML training
- `scripts/face_distance_report.py` - HTML report analyzing distances between two face groups
- `app/face_clustering_comparison.py` - Streamlit app for side-by-side visual clustering comparison
- `notebooks/debug_knn_graph_clustering.ipynb` - Interactive step-by-step clustering pipeline

### Face Clustering Debug App
Run: `streamlit run app/face_clustering_debug/main.py`

Features:
- **Overview** - Gallery with 🔍 button for three-version debug panel (original→raw→aligned)
- **Merge/Attach Decisions** - See threshold values and decision outcomes
- **Algorithm Comparison** - Side-by-side method comparison
- **Dynamic Documentation** - Each clustering method has `doc_explanation` and `decision_parameters`

### Clustering Algorithm Documentation
All clustering methods have:
- `doc_explanation` - 5-6 line explanation of how algorithm works
- `decision_parameters` - Dict of params with description, default, and decision_role
- `get_decision_info()` - Returns current values for UI display

---

## Change Log Location

**Always maintain**: `CHANGES_LOG.md` at project root
