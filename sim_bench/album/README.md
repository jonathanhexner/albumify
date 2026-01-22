# Album Organization Module

Workflow pipeline for organizing photo albums with quality filtering, clustering, and best image selection.

## Overview

The album module provides an end-to-end workflow for:
1. Discovering images in a directory
2. **Preprocessing with thumbnails** (50%+ speedup)
3. Analyzing quality and portrait metrics
4. Filtering by quality thresholds
5. Clustering similar images
6. Selecting best images per cluster
7. Exporting results
8. **Performance telemetry** for debugging

## Usage

### Basic Workflow

```python
from pathlib import Path
from sim_bench.album import create_album_workflow

# Create workflow with default settings
workflow = create_album_workflow(
    source_directory=Path("~/Photos/Vacation2024"),
    album_name="Summer Vacation"
)

# Run complete pipeline
result = workflow.run(
    source_directory=Path("~/Photos/Vacation2024"),
    output_directory=Path("~/Photos/Organized/Vacation2024")
)

# Check results
print(f"Processed {result.total_images} images")
print(f"Found {len(result.clusters)} clusters")
print(f"Selected {len(result.selected_images)} best images")

# Access telemetry
print(f"Total time: {result.telemetry.total_duration_sec:.1f}s")
for timing in result.telemetry.timings:
    print(f"  {timing.name}: {timing.duration_sec:.2f}s")
```

### With Enhanced Progress Callback

```python
def show_progress(stage: str, progress: float, operation: str = None, image_name: str = None):
    """Enhanced callback showing detailed progress."""
    if operation and image_name:
        print(f"{stage}: {operation} - {image_name}")
    else:
        print(f"{stage}: {progress*100:.0f}%")

result = workflow.run(
    source_directory=Path("~/Photos"),
    output_directory=Path("~/Photos/Best"),
    progress_callback=show_progress
)
```

### Custom Configuration

```python
# Override specific settings
overrides = {
    'album': {
        'quality': {'min_ava_score': 6.0},  # More strict
        'portrait': {'require_eyes_open': False},
        'selection': {'images_per_cluster': 2}
    }
}

workflow = create_album_workflow(
    source_directory=Path("~/Photos"),
    album_name="My Album",
    overrides=overrides
)
```

## Pipeline Stages

### 1. Discover Images
Scans source directory for image files (jpg, png, heic, raw).

### 2. Preprocess Images (NEW)
**Performance Optimization**: Generates thumbnails at appropriate resolutions:
- **1024px** for IQA and feature extraction
- **2048px** for portrait/face detection
- **Cached to disk** for reuse across runs
- **50-65% faster** than analyzing full-resolution images

### 3. Analyze Quality
Runs quality assessment (IQA, AVA) and portrait analysis (face, eyes, smile) using thumbnails.

### 4. Filter Quality
Applies minimum thresholds:
- IQA score (technical quality)
- AVA score (aesthetic quality)
- Sharpness

### 5. Filter Portrait
For portrait images:
- Optionally require eyes open
- Apply portrait-specific preferences

### 6. Extract Features
Extracts embeddings using DINOv2 or other feature extractors.

### 7. Cluster Images
Groups similar images using HDBSCAN or other clustering algorithms.
- **Configurable min_cluster_size** (1 = allow singletons)
- Noise handling options

### 8. Select Best
Selects best N images per cluster based on weighted scoring:
- AVA weight (aesthetic quality)
- IQA weight (technical quality)
- Portrait weight (eyes open, smiling)

### 9. Export Results
Copies selected images to output directory, optionally organized by cluster.
- Exports telemetry JSON with performance metrics

## Configuration

Settings are defined in `configs/global_config.yaml`:

```yaml
album:
  # NEW: Preprocessing for 50%+ speedup
  preprocessing:
    enabled: true                # Enable thumbnail preprocessing
    cache_thumbnails: true       # Cache to disk for reuse
    num_workers: 4               # Parallel workers
  
  quality:
    min_iqa_score: 0.3
    min_ava_score: 4.0
    min_sharpness: 0.2
  
  portrait:
    require_eyes_open: true
    prefer_smiling: true
    smile_importance: 0.3
    eyes_open_importance: 0.4
  
  clustering:
    method: hdbscan
    feature_method: dinov2
    min_cluster_size: 3          # 1 = allow singletons
    handle_noise: as_individual_clusters
  
  selection:
    images_per_cluster: 1
    ava_weight: 0.5
    iqa_weight: 0.2
    portrait_weight: 0.3
  
  export:
    format: folder
    organize_by_cluster: true
    include_thumbnails: true
```

## WorkflowResult

The `run()` method returns a `WorkflowResult` with:

```python
@dataclass
class WorkflowResult:
    source_directory: Path           # Input directory
    total_images: int                # Total images found
    filtered_images: int             # Images passing filters
    clusters: Dict[int, List[str]]   # Cluster assignments
    selected_images: List[str]       # Best images selected
    metrics: Dict[str, ImageMetrics] # Full metrics for all images
    cluster_stats: Dict[str, Any]    # Cluster statistics
    export_path: Optional[Path]      # Where results were exported
    run_id: Optional[str]            # NEW: Unique run identifier
    telemetry: Optional[WorkflowTelemetry]  # NEW: Performance metrics
```

## Design Principles

1. **Config-only constructors** - All settings from config dict
2. **Stage-based pipeline** - Clear separation of concerns
3. **Progress tracking** - Callback support for UI integration
4. **Flexible filtering** - Configurable quality and portrait thresholds
5. **Smart selection** - Weighted scoring for best image selection
