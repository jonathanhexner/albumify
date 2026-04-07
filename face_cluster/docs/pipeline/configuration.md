# Pipeline Configuration Reference

**File**: `configs/face_clustering_experiment.yaml`

---

## Complete Configuration

```yaml
pipeline:
  name: "face_clustering_experiment"
  description: "Pipeline for face clustering experimentation and ML training"

  steps:
    - discover_images
    - insightface_detect_faces
    - align_faces
    - extract_face_embeddings
    - filter_quality_gate
    - build_knn_graph
    - cluster_connected_components
    - select_exemplars
    - compute_debug_distances
    - export_for_labeling

step_configs:
  # Image Discovery
  discover_images:
    extensions:
      - ".jpg"
      - ".jpeg"
      - ".png"
      - ".heic"
      - ".HEIC"
    recursive: true

  # Face Detection (InsightFace)
  insightface_detect_faces:
    model_name: 'buffalo_l'
    det_size: [640, 640]
    conf_threshold: 0.5

  # Face Alignment
  align_faces:
    target_size: 112
    use_insightface_alignment: true

  # Face Embedding Extraction
  extract_face_embeddings:
    backend: 'insightface'
    model_name: 'buffalo_l'
    normalize: true
    verify_norm: true

  # Quality Gate Filtering
  filter_quality_gate:
    yaw_max: 45.0
    pitch_max: 30.0
    roll_max: 30.0
    blur_min: 100.0
    min_face_area: null
    max_faces_per_image_core: 10
    use_pose_estimation: false

  # Build KNN Graph
  build_knn_graph:
    K: 5
    distance_threshold: 0.35

  # Cluster Connected Components
  cluster_connected_components:
    min_cluster_size: 2
    split_enabled: false

  # Select Exemplars
  select_exemplars:
    d10_k: 10
    exemplars_d10_threshold: 0.25
    exemplar_suppression_radius: 0.15
    N_exemplars_max: 5

  # Compute Debug Distances
  compute_debug_distances:
    n_closest_within: 5
    n_furthest_within: 5
    n_closest_cross: 5

  # Export for Labeling
  export_for_labeling:
    output_dir: null  # Set via CLI
```

---

## Step-by-Step Configuration

### 1. Image Discovery

```yaml
discover_images:
  extensions:
    - ".jpg"
    - ".jpeg"
    - ".png"
    - ".heic"
    - ".HEIC"
  recursive: true
```

**Parameters**:
- `extensions`: List of image file extensions to process
- `recursive`: If true, search subdirectories recursively

**Usage**: Scans album directory for images

---

### 2. Face Detection

```yaml
insightface_detect_faces:
  model_name: 'buffalo_l'     # InsightFace model name
  det_size: [640, 640]        # Detection resolution
  conf_threshold: 0.5         # Minimum confidence (0.0-1.0)
```

**Parameters**:
- `model_name`: InsightFace model (`'buffalo_l'`, `'buffalo_s'`, etc.)
- `det_size`: Detection resolution `[width, height]` (higher = slower but more accurate)
- `conf_threshold`: Minimum face detection confidence

**Tuning**:
- **High precision**: `conf_threshold: 0.7`, `det_size: [640, 640]`
- **High recall**: `conf_threshold: 0.3`, `det_size: [1280, 1280]`
- **Fast**: `conf_threshold: 0.5`, `det_size: [320, 320]`

---

### 3. Face Alignment

```yaml
align_faces:
  target_size: 112                   # Output size (112×112)
  use_insightface_alignment: true    # Use InsightFace native alignment
```

**Parameters**:
- `target_size`: Output crop size (112 for ArcFace models)
- `use_insightface_alignment`: Use InsightFace's alignment (recommended)

**Note**: Always use `target_size: 112` for ArcFace embeddings

---

### 4. Face Embedding Extraction

```yaml
extract_face_embeddings:
  backend: 'insightface'      # Backend: 'insightface' or 'custom'
  model_name: 'buffalo_l'     # Model name (for insightface backend)
  normalize: true             # L2-normalize embeddings
  verify_norm: true           # Check embeddings are normalized
```

**Parameters**:
- `backend`:
  - `'insightface'` (recommended): Use InsightFace native w600k_r50 model
  - `'custom'`: Use custom trained ArcFace model
- `model_name`: Model name (for insightface backend)
- `normalize`: L2-normalize embeddings (required for cosine distance)
- `verify_norm`: Validate normalization (assert norm ≈ 1.0)

**Recommendation**: Always use `backend: 'insightface'` for production

---

### 5. Quality Gate Filtering

```yaml
filter_quality_gate:
  yaw_max: 45.0                      # Max absolute yaw (degrees)
  pitch_max: 30.0                    # Max absolute pitch (degrees)
  roll_max: 30.0                     # Max absolute roll (degrees)
  blur_min: 100.0                    # Min blur score (Laplacian variance)
  min_face_area: null                # Min face area in pixels (optional)
  max_faces_per_image_core: 10       # Max faces per image for core set
  use_pose_estimation: false         # Use SixDRepNet pose estimation
```

**Parameters**:
- `yaw_max`: Max absolute yaw angle (0° = frontal, ±90° = profile)
- `pitch_max`: Max absolute pitch angle (0° = level, ±90° = up/down)
- `roll_max`: Max absolute roll angle (0° = upright, ±90° = sideways)
- `blur_min`: Min Laplacian variance (100 = moderate quality)
- `min_face_area`: Min face area in pixels (null = no filter)
- `max_faces_per_image_core`: Max faces per image in core set (prevents group photo overload)
- `use_pose_estimation`: Compute pose using SixDRepNet (slow, requires GPU)

**Presets**:

**High Precision** (fewer faces, higher quality):
```yaml
yaw_max: 30.0
blur_min: 200.0
max_faces_per_image_core: 3
```

**High Recall** (more faces, tolerate lower quality):
```yaml
yaw_max: 60.0
blur_min: 50.0
max_faces_per_image_core: 20
```

**Balanced** (recommended):
```yaml
yaw_max: 45.0
blur_min: 100.0
max_faces_per_image_core: 10
```

---

### 6. Build KNN Graph

```yaml
build_knn_graph:
  K: 5                        # Number of nearest neighbors
  distance_threshold: 0.35    # Max distance for edge creation
```

**Parameters**:
- `K`: Number of nearest neighbors (5-10 typical)
- `distance_threshold`: Max cosine distance for creating edges (0.0-2.0)

**Tuning**:
- **Strict** (fewer edges): `K: 3`, `distance_threshold: 0.30`
- **Lenient** (more edges): `K: 10`, `distance_threshold: 0.40`
- **Balanced**: `K: 5`, `distance_threshold: 0.35`

**Distance scale**:
- 0.00-0.25: Very similar (same person, similar pose)
- 0.25-0.40: Moderately similar (same person, different pose OR different person, similar appearance)
- 0.40-0.60: Dissimilar (different people)
- 0.60+: Very dissimilar (completely different)

---

### 7. Cluster Connected Components

```yaml
cluster_connected_components:
  min_cluster_size: 2         # Min cluster size (smaller = noise)
  split_enabled: false        # Enable cluster splitting (Phase 2)
```

**Parameters**:
- `min_cluster_size`: Min faces per cluster (smaller components marked as noise)
- `split_enabled`: Enable cluster splitting for wide clusters (Phase 2 feature, not yet implemented)

**Tuning**:
- **Conservative** (no singletons): `min_cluster_size: 2`
- **Aggressive** (allow singletons): `min_cluster_size: 1`

---

### 8. Select Exemplars

```yaml
select_exemplars:
  d10_k: 10                          # K for d10 metric
  exemplars_d10_threshold: 0.25      # Max d10 for candidates
  exemplar_suppression_radius: 0.15  # Min distance between exemplars
  N_exemplars_max: 5                 # Max exemplars per cluster
```

**Parameters**:
- `d10_k`: K for d10 density metric (distance to Kth neighbor)
- `exemplars_d10_threshold`: Max d10 value for exemplar candidates
- `exemplar_suppression_radius`: Min distance between selected exemplars
- `N_exemplars_max`: Max exemplars per cluster

**Presets**:

**High Quality** (strict, few exemplars):
```yaml
d10_k: 10
exemplars_d10_threshold: 0.20
exemplar_suppression_radius: 0.20
N_exemplars_max: 3
```

**Diverse Coverage** (relaxed, more exemplars):
```yaml
d10_k: 10
exemplars_d10_threshold: 0.30
exemplar_suppression_radius: 0.10
N_exemplars_max: 10
```

---

### 9. Compute Debug Distances

```yaml
compute_debug_distances:
  n_closest_within: 5         # Number of closest within-cluster neighbors
  n_furthest_within: 5        # Number of furthest within-cluster neighbors
  n_closest_cross: 5          # Number of closest cross-cluster neighbors
```

**Parameters**:
- `n_closest_within`: How many closest neighbors within cluster to store
- `n_furthest_within`: How many furthest neighbors within cluster to store
- `n_closest_cross`: How many closest neighbors from other clusters to store

**Usage**: Pre-computes neighbors for debug UI visualization

**Memory**: Each face stores 15 neighbors (5+5+5), ~200 bytes per face

---

### 10. Export for Labeling

```yaml
export_for_labeling:
  output_dir: null            # Output directory (set via CLI)
```

**Parameters**:
- `output_dir`: Where to export clustering results (set via CLI argument)

**Outputs**:
- `faces.csv`: Face metadata with cluster assignments
- `clusters.csv`: Cluster statistics
- `face_crops/`: Aligned 112×112 face crops
- `debug_neighbors.json`: Pre-computed neighbors for UI
- `export_summary.json`: Metadata and validation results

---

## Running the Pipeline

### Command Line

```bash
python scripts/run_face_clustering_pipeline.py \
    --album /path/to/photos \
    --output results/my_experiment \
    --config configs/face_clustering_experiment.yaml
```

### Programmatic

```python
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import get_registry
import yaml

# Load config
with open('configs/face_clustering_experiment.yaml') as f:
    config = yaml.safe_load(f)

# Override output directory
config['step_configs']['export_for_labeling']['output_dir'] = 'results/test'

# Create context
context = PipelineContext(source_directory='test_data/face_clustering')

# Execute
executor = PipelineExecutor(get_registry())
result = executor.execute(
    context,
    step_names=config['pipeline']['steps'],
    config=config
)
```

---

## Configuration Tips

### For Small Albums (< 100 faces)
```yaml
# Use lenient thresholds to avoid over-filtering
filter_quality_gate:
  blur_min: 50.0
  yaw_max: 60.0

build_knn_graph:
  K: 3
  distance_threshold: 0.40
```

### For Large Albums (> 1000 faces)
```yaml
# Use strict thresholds to reduce computation
filter_quality_gate:
  blur_min: 150.0
  max_faces_per_image_core: 5

build_knn_graph:
  K: 5
  distance_threshold: 0.30
```

### For High Accuracy
```yaml
# Strict quality gate + conservative clustering
filter_quality_gate:
  yaw_max: 30.0
  blur_min: 200.0

build_knn_graph:
  distance_threshold: 0.30

cluster_connected_components:
  min_cluster_size: 3
```

---

## Validation

After running pipeline, check `export_summary.json`:

```json
{
  "validations": {
    "faces_count_matches_face_records": true,
    "crops_exist_for_all_faces": true,
    "no_null_image_paths": true,
    "face_ids_sequential": true,
    "debug_neighbors_complete": true
  }
}
```

If any validation fails, there's a bug in the pipeline.

---

**See Also**:
- [Pipeline Overview](overview.md) - Complete pipeline flow
- [Pipeline Steps](steps.md) - Individual step documentation
- [Troubleshooting](troubleshooting.md) - Common configuration issues
