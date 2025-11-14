# Budapest2025_Google Dataset

## Overview

Personal photo collection from Budapest (Google Photos export). This is a **flat directory dataset with no ground truth**, suitable for **clustering only** (not evaluation).

## Dataset Statistics

- **Location**: D:\Budapest2025_Google
- **Total images**: 310
  - JPEG files: 122 (39.4%)
  - HEIC files: 188 (60.6%)
- **Structure**: All files in single directory
- **Ground truth**: None (personal collection)

## Setup

### Step 1: Convert HEIC to JPEG

Many feature extractors work best with JPEG. Run the conversion script:

```bash
python convert_heic_to_jpeg.py
```

This will:
- Convert all `.heic` files to `.jpg` (quality=95)
- Keep original HEIC files (delete manually if needed)
- Skip files that already have JPEG versions
- Take ~1-2 minutes for 188 files

### Step 2: Verify Dataset

```bash
# After conversion, you should have 310 JPEG files
dir D:\Budapest2025_Google\*.jpg /B | find /C /V ""
```

## Configuration

### Dataset Config

Located at `configs/dataset.budapest.yaml`:

```yaml
name: budapest
root: "D:/Budapest2025_Google"
subdirs:
  images: "."  # All in root directory
pattern:
  - "*.jpg"
  - "*.JPG"
  - "*.jpeg"
  - "*.JPEG"
```

### Clustering Config

Located at `configs/run.cluster_budapest.yaml`:

```yaml
experiment:
  name: budapest_clustering

dataset: budapest
method: sift_bovw  # Fast; use dinov2 for better results

clustering:
  enabled: true
  algorithm: dbscan
  params:
    metric: cosine
    eps: 0.4          # Tune between 0.3-0.5
    min_samples: 3

output_dir: outputs/cluster_runs/budapest_clustering
```

## Usage

### Quick Clustering with SIFT (Fast)

```bash
# 1. Convert HEIC files
python convert_heic_to_jpeg.py

# 2. Run clustering (~45 seconds for 310 images)
python -m sim_bench.cli --run-config configs/run.cluster_budapest.yaml
```

### Better Clustering with DINOv2 (Requires PyTorch)

```bash
# 1. Edit config: method: dinov2
# 2. Activate venv
.venv\Scripts\activate.bat

# 3. Run clustering (~3-5 minutes on CPU)
python -m sim_bench.cli --run-config configs/run.cluster_budapest.yaml
```

## Results

Clustering outputs will be saved to:
```
outputs/cluster_runs/budapest_clustering/YYYY-MM-DD_HH-MM-SS/
├── clusters.csv           # Image path, cluster_id
├── cluster_stats.json     # Statistics (n_clusters, sizes, etc.)
└── experiment.log         # Execution log
```

### Understanding Results

**clusters.csv** format:
```csv
image_path,cluster_id
D:\Budapest2025_Google\20250822_080658.jpg,0
D:\Budapest2025_Google\20250822_110536.jpg,0
D:\Budapest2025_Google\20250822_135420.jpg,1
D:\Budapest2025_Google\20250823_093015.jpg,-1
...
```

- **cluster_id >= 0**: Images in a cluster (similar photos)
- **cluster_id = -1**: Noise/outliers (DBSCAN only)

**cluster_stats.json** example:
```json
{
  "algorithm": "dbscan",
  "n_clusters": 45,
  "n_noise": 67,
  "noise_ratio": 0.216,
  "cluster_sizes": {
    "0": 12,
    "1": 8,
    "2": 6,
    ...
  }
}
```

## Tuning DBSCAN Parameters

If you get poor results, adjust `eps`:

| eps Value | Result | When to Use |
|-----------|--------|-------------|
| 0.1-0.2 | Everything is noise (-1) | Never (too strict) |
| 0.3-0.4 | Many small tight clusters | Similar composition/lighting |
| 0.5-0.6 | Fewer, larger clusters | Same location/event |
| 0.7+ | Everything in one cluster | Too loose |

**Recommended starting point**: `eps: 0.4`

## Alternative: KMeans

If you want a fixed number of clusters:

```yaml
clustering:
  algorithm: kmeans
  params:
    n_clusters: 20    # Specify number of clusters
    n_init: 10
    random_state: 42
```

**Pros**: Guaranteed clusters, no noise
**Cons**: Must specify cluster count

## Expected Clusters for Budapest Photos

Typical clusters might include:
- Photos from same landmark/location
- Photos with similar composition
- Photos taken in sequence (bursts)
- Similar lighting conditions
- Same people/subjects

## Time Estimates

For 310 images:

| Method | Time | Quality |
|--------|------|---------|
| **SIFT BoVW** | ~45 sec | Good for geometric similarity |
| **DINOv2 (CPU)** | ~3-5 min | Best for semantic similarity |
| **DINOv2 (GPU)** | ~1 min | Best + fastest (if GPU available) |

## Notes

- **No evaluation metrics**: Since there's no ground truth, evaluation metrics (mAP, recall) are meaningless
- **Clustering only**: Perfect use case for exploratory analysis
- **HEIC conversion**: Recommended but not strictly required (PIL can read HEIC with `pillow-heif`)
- **File naming**: Timestamp-based names (YYYYMMDD_HHMMSS.jpg) from Google Photos export





