# Model Hub

Unified interface to all image analysis models.

## Overview

The Model Hub provides a single entry point for all ML models used in image analysis:
- Technical quality assessment (IQA)
- Aesthetic quality (AVA)
- Portrait analysis (MediaPipe)
- Feature extraction (DINOv2, etc.)
- Clustering (HDBSCAN, K-Means, etc.)

## Usage

```python
from sim_bench.config import GlobalConfig
from sim_bench.model_hub import ModelHub
from pathlib import Path

# Initialize with config
config = GlobalConfig.get_instance().as_dict()
hub = ModelHub(config)

# Single image analysis
metrics = hub.analyze_image(Path("photo.jpg"))
print(f"IQA: {metrics.iqa_score}, AVA: {metrics.ava_score}")
print(f"Portrait: {metrics.is_portrait}, Eyes open: {metrics.eyes_open}")

# Batch analysis
paths = [Path("img1.jpg"), Path("img2.jpg"), Path("img3.jpg")]
all_metrics = hub.analyze_batch(paths, include_features=True)

# Feature extraction and clustering
features = hub.extract_features(paths)
labels, stats = hub.cluster_images(features)
```

## Components

### ImageMetrics

Unified data structure for all image metrics:

```python
@dataclass
class ImageMetrics:
    image_path: str
    
    # Technical quality (0-1 scale)
    iqa_score: Optional[float]
    sharpness: Optional[float]
    exposure: Optional[float]
    
    # Aesthetics (1-10 scale)
    ava_score: Optional[float]
    
    # Portrait metrics
    has_face: bool
    is_portrait: bool
    eyes_open: Optional[bool]
    is_smiling: Optional[bool]
    
    # Clustering
    scene_embedding: Optional[np.ndarray]
    cluster_id: Optional[int]
```

### ModelHub API

**Quality Assessment:**
- `score_quality(image_path)` - Technical quality scores
- `score_aesthetics(image_path)` - AVA aesthetic score
- `compare_images(img1, img2)` - Compare two images

**Portrait Analysis:**
- `analyze_portrait(image_path)` - Face, eyes, smile detection

**Feature Extraction:**
- `extract_features(image_paths)` - Extract embeddings for clustering
- `cluster_images(features)` - Cluster based on features

**Unified Analysis:**
- `analyze_image(path)` - Complete analysis of single image
- `analyze_batch(paths)` - Batch processing with progress callback

## Configuration

All models are configured via `configs/global_config.yaml`:

```yaml
device: cuda  # or cpu

album:
  clustering:
    feature_method: dinov2
    method: hdbscan
    min_cluster_size: 5

portrait_analysis:
  face_detection_confidence: 0.5
  eye_open_ear_threshold: 0.2
  smile_width_threshold: 0.15

quality_assessment:
  ava_checkpoint: path/to/ava_model.pth
```

## Design Principles

1. **Config-only constructors** - All models receive config dict
2. **Lazy loading** - Models loaded on first use
3. **Unified interface** - Single entry point for all analyses
4. **Batch support** - Efficient processing of multiple images
5. **Progress tracking** - Callbacks for UI integration
