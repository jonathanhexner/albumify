# Album Organization App - Complete Architecture

**Purpose**: This document explains the entire software design of the album organization application, how it uses your trained models, and what each file does.

**Target Audience**: You (the model trainer) who understands the ML models but needs to understand how the application layer works.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Directory Structure](#directory-structure)
3. [Model Loading & Usage](#model-loading--usage)
4. [Component Architecture](#component-architecture)
5. [Data Flow](#data-flow)
6. [File Reference Guide](#file-reference-guide)

---

## System Overview

### What Does This App Do?

The album organization app takes a folder of photos and:
1. **Analyzes** each image (quality, aesthetics, faces, eyes, smiles)
2. **Filters** low-quality images
3. **Clusters** similar images together
4. **Selects** the best image(s) from each cluster
5. **Exports** the selected images

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│  Streamlit Web UI (app/album/)                              │
│  - Configuration Panel                                       │
│  - Workflow Runner                                          │
│  - Results Viewer                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓ calls
┌─────────────────────────────────────────────────────────────┐
│                  WORKFLOW ORCHESTRATION LAYER                │
│  Album Workflow (sim_bench/album/)                          │
│  - Preprocessing (thumbnails)                               │
│  - Pipeline execution (8 stages)                            │
│  - Telemetry tracking                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓ uses
┌─────────────────────────────────────────────────────────────┐
│                    MODEL & ANALYSIS LAYER                    │
│  Model Hub (sim_bench/model_hub/)                           │
│  - IQA Quality Assessment (rule-based)                      │
│  - AVA Aesthetic Scoring (YOUR TRAINED MODEL)               │
│  - Portrait Analysis (MediaPipe)                            │
│  - Feature Extraction (DINOv2/CLIP)                         │
│  - Clustering (HDBSCAN)                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓ loads
┌─────────────────────────────────────────────────────────────┐
│                       TRAINED MODELS                         │
│  Model Checkpoints (.pt / .pth files)                       │
│  - AVA ResNet (aesthetics) - YOUR TRAINED MODEL             │
│  - Siamese CNN (comparisons) - YOUR TRAINED MODEL           │
│  - DINOv2 (features) - Pre-trained                          │
│  - CLIP (features) - Pre-trained                            │
│  - MediaPipe (faces) - Pre-trained                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

### Complete File Map

```
sim-bench/
├── app/                          # Streamlit UI Layer
│   └── album/                    # Album organization UI
│       ├── __init__.py           # Module exports
│       ├── main.py               # Streamlit entry point
│       ├── config_panel.py       # Settings UI (sliders, checkboxes)
│       ├── workflow_runner.py    # Progress display & execution
│       └── results_viewer.py     # Gallery, metrics, performance tabs
│
├── sim_bench/                    # Core Application Layer
│   ├── album/                    # Workflow orchestration
│   │   ├── __init__.py           # Exports: AlbumWorkflow, create_album_workflow
│   │   ├── workflow.py           # Main pipeline (8 stages)
│   │   ├── stages.py             # Individual stage functions
│   │   ├── preprocessor.py       # Thumbnail generation
│   │   ├── telemetry.py          # Performance tracking
│   │   ├── selection.py          # Best image selector
│   │   └── export/               # Export functionality
│   │       ├── base.py           # BaseExporter abstract class
│   │       ├── folder.py         # Folder export
│   │       └── zip.py            # ZIP export
│   │
│   ├── model_hub/                # Model orchestration
│   │   ├── __init__.py           # Exports: ModelHub, ImageMetrics
│   │   ├── hub.py                # Unified model interface
│   │   └── types.py              # ImageMetrics dataclass
│   │
│   ├── portrait_analysis/        # Face/eyes/smile detection
│   │   ├── __init__.py           # Exports: MediaPipePortraitAnalyzer
│   │   ├── analyzer.py           # MediaPipe wrapper
│   │   ├── eye_state.py          # Eye aspect ratio (EAR)
│   │   ├── smile_detection.py    # Smile detection
│   │   └── types.py              # PortraitMetrics, EyeState, SmileState
│   │
│   ├── image_quality_models/     # Model wrappers (THIS IS WHERE MODELS LOAD)
│   │   ├── ava_model_wrapper.py  # AVA aesthetic model (YOUR TRAINED MODEL)
│   │   ├── siamese_model_wrapper.py  # Siamese comparison model (YOUR TRAINED MODEL)
│   │   ├── base_model.py         # Abstract base class
│   │   └── model_factory.py      # Factory for creating models
│   │
│   ├── models/                   # Model architectures (network definitions)
│   │   ├── ava_resnet.py         # AVA ResNet architecture
│   │   ├── siamese_cnn_ranker.py # Siamese CNN architecture
│   │   └── siamese_resnet.py     # Siamese ResNet architecture
│   │
│   ├── quality_assessment/       # Rule-based quality assessment
│   │   └── rule_based.py         # RuleBasedQuality (no trained model)
│   │
│   ├── feature_extraction/       # Feature extractors
│   │   ├── dinov2.py             # DINOv2 (pre-trained)
│   │   ├── openclip.py           # OpenCLIP (pre-trained)
│   │   └── resnet50.py           # ResNet50 (pre-trained)
│   │
│   ├── clustering/               # Clustering algorithms
│   │   ├── hdbscan.py            # HDBSCAN clustering
│   │   ├── dbscan.py             # DBSCAN clustering
│   │   └── kmeans.py             # K-Means clustering
│   │
│   └── image_processing/         # Image utilities
│       └── thumbnail.py          # Multi-resolution thumbnails
│
├── configs/                      # Configuration files
│   └── global_config.yaml        # Main config (paths, thresholds, settings)
│
└── [MODEL CHECKPOINTS]           # Where trained models should be stored
    └── best_model.pt             # YOUR TRAINED AVA MODEL (path configured in code)
```

---

## Model Loading & Usage

### Critical: Which Models Are Used?

The app uses **FIVE types of models**:

| Model | Type | File Loaded | Purpose | Source |
|-------|------|-------------|---------|--------|
| **AVA ResNet** | Trained | `best_model.pt` | Aesthetic scoring (1-10) | **YOUR TRAINING** |
| **Siamese CNN** | Trained | `best_model.pt` | Image comparison | **YOUR TRAINING** |
| **MediaPipe** | Pre-trained | (auto-download) | Face/eyes/smile detection | Google |
| **DINOv2** | Pre-trained | (auto-download) | Image embeddings | Meta |
| **HDBSCAN** | Algorithm | (no weights) | Clustering | scikit-learn |

### Model Loading Locations

#### 1. AVA Model (Aesthetic Scoring)

**Where loaded**: `sim_bench/image_quality_models/ava_model_wrapper.py`

```python
class AVAQualityModel(BaseQualityModel):
    def __init__(self, checkpoint_path: Path, device: str = 'cpu'):
        # Loads YOUR trained model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint['config']
        
        # Create architecture
        self.model = AVAResNet(self.config['model'])
        
        # Load YOUR trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
```

**Expected checkpoint structure**:
```python
{
    'model_state_dict': <trained_weights>,
    'config': {
        'model': {...},
        'transform': {...}
    },
    'epoch': <last_epoch>,
    'val_spearman': <validation_score>
}
```

**How to specify checkpoint**:
Currently in `sim_bench/model_hub/hub.py` line 140-147:

```python
def score_aesthetics(self, image_path: Path) -> Optional[float]:
    ava_checkpoint = self._config.get('quality_assessment', {}).get('ava_checkpoint')
    if not ava_checkpoint:
        return None  # AVA disabled if no checkpoint
    
    if self._ava_model is None:
        self._ava_model = AVAQualityModel(Path(ava_checkpoint), self._device)
```

**To enable AVA**: Add to `configs/global_config.yaml`:

```yaml
quality_assessment:
  ava_checkpoint: path/to/your/best_model.pt
```

#### 2. Siamese Model (Image Comparison)

**Where loaded**: `sim_bench/image_quality_models/siamese_model_wrapper.py`

**Current status**: NOT USED in album workflow yet (deferred feature)

**When needed**: For tiebreaker comparisons in selection

#### 3. MediaPipe (Portrait Analysis)

**Where loaded**: `sim_bench/portrait_analysis/analyzer.py` line 58-69

```python
def _load_models(self):
    import mediapipe as mp
    self._face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=self._face_confidence
    )
    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    )
```

**Auto-downloads on first use** - no manual setup needed

#### 4. DINOv2 (Feature Extraction)

**Where loaded**: `sim_bench/feature_extraction/dinov2.py`

**Auto-downloads from Hugging Face** - no manual setup needed

#### 5. Rule-Based IQA (Technical Quality)

**Where implemented**: `sim_bench/quality_assessment/rule_based.py`

**No trained model** - uses OpenCV algorithms (sharpness, exposure, colorfulness)

---

## Component Architecture

### 1. User Interface (Streamlit)

**Entry Point**: `app/album/main.py`

```python
def main():
    # 1. Render config panel
    config_overrides = render_album_config()
    
    # 2. Get album info from user
    form_data = render_workflow_form()
    
    # 3. Run workflow
    result = render_workflow_runner(
        source_directory,
        output_directory,
        album_name,
        config_overrides
    )
    
    # 4. Display results
    render_results(result)
```

**Files**:
- `config_panel.py`: All UI sliders, checkboxes, settings
- `workflow_runner.py`: Progress bar, status updates, execution
- `results_viewer.py`: Gallery, metrics table, performance charts

### 2. Workflow Orchestration

**Main Class**: `AlbumWorkflow` in `sim_bench/album/workflow.py`

```python
class AlbumWorkflow:
    def __init__(self, config):
        self._hub = ModelHub(config)          # Model orchestrator
        self._preprocessor = ImagePreprocessor(config)  # Thumbnail generator
    
    def run(self, source_directory, output_directory):
        # 8-stage pipeline
        images = stages.discover_images(source_directory)
        thumbnails = self._preprocessor.preprocess_batch(images)
        metrics = self._hub.analyze_batch(images, thumbnails)
        quality_passed = stages.filter_by_quality(metrics, ...)
        clusters = self._hub.cluster_images(features)
        selected = self._select_best_images(clusters, metrics)
        self._export_results(selected, output_directory)
```

**8 Workflow Stages**:
1. **discover_images**: Find all .jpg/.png files in directory
2. **preprocess**: Generate thumbnails (1024px, 2048px)
3. **analyze_quality**: Run IQA + AVA + Portrait analysis
4. **filter_quality**: Remove images below thresholds
5. **extract_features**: Get DINOv2 embeddings
6. **cluster_images**: Group similar images (HDBSCAN)
7. **select_best**: Pick best from each cluster
8. **export_results**: Copy selected images to output

### 3. Model Hub (Orchestrator)

**Main Class**: `ModelHub` in `sim_bench/model_hub/hub.py`

```python
class ModelHub:
    def __init__(self, config):
        # Lazy-loaded models (created on first use)
        self._iqa_model = None           # Rule-based quality
        self._ava_model = None           # YOUR trained AVA model
        self._portrait_analyzer = None   # MediaPipe
        self._feature_extractor = None   # DINOv2
    
    def analyze_image(self, image_path):
        # Run all analyses
        quality = self.score_quality(image_path)        # IQA
        ava = self.score_aesthetics(image_path)         # AVA (your model)
        portrait = self.analyze_portrait(image_path)    # MediaPipe
        
        return ImageMetrics(
            iqa_score=quality['overall'],
            ava_score=ava,
            has_face=portrait.has_face,
            eyes_open=portrait.eye_state.both_eyes_open,
            is_smiling=portrait.smile_state.is_smiling
        )
```

**Key Methods**:
- `score_quality()`: Rule-based IQA
- `score_aesthetics()`: YOUR AVA model
- `analyze_portrait()`: MediaPipe
- `extract_features()`: DINOv2
- `cluster_images()`: HDBSCAN

---

## Data Flow

### Complete Image Analysis Flow

```
User uploads photo folder
         ↓
┌─────────────────────────────┐
│  1. DISCOVER IMAGES         │
│  stages.discover_images()   │
│  → List of image paths      │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  2. PREPROCESS              │
│  ImagePreprocessor          │
│  → Generate thumbnails:     │
│    - 1024px for quality     │
│    - 2048px for portraits   │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  3. ANALYZE (ModelHub)      │
│                             │
│  For each image:            │
│  ┌─────────────────────┐   │
│  │ IQA Quality         │   │
│  │ (rule_based.py)     │   │
│  │ → sharpness: 0.8    │   │
│  │ → exposure: 0.7     │   │
│  └─────────────────────┘   │
│           +                 │
│  ┌─────────────────────┐   │
│  │ AVA Aesthetics      │   │
│  │ (YOUR TRAINED MODEL)│   │
│  │ → ava_score: 7.2    │   │
│  └─────────────────────┘   │
│           +                 │
│  ┌─────────────────────┐   │
│  │ Portrait Analysis   │   │
│  │ (MediaPipe)         │   │
│  │ → has_face: True    │   │
│  │ → eyes_open: True   │   │
│  │ → is_smiling: False │   │
│  └─────────────────────┘   │
│                             │
│  → ImageMetrics object      │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  4. FILTER                  │
│  stages.filter_by_quality() │
│  → Keep if:                 │
│    iqa >= 0.3               │
│    ava >= 4.0               │
│    sharpness >= 0.2         │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  5. FEATURE EXTRACTION      │
│  DINOv2                     │
│  → 768-dim embedding        │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  6. CLUSTERING              │
│  HDBSCAN                    │
│  → Cluster labels           │
│    [0, 0, 1, 1, 2, -1, ...]│
│    (-1 = noise/singleton)   │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  7. SELECT BEST             │
│  BestImageSelector          │
│  → For each cluster:        │
│    Score = 0.5*AVA +        │
│            0.2*IQA +        │
│            0.3*Portrait     │
│  → Pick highest score       │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  8. EXPORT                  │
│  FolderExporter/ZipExporter │
│  → Copy selected images     │
└─────────────────────────────┘
         ↓
User sees selected photos
```

---

## File Reference Guide

### Core Files You Need to Understand

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `app/album/main.py` | Streamlit entry point | `main()` |
| `sim_bench/album/workflow.py` | Pipeline orchestrator | `AlbumWorkflow`, `run()` |
| `sim_bench/model_hub/hub.py` | Model coordinator | `ModelHub`, `analyze_image()` |
| `sim_bench/image_quality_models/ava_model_wrapper.py` | AVA loader | `AVAQualityModel` |
| `sim_bench/models/ava_resnet.py` | AVA architecture | `AVAResNet` |
| `sim_bench/portrait_analysis/analyzer.py` | MediaPipe wrapper | `MediaPipePortraitAnalyzer` |
| `sim_bench/quality_assessment/rule_based.py` | Technical quality | `RuleBasedQuality` |
| `configs/global_config.yaml` | All settings | (YAML config) |

### Model Architecture Files vs Wrapper Files

**Architecture Files** (`sim_bench/models/`):
- Define network structure (layers, forward pass)
- Example: `ava_resnet.py` defines the AVAResNet class
- Pure PyTorch nn.Module

**Wrapper Files** (`sim_bench/image_quality_models/`):
- Load trained weights
- Handle preprocessing
- Provide simple `score_image()` interface
- Example: `ava_model_wrapper.py` loads YOUR checkpoint

### Where Your Training Connects

Your training scripts (`sim_bench/training/`) create checkpoints like `best_model.pt`.

The app loads these via wrappers:
```
Training:
  train_ava_resnet.py → saves → best_model.pt

Application:
  ava_model_wrapper.py → loads → best_model.pt → used in → ModelHub
```

---

## Quick Reference

### How to Enable AVA Model

1. Train AVA model (or locate existing checkpoint)
2. Add to `configs/global_config.yaml`:
   ```yaml
   quality_assessment:
     ava_checkpoint: path/to/your/best_model.pt
   ```
3. Restart app - AVA will auto-load

### How to Debug Model Loading

Check logs at `logs/sim-bench.log`:
```
INFO - Loaded AVA model from epoch 25, val_spearman=0.742, mode=distribution
```

If missing:
```
INFO - ModelHub initialized (device=cpu)
INFO - Loaded IQA model (RuleBasedQuality)
INFO - Loaded portrait analyzer (MediaPipe)
# No AVA line = checkpoint not configured
```

### How to Verify Thumbnails Working

Check cache directory:
```bash
ls -la .cache/album_analysis/medium/  # Should have .jpg files
ls -la .cache/album_analysis/large/   # Should have .jpg files
```

---

## Summary

**What you trained**:
- AVA ResNet model for aesthetic scoring
- Siamese CNN for image comparisons (not yet used in album app)

**What the app uses**:
- YOUR AVA model (if configured)
- Rule-based IQA (no training)
- MediaPipe (pre-trained)
- DINOv2 (pre-trained)
- HDBSCAN (algorithm)

**How to verify**:
1. Check `configs/global_config.yaml` for `ava_checkpoint` path
2. Run app and check logs for "Loaded AVA model"
3. Test with photos - aesthetic scores should appear in results

**Next Steps**:
1. Locate your trained `best_model.pt` checkpoint
2. Update config with correct path
3. Run app and verify AVA scores appear
4. Check telemetry to see which operations are slow

---

**Questions to answer**:
- Do you have a trained AVA checkpoint? Where is it?
- Do you want to use it in the album app?
- Should I help you configure it?
