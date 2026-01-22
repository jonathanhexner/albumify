# File Dependency Map

## How to Read This Map

This shows **exactly** which files call which, so you can trace any functionality.

Format: `file.py` → calls → `other_file.py`

---

## User Interface → Workflow → Models

### 1. Entry Point (User starts here)

```
app/album/main.py
  ├─ imports: config_panel, workflow_runner, results_viewer
  └─ function: main()
      ├─ calls → render_album_config()
      ├─ calls → render_workflow_form()
      ├─ calls → render_workflow_runner()
      └─ calls → render_results()
```

### 2. UI Components

```
app/album/config_panel.py
  └─ function: render_album_config()
      └─ returns: config_overrides (dict)

app/album/workflow_runner.py
  ├─ imports: sim_bench.album.workflow
  └─ function: render_workflow_runner()
      ├─ calls → create_album_workflow(config)
      ├─ calls → workflow.run(source, output)
      └─ displays: progress bar + status

app/album/results_viewer.py
  └─ function: render_results(result)
      ├─ tab: Gallery (image grid)
      ├─ tab: Metrics (dataframe)
      └─ tab: Performance (timings chart)
```

### 3. Workflow Orchestrator

```
sim_bench/album/workflow.py
  ├─ imports: model_hub.ModelHub, album.stages, album.preprocessor
  └─ class: AlbumWorkflow
      ├─ __init__():
      │   ├─ creates → ModelHub(config)
      │   ├─ creates → ImagePreprocessor(config)
      │   └─ creates → BestImageSelector(config)
      │
      └─ run(source_dir, output_dir):
          ├─ calls → stages.discover_images()
          ├─ calls → self._preprocessor.preprocess_batch()
          ├─ calls → self._hub.analyze_batch()          # KEY: Calls models
          ├─ calls → stages.filter_by_quality()
          ├─ calls → self._hub.extract_features_batch()  # KEY: Calls DINOv2
          ├─ calls → self._hub.cluster_images()          # KEY: Calls HDBSCAN
          ├─ calls → self._selector.select_best_images()
          └─ calls → self._export_results()
```

### 4. Model Hub (Coordinates all models)

```
sim_bench/model_hub/hub.py
  ├─ imports: 
  │   ├─ quality_assessment.rule_based
  │   ├─ image_quality_models.ava_model_wrapper
  │   ├─ portrait_analysis.analyzer
  │   ├─ feature_extraction.dinov2
  │   └─ clustering.hdbscan
  │
  └─ class: ModelHub
      ├─ analyze_image(path):                    # Called for each image
      │   ├─ calls → self.score_quality()        → RuleBasedQuality
      │   ├─ calls → self.score_aesthetics()     → AVAQualityModel (YOUR MODEL)
      │   └─ calls → self.analyze_portrait()     → MediaPipePortraitAnalyzer
      │
      ├─ extract_features(path):
      │   └─ calls → DINOv2FeatureExtractor.extract()
      │
      └─ cluster_images(features):
          └─ calls → HDBSCANClusterer.cluster()
```

---

## Model Loading Chain (Critical for Understanding)

### AVA Model (Your Trained Model)

```
ModelHub.score_aesthetics()
  ↓
sim_bench/image_quality_models/ava_model_wrapper.py
  ├─ class: AVAQualityModel
  ├─ __init__(checkpoint_path):
  │   ├─ torch.load(checkpoint_path)           ← LOADS YOUR best_model.pt
  │   ├─ imports: models.ava_resnet.AVAResNet
  │   └─ creates architecture + loads weights
  └─ score_image(path):
      └─ returns: 1-10 aesthetic score

sim_bench/models/ava_resnet.py
  └─ class: AVAResNet(nn.Module)
      ├─ __init__(): defines network layers
      └─ forward(): network forward pass
```

**Checkpoint must contain**:
```python
{
    'model_state_dict': {...},  # Trained weights
    'config': {
        'model': {
            'cnn_backbone': 'resnet50',
            'output_mode': 'distribution'
        },
        'transform': {...}
    }
}
```

### Rule-Based IQA (No Model File)

```
ModelHub.score_quality()
  ↓
sim_bench/quality_assessment/rule_based.py
  └─ class: RuleBasedQuality
      ├─ assess(path):
      │   ├─ calls → _assess_sharpness()     (OpenCV Laplacian)
      │   ├─ calls → _assess_exposure()      (histogram analysis)
      │   └─ calls → _assess_colorfulness()  (std of color channels)
      └─ returns: dict with scores (0-1)
```

### MediaPipe (Auto-download)

```
ModelHub.analyze_portrait()
  ↓
sim_bench/portrait_analysis/analyzer.py
  └─ class: MediaPipePortraitAnalyzer
      ├─ _load_models():
      │   ├─ import mediapipe as mp
      │   ├─ mp.solutions.face_detection.FaceDetection()  ← auto-downloads
      │   └─ mp.solutions.face_mesh.FaceMesh()            ← auto-downloads
      │
      └─ analyze(path):
          ├─ calls → face_detection.process()
          ├─ calls → face_mesh.process()
          ├─ calls → eye_state.calculate_eye_aspect_ratio()
          └─ calls → smile_detection.detect_smile()
```

### DINOv2 (Auto-download from Hugging Face)

```
ModelHub.extract_features()
  ↓
sim_bench/feature_extraction/dinov2.py
  └─ class: DINOv2FeatureExtractor
      ├─ __init__():
      │   └─ torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
      │       ↑ auto-downloads from Hugging Face
      │
      └─ extract(path):
          └─ returns: 768-dim embedding
```

---

## Configuration Flow

### How Config Gets to Models

```
configs/global_config.yaml                      ← YOU EDIT THIS
  ↓ loaded by
sim_bench/config.py: get_global_config()
  ↓ passed to
app/album/workflow_runner.py: create_album_workflow(config)
  ↓ creates
sim_bench/album/workflow.py: AlbumWorkflow(config)
  ↓ creates
sim_bench/model_hub/hub.py: ModelHub(config)
  ↓ reads
config['quality_assessment']['ava_checkpoint']
  ↓ if exists
sim_bench/image_quality_models/ava_model_wrapper.py: AVAQualityModel(checkpoint_path)
  ↓ loads
YOUR best_model.pt file
```

**Current config does NOT have ava_checkpoint**, so AVA returns `None`.

---

## Import Hierarchy (Who Imports Who)

### Top Level (UI)
```
app/album/main.py
  └─ imports: streamlit, app.album.{config_panel, workflow_runner, results_viewer}
```

### Middle Level (Orchestration)
```
app/album/workflow_runner.py
  └─ imports: sim_bench.album.workflow

sim_bench/album/workflow.py
  ├─ imports: sim_bench.model_hub.hub
  ├─ imports: sim_bench.album.stages
  ├─ imports: sim_bench.album.preprocessor
  ├─ imports: sim_bench.album.selection
  └─ imports: sim_bench.album.export
```

### Bottom Level (Models & Analysis)
```
sim_bench/model_hub/hub.py
  ├─ imports: sim_bench.quality_assessment.rule_based
  ├─ imports: sim_bench.image_quality_models.ava_model_wrapper
  ├─ imports: sim_bench.portrait_analysis.analyzer
  ├─ imports: sim_bench.feature_extraction.dinov2
  └─ imports: sim_bench.clustering.hdbscan

sim_bench/image_quality_models/ava_model_wrapper.py
  └─ imports: sim_bench.models.ava_resnet
```

**Rule**: UI depends on nothing below it. Models depend on nothing above them. Clean separation!

---

## Data Structures Passed Between Files

### ImageMetrics (from ModelHub to Workflow)

```python
# Defined in: sim_bench/model_hub/types.py
@dataclass
class ImageMetrics:
    iqa_score: float              # From RuleBasedQuality (0-1)
    ava_score: Optional[float]    # From AVAQualityModel (1-10) or None
    has_face: bool                # From MediaPipe
    eyes_open: bool               # From MediaPipe + eye_state.py
    is_smiling: bool              # From MediaPipe + smile_detection.py
    sharpness: float              # From RuleBasedQuality
    exposure: float               # From RuleBasedQuality
```

**Created in**: `ModelHub.analyze_image()`
**Used in**: `stages.filter_by_quality()`, `BestImageSelector.select_best_images()`

### PortraitMetrics (from MediaPipe to ModelHub)

```python
# Defined in: sim_bench/portrait_analysis/types.py
@dataclass
class PortraitMetrics:
    has_face: bool
    is_portrait: bool
    face_ratio: float
    eye_state: EyeState
    smile_state: SmileState
```

**Created in**: `MediaPipePortraitAnalyzer.analyze()`
**Consumed by**: `ModelHub.analyze_image()` → converted to ImageMetrics

---

## Execution Flow Example: Single Image

### What happens when you analyze `photo.jpg`

```
1. User clicks "Run Workflow" in Streamlit
   └─ app/album/main.py

2. UI calls workflow
   └─ workflow_runner.render_workflow_runner()

3. Workflow discovers images
   └─ workflow.run() → stages.discover_images()
   └─ finds: ["photo.jpg"]

4. Workflow preprocesses
   └─ preprocessor.preprocess_batch(["photo.jpg"])
   └─ generates:
       .cache/album_analysis/medium/photo_1024.jpg
       .cache/album_analysis/large/photo_2048.jpg

5. Workflow analyzes (THE KEY STEP)
   └─ hub.analyze_batch(["photo.jpg"], thumbnails)
   └─ for each image:
       
       5a. IQA Quality
           └─ hub.score_quality("photo.jpg")
           └─ rule_based.assess(medium_thumbnail)
           └─ returns: {overall: 0.75, sharpness: 0.8, exposure: 0.7}
       
       5b. AVA Aesthetics (IF CONFIGURED)
           └─ hub.score_aesthetics("photo.jpg")
           └─ ava_model_wrapper.score_image(medium_thumbnail)
           └─ loads YOUR model → runs forward pass
           └─ returns: 7.2
       
       5c. Portrait Analysis
           └─ hub.analyze_portrait("photo.jpg")
           └─ mediapipe_analyzer.analyze(large_thumbnail)
           └─ face_detection.process()
           └─ face_mesh.process()
           └─ eye_state.calculate_ear()
           └─ smile_detection.detect_smile()
           └─ returns: PortraitMetrics(has_face=True, eyes_open=True, ...)
       
       5d. Combine into ImageMetrics
           └─ return ImageMetrics(
                   iqa_score=0.75,
                   ava_score=7.2,
                   has_face=True,
                   eyes_open=True,
                   is_smiling=False
               )

6. Workflow filters
   └─ stages.filter_by_quality(metrics)
   └─ keeps if: iqa >= 0.3 AND ava >= 4.0
   └─ result: ["photo.jpg"] PASSED

7. Workflow extracts features
   └─ hub.extract_features("photo.jpg")
   └─ dinov2.extract(medium_thumbnail)
   └─ returns: [768-dim vector]

8. Workflow clusters
   └─ hub.cluster_images(features)
   └─ hdbscan.cluster(features)
   └─ returns: labels = [0]  (cluster 0)

9. Workflow selects best
   └─ selector.select_best_images(clusters, metrics)
   └─ for cluster 0:
       score = 0.5*7.2 + 0.2*0.75 + 0.3*1.0 = 3.6 + 0.15 + 0.3 = 4.05
   └─ best in cluster: "photo.jpg"

10. Workflow exports
    └─ export.copy_selected(["photo.jpg"], output_dir)
    └─ result: output_dir/photo.jpg

11. UI displays results
    └─ results_viewer.render_results(result)
    └─ shows: gallery with 1 image, metrics table
```

---

## File You Need to Check to Enable AVA

### Step-by-Step Checklist

1. **Find your trained checkpoint**
   ```bash
   # Search for it
   find . -name "*ava*.pt" -o -name "best_model.pt"
   ```

2. **Verify checkpoint format**
   ```python
   import torch
   ckpt = torch.load('path/to/best_model.pt')
   print(ckpt.keys())  # Must have: model_state_dict, config
   ```

3. **Edit config file**
   ```yaml
   # File: configs/global_config.yaml
   # Add under quality_assessment section (line 50):
   
   quality_assessment:
     default_method: clip_aesthetic
     enable_cache: true
     batch_size: 16
     ava_checkpoint: D:\path\to\your\best_model.pt  # ADD THIS
   ```

4. **Restart app**
   ```bash
   streamlit run app/album/main.py
   ```

5. **Check logs**
   ```bash
   tail -f logs/sim-bench.log
   # Look for: "Loaded AVA model from epoch X, val_spearman=Y"
   ```

6. **Test workflow**
   - Run on test photos
   - Check results → Metrics tab → should see `ava_score` column
   - Values should be 1-10 range

---

## Summary: Where Your Training Connects to the App

```
Your Training                     Application Usage
═══════════════                   ═════════════════

train_ava_resnet.py     ─────→    ava_model_wrapper.py
  ↓ trains                          ↓ loads
  ↓ saves                           ↓ uses
best_model.pt          ─────────→  ModelHub.score_aesthetics()
                                     ↓ returns
                                   aesthetic score (1-10)
                                     ↓ used in
                                   BestImageSelector
                                     ↓ affects
                                   which photos are selected
```

**Currently**: best_model.pt path NOT configured → app runs without aesthetics

**To enable**: Add `ava_checkpoint` to `configs/global_config.yaml`

---

## Questions This Should Answer

- ✅ What files exist? (See directory structure)
- ✅ What do they do? (See descriptions)
- ✅ How do they connect? (See dependency chains)
- ✅ Which models are used? (See model loading chains)
- ✅ Where are checkpoints loaded? (See AVA loading flow)
- ✅ How to enable YOUR models? (See configuration section)

**Still unclear?** Check which specific file or connection!
