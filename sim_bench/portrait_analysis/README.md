# Portrait Analysis Module

MediaPipe-based portrait analysis for detecting faces, eye states, and smiles.

## Usage

```python
from sim_bench.config import get_global_config
from sim_bench.portrait_analysis import MediaPipePortraitAnalyzer

config = get_global_config().to_dict()
analyzer = MediaPipePortraitAnalyzer(config)

# Analyze single image
metrics = analyzer.analyze_image("path/to/image.jpg")
print(f"Is portrait: {metrics.is_portrait}")
print(f"Eyes open: {metrics.eye_state.both_eyes_open if metrics.eye_state else 'N/A'}")
print(f"Smiling: {metrics.smile_state.is_smiling if metrics.smile_state else 'N/A'}")

# Analyze batch
results = analyzer.analyze_batch(["img1.jpg", "img2.jpg"])
```

## Configuration

Settings in `configs/global_config.yaml` under `portrait_analysis`:

| Key | Default | Description |
|-----|---------|-------------|
| face_detection_confidence | 0.5 | Minimum confidence for face detection |
| portrait_face_ratio_threshold | 0.0005 | Min face area ratio for portrait |
| portrait_center_offset_threshold | 0.3 | Max horizontal offset from center |
| eye_open_ear_threshold | 0.2 | Eye Aspect Ratio threshold |
| smile_width_threshold | 0.15 | Mouth width ratio for smile |
| smile_elevation_threshold | 0.005 | Corner elevation for smile |

## Data Types

- `EyeState`: left/right eye open status and EAR values
- `SmileState`: smile detection with score and mouth metrics
- `PortraitMetrics`: Complete analysis result for an image

## Dependencies

- mediapipe
- opencv-python
- numpy
