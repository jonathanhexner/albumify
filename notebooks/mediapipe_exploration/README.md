# MediaPipe Portrait Analysis

This notebook explores MediaPipe for portrait image analysis, including:
- **Portrait Detection**: Identifying if people are the main subject
- **Eye State**: Detecting if eyes are open or closed
- **Smile Detection**: Identifying if the subject is smiling

## Setup

The notebook uses MediaPipe 0.10.9 with the `solutions` API.

If you encounter import errors:
1. Restart Jupyter kernel
2. Run: `pip install mediapipe==0.10.9 "numpy<2.0"`
3. Restart kernel again

## Configuration

All configurable parameters are at the top of the notebook (Section 2):

### Image Paths
- `IMAGE_DIR`: Directory containing images to analyze
- `OUTPUT_DIR`: Where to save analysis results

### Detection Thresholds
- `FACE_DETECTION_CONFIDENCE`: Face detection confidence (0.0-1.0)
- `PORTRAIT_FACE_RATIO_THRESHOLD`: Min face area ratio for portrait (default: 0.15)
- `PORTRAIT_CENTER_OFFSET_THRESHOLD`: Max center offset for portrait (default: 0.3)
- `EYE_OPEN_EAR_THRESHOLD`: Eye Aspect Ratio threshold (default: 0.2)
- `SMILE_WIDTH_THRESHOLD`: Smile width threshold (default: 0.45)
- `SMILE_ELEVATION_THRESHOLD`: Smile elevation threshold (default: 0.01)

### Visualization
- `NUM_SAMPLE_IMAGES`: Number of images to show in sample grids (default: 10)

**Adjust these at the top of the notebook to tune detection sensitivity!**

## Features

- Face detection and bounding box analysis
- Eye Aspect Ratio (EAR) calculation for eye state
  - **EAR**: Vertical-to-horizontal eye distance ratio
  - **Eye Decision**: EAR >= threshold means OPEN, EAR < threshold means CLOSED
  - Typical values: 0.15-0.25 for open eyes, <0.15 for closed
- Mouth landmark analysis for smile detection
  - **Width**: Mouth width ratio relative to face width
  - **Elevation**: How much mouth corners are raised
  - **Score**: Combined metric (0-1) weighting width, elevation, and lip separation
  - **Smile Decision**: BOTH width > threshold AND elevation > threshold must be true
- Batch processing and visualization
- DataFrame export for analysis

## Usage

Run the notebook cells in order. The main functions are:
- `is_portrait()`: Determine if image is a portrait
- `detect_eye_state()`: Check if eyes are open/closed
- `detect_smile()`: Detect smiling
- `analyze_image()`: Combined pipeline
- `analyze_batch()`: Process multiple images
