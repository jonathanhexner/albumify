# Photo Analysis App

Streamlit app for analyzing photos with CLIP, face detection, and landmarks.

## App

### **main.py**
**Run:** `streamlit run app/photo_analysis/main.py`

Analyzes photos and generates HTML reports.

**Features:**
- 🏷️ **CLIP Tagging** - Automatic scene and object detection
- 👤 **Face Detection** - Find and analyze faces
- 🗺️ **Landmark Recognition** - Identify famous landmarks
- 📄 **HTML Reports** - Generate comprehensive analysis reports

## Quick Start

```bash
streamlit run app/photo_analysis/main.py
```

Then:
1. Upload photos OR specify directory path
2. Configure analysis options:
   - Enable/disable CLIP tagging
   - Enable/disable face detection
   - Enable/disable landmark detection
3. Click "Analyze Photos"
4. View results and download HTML report

## Configuration Options

- **CLIP Tagging:**
  - Top-K tags (default: 5)
  - Threshold (default: 0.3)

- **Face Detection:**
  - Recognition mode
  - Confidence threshold

- **Landmark Detection:**
  - Recognition confidence

## Output

Generates comprehensive HTML report with:
- Image thumbnails
- Detected tags/scenes
- Faces (if enabled)
- Landmarks (if enabled)
- Metadata

## Requirements

```bash
pip install -r ../requirements.txt
```

Core dependencies:
- `streamlit`
- `sim_bench` (parent package)
- CLIP model
- Face detection models
- Landmark detection models
