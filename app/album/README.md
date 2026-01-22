# Album Organization UI

Streamlit UI for photo album organization workflow.

## Overview

This module provides an interactive web interface for organizing photo albums:
- Configure quality and portrait preferences
- Run workflow with real-time progress
- View results in cluster gallery
- Download selected images and metrics

## Components

### Config Panel (`config_panel.py`)
Interactive configuration interface with sliders and checkboxes for:
- Quality thresholds (IQA, AVA, sharpness)
- Portrait preferences (eyes open, smiling)
- Clustering settings (method, min cluster size)
- Selection criteria (weights, images per cluster)
- Export options (format, organization)

### Workflow Runner (`workflow_runner.py`)
Executes album workflow with:
- Progress bar and stage descriptions
- Real-time status updates
- Summary metrics display
- Error handling and validation

### Results Viewer (`results_viewer.py`)
Displays workflow results with:
- Cluster gallery with images
- Selected images highlighted
- Detailed metrics table (CSV download)
- Quality distribution charts
- Export information and file list

## Usage

### Running the App

```bash
streamlit run app/album/main.py
```

### Workflow Steps

1. **Configure Settings**
   - Adjust quality thresholds
   - Set portrait preferences
   - Choose clustering method
   - Configure selection weights

2. **Select Album**
   - Enter source directory path
   - Provide album name
   - Specify output directory

3. **Run Workflow**
   - Click "Start Workflow"
   - Monitor progress
   - View summary metrics

4. **View Results**
   - Browse cluster gallery
   - Review image metrics
   - Download selected images
   - Export metrics as CSV

## Example Session

```python
# Example configuration from UI
config = {
    'album': {
        'quality': {
            'min_iqa_score': 0.3,
            'min_ava_score': 4.0,
            'min_sharpness': 0.2
        },
        'portrait': {
            'require_eyes_open': True,
            'prefer_smiling': True
        },
        'clustering': {
            'method': 'hdbscan',
            'min_cluster_size': 3
        },
        'selection': {
            'images_per_cluster': 1,
            'ava_weight': 0.5,
            'iqa_weight': 0.2,
            'portrait_weight': 0.3
        }
    }
}
```

## Features

### Quality Configuration
- **IQA Score**: Technical quality (sharpness, exposure, etc.)
- **AVA Score**: Aesthetic quality from AVA model
- **Sharpness**: Dedicated sharpness threshold

### Portrait Preferences
- **Eyes Open Filter**: Automatically exclude closed-eye photos
- **Smile Preference**: Bonus scoring for smiling faces
- **Configurable Weights**: Adjust importance of eyes/smile

### Smart Clustering
- **HDBSCAN**: Density-based clustering (default)
- **DBSCAN**: Alternative density-based method
- **K-Means**: Fixed number of clusters

### Flexible Selection
- **Weighted Scoring**: Combine quality, aesthetics, portrait metrics
- **Multi-image Selection**: Choose top N per cluster
- **Siamese Tiebreaker**: Use ML model for close decisions

### Export Options
- **Folder Export**: Organized directory structure
- **ZIP Export**: Compressed archive
- **Cluster Organization**: Separate folders per cluster
- **Thumbnails**: Auto-generated preview images
- **Metadata**: JSON file with all metrics

## UI Layout

```
┌─────────────────────────────────────┐
│  Album Organization                  │
├─────────────────────────────────────┤
│  📂 Album Selection                  │
│    - Source Directory                │
│    - Album Name                      │
│    - Output Directory                │
├─────────────────────────────────────┤
│  ⚙️ Album Configuration             │
│    📊 Quality Thresholds             │
│    👤 Portrait Preferences           │
│    🔍 Clustering Settings            │
│    ✨ Selection Settings             │
│    📤 Export Settings                │
├─────────────────────────────────────┤
│  🚀 Workflow Execution               │
│    ▶️ Start Workflow                 │
│    [Progress Bar]                    │
│    📊 Summary Metrics                │
├─────────────────────────────────────┤
│  📸 Results                          │
│    🖼️ Gallery | 📊 Metrics | 📁 Export │
└─────────────────────────────────────┘
```

## Integration

The UI integrates with:
- `sim_bench.album.workflow` - Core workflow engine
- `sim_bench.model_hub` - ML model orchestration
- `sim_bench.portrait_analysis` - Face/eyes/smile detection
- `sim_bench.config` - Global configuration system

## Development

### Adding New Settings

1. Update `config_panel.py` with new UI control
2. Add setting to `config_overrides` dictionary
3. Ensure corresponding setting exists in `global_config.yaml`

### Customizing Display

Modify `results_viewer.py` to change:
- Gallery layout (columns, image size)
- Metrics displayed
- Chart types and styling
- Download formats
