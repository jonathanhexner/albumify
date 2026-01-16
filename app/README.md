# Photo Apps

Streamlit applications for photo organization and analysis.

## Available Apps

### 📁 Photo Organization
**Directory:** `photo_organization/`

AI agent-based apps for organizing photos by events, people, landmarks, and quality.

```bash
# Production app (recommended)
streamlit run app/photo_organization/main.py
```

**Features:**
- Event clustering
- Face recognition
- Landmark detection
- Quality assessment
- Smart tagging

See [`photo_organization/README.md`](photo_organization/README.md) for details.

---

### 🔍 Photo Analysis
**Directory:** `photo_analysis/`

Analyze photos with CLIP tagging, face detection, and landmark recognition. Generates HTML reports.

```bash
streamlit run app/photo_analysis/main.py
```

**Features:**
- CLIP-based scene/object tagging
- Face detection and recognition
- Landmark identification
- HTML report generation

See [`photo_analysis/README.md`](photo_analysis/README.md) for details.

---

## Quick Start

**For organizing photos:**
```bash
streamlit run app/photo_organization/main.py
```

**For analyzing photos:**
```bash
streamlit run app/photo_analysis/main.py
```

## Directory Structure

```
app/
├── photo_organization/          # Photo organization apps
│   ├── main.py                 # Production app ⭐
│   ├── agent_v1.py             # Experimental v1
│   ├── agent_v2.py             # Experimental v2
│   └── README.md
│
├── photo_analysis/              # Photo analysis app
│   ├── main.py                 # Analysis + HTML reports
│   └── README.md
│
├── config/                      # Shared configuration
├── core/                        # Shared business logic
├── state/                       # Shared state management
├── ui/                          # Shared UI components
│
├── ARCHITECTURE.md              # Architecture documentation
└── README.md                    # This file
```

## Shared Components

Both app categories share:
- **config/** - Configuration and constants
- **core/** - Business logic (models, services)
- **state/** - State management
- **ui/** - Reusable UI components

This enables code reuse and consistent architecture.

## Clustering & Similarity Visualization

**Note:** Clustering results are visualized as **HTML galleries** (not in Streamlit).

**Location:** `../sim_bench/clustering/gallery.py`

When you run clustering operations, interactive HTML galleries are generated:
- Cluster statistics
- Image thumbnails grouped by cluster
- Interactive navigation
- Noise detection (DBSCAN)

**Output:** `outputs/clustering_*/clusters.html`

## Requirements

```bash
# From project root
pip install -r requirements.txt
```

Core dependencies:
- `streamlit` - Web framework
- `sim_bench` - Image analysis (this project)

## Development

### Adding New Apps

1. Choose category: `photo_organization/` or `photo_analysis/`
2. Create new app file in appropriate directory
3. Update category README
4. Use shared components from parent directories

### Architecture

Photo organization apps follow clean architecture:
- **Core** - Framework-agnostic business logic
- **UI** - Streamlit-specific presentation
- **State** - State management
- **Config** - Configuration

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Troubleshooting

### App won't start
- Ensure Python 3.10+ is installed
- Install dependencies: `pip install -r ../requirements.txt`
- Check console for errors

### No images found
- Verify directory path
- Ensure `.jpg`/`.jpeg`/`.png` files exist
- Check file permissions

### Agent initialization fails
- Check logs for detailed error
- Ensure sim_bench is installed correctly

## License

MIT License - see parent LICENSE file.
