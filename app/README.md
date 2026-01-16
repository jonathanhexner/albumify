# Photo Organization & Visualization Apps

Streamlit applications for photo analysis, organization, and clustering visualization.

## Available Apps

### 1. **Main App** (Recommended) 
**File:** `main.py`  
**Run:** `streamlit run app/main.py`

Production-grade photo organization app with clean architecture.

**Features:**
- 📁 Event clustering
- 👤 Face recognition  
- 🗺️ Landmark detection
- ⭐ Quality assessment
- 🏷️ Smart tagging

See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.

---

### 2. **Legacy Photo Analysis**
**File:** `legacy_photo_analysis.py`  
**Run:** `streamlit run app/legacy_photo_analysis.py`

Original photo analysis interface (simpler version).

**Features:**
- CLIP tagging
- Face detection
- Landmark recognition
- HTML report generation

---

### 3. **Agent Apps**
**Files:** `agent_v1.py`, `agent_v2.py`  
**Run:** 
```bash
streamlit run app/agent_v1.py
streamlit run app/agent_v2.py
```

Experimental AI agent interfaces for photo organization.

---

## Quick Start

**Recommended for new users:**
```bash
streamlit run app/main.py
```

Then:
1. Click "Initialize Agent" in sidebar
2. Enter your photo directory path
3. Start asking questions!

## Example Queries

- "Organize my photos by event"
- "Find my best 10 photos"
- "Group photos by person"
- "Show me all portrait photos"
- "Find photos from my vacation"

## Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

Core dependencies:
- `streamlit` - Web framework
- `sim_bench` - Image analysis package (this project)

## Development

### Project Structure
```
app/
├── main.py                    # Main app entry point ⭐
├── legacy_photo_analysis.py   # Legacy interface
├── agent_v1.py                # Agent interface v1
├── agent_v2.py                # Agent interface v2
├── config/                    # Configuration
├── core/                      # Business logic
├── state/                     # State management
├── ui/                        # UI components
└── README.md                  # This file
```

### Architecture

The main app (`main.py`) follows clean architecture principles:
- **Core** - Framework-agnostic business logic
- **UI** - Streamlit-specific presentation
- **State** - State management layer
- **Config** - Configuration and settings

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Clustering & Similarity Visualization

Clustering results are visualized as **interactive HTML galleries**, not in Streamlit.

**Location:** `sim_bench/clustering/gallery.py`

When you run clustering operations, HTML galleries are automatically generated with:
- Cluster statistics
- Image thumbnails grouped by cluster
- Interactive navigation
- Noise detection (DBSCAN)

**Generated files:**
- `outputs/clustering_*/clusters.html` - Interactive gallery
- `outputs/clustering_*/clusters.csv` - Cluster assignments

## Troubleshooting

### App won't start
- Ensure Python 3.10+ is installed
- Install all dependencies: `pip install -r requirements.txt`
- Check logs in console for errors

### No images found
- Verify directory path is correct
- Ensure `.jpg`/`.jpeg`/`.png` files exist
- Check file permissions

### Agent initialization fails
- Check logs for detailed error
- Ensure sim_bench package is installed correctly

## License

MIT License - see parent LICENSE file.

## Contributing

When adding new Streamlit apps:
1. Place them in `app/` directory
2. Use descriptive filenames
3. Add entry to this README
4. Follow clean architecture (see main.py)
