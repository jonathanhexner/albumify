# Photo Organization Apps

AI agent-based photo organization applications.

## Apps

### 1. **main.py** (Production - Recommended)
**Run:** `streamlit run app/photo_organization/main.py`

Production app with clean architecture.

**Features:**
- 📁 Event clustering
- 👤 Face recognition
- 🗺️ Landmark detection
- ⭐ Quality assessment
- 🏷️ Smart tagging

**Architecture:** Uses modular design with `../core/`, `../ui/`, `../state/`

---

### 2. **agent_v1.py** (Experimental)
**Run:** `streamlit run app/photo_organization/agent_v1.py`

Simple experimental agent interface with modular design.

---

### 3. **agent_v2.py** (Experimental)
**Run:** `streamlit run app/photo_organization/agent_v2.py`

Feature-rich experimental agent with professional architecture.

---

## Quick Start

```bash
# Recommended: Use the production app
streamlit run app/photo_organization/main.py
```

Then:
1. Click "Initialize Agent" in sidebar
2. Enter your photo directory path
3. Ask questions like:
   - "Organize my photos by event"
   - "Find my best 10 photos"
   - "Group photos by person"

## Shared Components

These apps use shared components from parent directory:
- `../config/` - Configuration
- `../core/` - Business logic
- `../state/` - State management
- `../ui/` - UI components

See `../ARCHITECTURE.md` for details.
