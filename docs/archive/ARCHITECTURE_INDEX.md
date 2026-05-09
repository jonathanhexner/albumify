# Architecture Documentation Index

**Purpose**: This index helps you navigate the complete architecture documentation for the Album Organization App.

**Start here if you're confused!**

---

## Quick Navigation

### Just Want to Get Started?
→ **[GETTING_STARTED.md](GETTING_STARTED.md)** ← READ THIS FIRST
- What the app does
- Current status (AVA not enabled)
- How to enable your trained model (30 seconds)
- Quick troubleshooting

### Want to Understand the Design?
→ **[ALBUM_APP_ARCHITECTURE.md](ALBUM_APP_ARCHITECTURE.md)**
- Complete system architecture (3 layers)
- All components explained
- Model loading details
- Data flow diagrams

### Want to Know Which Files Do What?
→ **[FILE_DEPENDENCY_MAP.md](FILE_DEPENDENCY_MAP.md)**
- Exact file-by-file dependency chains
- "This file calls this file" mappings
- Example execution traces
- Import hierarchy

### Want Model-Specific Details?
→ **[MODEL_USAGE_QUICK_REFERENCE.md](MODEL_USAGE_QUICK_REFERENCE.md)**
- Which models are used
- Which are YOUR trained models
- How to configure checkpoints
- Model loading troubleshooting

---

## Documentation Organization

```
docs/
├── ARCHITECTURE_INDEX.md           ← YOU ARE HERE
│   └─ Navigation hub for all architecture docs
│
├── GETTING_STARTED.md              ← START HERE
│   ├─ TL;DR summary
│   ├─ How to enable AVA model
│   ├─ Basic architecture overview
│   └─ Quick troubleshooting
│
├── ALBUM_APP_ARCHITECTURE.md       ← COMPREHENSIVE REFERENCE
│   ├─ System overview (3-layer architecture)
│   ├─ Directory structure
│   ├─ Model loading chains
│   ├─ Component details
│   ├─ Data flow diagrams
│   └─ File reference guide
│
├── FILE_DEPENDENCY_MAP.md          ← DETAILED CONNECTIONS
│   ├─ UI → Workflow → Models flow
│   ├─ Model loading chains
│   ├─ Configuration flow
│   ├─ Import hierarchy
│   ├─ Data structures
│   └─ Execution trace examples
│
└── MODEL_USAGE_QUICK_REFERENCE.md  ← MODEL SPECIFICS
    ├─ Active vs inactive models
    ├─ Your trained models location
    ├─ Configuration instructions
    ├─ Model loading flow
    └─ Troubleshooting model issues
```

---

## Read in This Order

### Level 1: Just Get It Working (15 min)
1. **[GETTING_STARTED.md](GETTING_STARTED.md)**
   - Read "TL;DR" section
   - Add `ava_checkpoint` to config
   - Verify app works with AVA

### Level 2: Understand the System (45 min)
2. **[ALBUM_APP_ARCHITECTURE.md](ALBUM_APP_ARCHITECTURE.md)**
   - Read "System Overview"
   - Study "Data Flow" section
   - Review "Model Loading & Usage"

3. **[MODEL_USAGE_QUICK_REFERENCE.md](MODEL_USAGE_QUICK_REFERENCE.md)**
   - Check which models are active
   - Understand checkpoint formats
   - Know where model files are

### Level 3: Deep Dive (2-3 hours)
4. **[FILE_DEPENDENCY_MAP.md](FILE_DEPENDENCY_MAP.md)**
   - Trace code execution paths
   - Understand file dependencies
   - See configuration flow

5. **Explore actual code files**
   - Start with `app/album/main.py`
   - Follow imports down the stack
   - Use FILE_DEPENDENCY_MAP as guide

---

## Key Concepts Explained

### Three-Layer Architecture

```
┌──────────────────────────────────────┐
│  LAYER 1: User Interface             │
│  What: Streamlit web app             │
│  Where: app/album/                   │
│  Read: GETTING_STARTED.md            │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│  LAYER 2: Workflow Orchestration     │
│  What: 8-stage pipeline              │
│  Where: sim_bench/album/             │
│  Read: ALBUM_APP_ARCHITECTURE.md     │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│  LAYER 3: ML Models                  │
│  What: Your trained models + others  │
│  Where: sim_bench/model_hub/         │
│  Read: MODEL_USAGE_QUICK_REFERENCE.md│
└──────────────────────────────────────┘
```

### Model Types

| Type | Example | Doc Section |
|------|---------|-------------|
| **Architecture** | `models/ava_resnet.py` | ALBUM_APP_ARCHITECTURE.md § "Model Loading" |
| **Wrapper** | `image_quality_models/ava_model_wrapper.py` | FILE_DEPENDENCY_MAP.md § "Model Loading Chain" |
| **Hub** | `model_hub/hub.py` | ALBUM_APP_ARCHITECTURE.md § "Component Architecture" |
| **Checkpoint** | `outputs/ava/.../best_model.pt` | MODEL_USAGE_QUICK_REFERENCE.md |

**Relationship**:
- Architecture defines network
- Checkpoint stores trained weights
- Wrapper loads checkpoint into architecture
- Hub uses wrapper to analyze images

### Configuration System

```
configs/global_config.yaml
  ↓ read by
sim_bench/config.py
  ↓ passed to
AlbumWorkflow(config)
  ↓ creates
ModelHub(config)
  ↓ reads
config['quality_assessment']['ava_checkpoint']
  ↓ loads
YOUR MODEL
```

**Details in**: FILE_DEPENDENCY_MAP.md § "Configuration Flow"

---

## Common Questions → Where to Find Answers

| Question | Document | Section |
|----------|----------|---------|
| How do I start the app? | GETTING_STARTED.md | "Step 1: Start the App" |
| Where is my trained model? | GETTING_STARTED.md | "TL;DR" or MODEL_USAGE_QUICK_REFERENCE.md |
| How do I enable AVA? | GETTING_STARTED.md | "To Enable Your AVA Model" |
| What does each file do? | FILE_DEPENDENCY_MAP.md | "File Reference Guide" |
| How does image analysis work? | ALBUM_APP_ARCHITECTURE.md | "Data Flow" |
| Which models are used? | MODEL_USAGE_QUICK_REFERENCE.md | "Currently Active Models" |
| Why is workflow slow? | GETTING_STARTED.md | "Troubleshooting" |
| How do files connect? | FILE_DEPENDENCY_MAP.md | "User Interface → Workflow → Models" |
| What is ModelHub? | ALBUM_APP_ARCHITECTURE.md | "Component Architecture § Model Hub" |
| What's in the config? | GETTING_STARTED.md | "Understanding the Config File" |

---

## Diagrams & Visual References

### System Overview Diagram
**Location**: ALBUM_APP_ARCHITECTURE.md § "System Overview"
```
User Interface → Workflow → Models → Trained Checkpoints
```

### Data Flow Diagram
**Location**: ALBUM_APP_ARCHITECTURE.md § "Data Flow"
```
Image → Preprocess → Analyze → Filter → Cluster → Select → Export
```

### File Dependency Chain
**Location**: FILE_DEPENDENCY_MAP.md § "User Interface → Workflow → Models"
```
main.py → workflow.py → hub.py → ava_model_wrapper.py → best_model.pt
```

### Configuration Flow
**Location**: FILE_DEPENDENCY_MAP.md § "Configuration Flow"
```
global_config.yaml → config.py → workflow → hub → models
```

---

## How to Use This Documentation

### Scenario 1: "I'm lost, what is this app?"
1. Read **GETTING_STARTED.md** fully (15 min)
2. Follow "Immediate Actions" checklist
3. Run app and observe results

### Scenario 2: "App works but I don't understand how"
1. Read **ALBUM_APP_ARCHITECTURE.md** § "System Overview"
2. Read **ALBUM_APP_ARCHITECTURE.md** § "Data Flow"
3. Trace one image through the system
4. Read **FILE_DEPENDENCY_MAP.md** § "Execution Flow Example"

### Scenario 3: "I want to modify/debug the code"
1. Read **FILE_DEPENDENCY_MAP.md** fully
2. Identify which file handles your area of interest
3. Read that file's code
4. Use dependency map to trace backwards/forwards

### Scenario 4: "Models not working / wrong model loaded"
1. Read **MODEL_USAGE_QUICK_REFERENCE.md** § "Troubleshooting"
2. Check **GETTING_STARTED.md** § "How to Verify Everything is Working"
3. Follow debugging steps
4. Check **FILE_DEPENDENCY_MAP.md** § "Model Loading Chain"

### Scenario 5: "Need to add a new model"
1. Read **ALBUM_APP_ARCHITECTURE.md** § "Model Loading & Usage"
2. Study existing wrapper (e.g., `ava_model_wrapper.py`)
3. Follow same pattern for new model
4. Update **FILE_DEPENDENCY_MAP.md** § "Model Loading Chain"

---

## Additional Resources

### Code-Level Documentation

```
sim_bench/
├── album/README.md              → Workflow module docs
├── model_hub/README.md          → Model hub docs
└── portrait_analysis/README.md  → MediaPipe docs

app/
└── album/README.md              → Streamlit UI docs
```

### Historical Context

```
docs/
├── RETROSPECTIVE.md             → Project history
├── PROJECT_SUMMARY.md           → Project overview
└── MILESTONES.md                → Development milestones
```

### User Guide (NOT YET WRITTEN)

Planned docs:
- `ALBUM_APP_USER_GUIDE.md` - For end users (non-technical)
- `CONFIGURATION_GUIDE.md` - All config options explained
- `MODEL_TRAINING_GUIDE.md` - How to train/retrain models

---

## Quick Reference Card

### Files You'll Edit Most
- `configs/global_config.yaml` - All settings
- `app/album/config_panel.py` - UI settings
- (Your training scripts) - To create new models

### Files You'll Read Most
- `sim_bench/album/workflow.py` - Pipeline logic
- `sim_bench/model_hub/hub.py` - Model coordination
- `sim_bench/image_quality_models/ava_model_wrapper.py` - AVA loading

### Files You'll Debug Most
- `logs/sim-bench.log` - Runtime logs
- `sim_bench/album/telemetry.py` - Performance tracking
- `sim_bench/model_hub/hub.py` - Model errors

### Commands You'll Run Most
```bash
# Start app
streamlit run app/album/main.py

# View logs
tail -f logs/sim-bench.log

# Find model files
ls outputs/ava/*/best_model.pt

# Test model loading
python -c "from sim_bench.model_hub import ModelHub; ..."
```

---

## Document Maintenance

### When to Update These Docs

**Update GETTING_STARTED.md when**:
- Default config changes
- New models added/removed
- Common troubleshooting issues found

**Update ALBUM_APP_ARCHITECTURE.md when**:
- Architecture changes (new layers/components)
- New model types added
- Data flow changes

**Update FILE_DEPENDENCY_MAP.md when**:
- Files renamed/moved
- New dependencies added
- Import structure changes

**Update MODEL_USAGE_QUICK_REFERENCE.md when**:
- New models added
- Checkpoint format changes
- Configuration options change

**Update ARCHITECTURE_INDEX.md (this file) when**:
- New architecture docs added
- Navigation structure changes
- Common questions evolve

---

## Feedback & Improvements

**Found something unclear?**
1. Note which document & section
2. Note what was confusing
3. Suggest clearer wording

**Found an error?**
1. Note which document & line
2. Note the incorrect info
3. Provide correct info

**Want more detail on something?**
1. Note which topic
2. Note what's missing
3. Suggest what to add

**These docs are living documents** - improve them as you learn!

---

## Summary

**Four main documents**:
1. **GETTING_STARTED.md** - Quick start & basic understanding
2. **ALBUM_APP_ARCHITECTURE.md** - Comprehensive system design
3. **FILE_DEPENDENCY_MAP.md** - Detailed code connections
4. **MODEL_USAGE_QUICK_REFERENCE.md** - Model-specific info

**Start with**: GETTING_STARTED.md

**Refer to**: Other docs as needed

**Goal**: Understand how YOUR trained models integrate with the app

---

**Ready to start?** → [GETTING_STARTED.md](GETTING_STARTED.md)
