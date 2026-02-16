# Pipeline Call Chain: Why MediaPipe Is Being Called

This document traces the exact call chain from user action to the MediaPipe error, explaining why MediaPipe code runs when the backend is configured for InsightFace.

---

## The Error

```
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
```

This error occurs in **MediaPipe** when using an incompatible protobuf version (4.x vs 3.x).

---

## Root Cause: Frontend/Backend Pipeline Mismatch

**The frontend sends the OLD MediaPipe pipeline, overriding the backend's InsightFace configuration.**

### Frontend `DEFAULT_PIPELINE` (pipeline_runner.py:12-27)

```python
# app/streamlit/components/pipeline_runner.py
DEFAULT_PIPELINE = [
    "discover_images",
    "detect_faces",           # ← MediaPipe step!
    "score_iqa",
    "score_ava",
    "score_face_pose",        # ← MediaPipe step!
    "score_face_eyes",        # ← MediaPipe step! THIS CAUSES THE ERROR
    "score_face_smile",       # ← MediaPipe step!
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "extract_face_embeddings",
    "cluster_people",
    "cluster_by_identity",
    "select_best",
]
```

### Backend `DEFAULT_PIPELINE` (pipeline_service.py:22-38)

```python
# sim_bench/api/services/pipeline_service.py
DEFAULT_PIPELINE = [
    "discover_images",
    "score_iqa",
    "score_ava",
    "detect_persons",              # ← YOLOv8-Pose (NO MediaPipe)
    "insightface_detect_faces",    # ← InsightFace SCRFD (NO MediaPipe)
    "insightface_score_expression",# ← InsightFace (NO MediaPipe)
    "insightface_score_eyes",      # ← InsightFace (NO MediaPipe)
    "insightface_score_pose",      # ← InsightFace (NO MediaPipe)
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "extract_face_embeddings",
    "cluster_people",
    "cluster_by_identity",
    "select_best",
]
```

---

## Complete Call Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ USER ACTION                                                                  │
│ User clicks "Run Pipeline" button in Streamlit UI                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: app/streamlit/components/pipeline_runner.py                        │
│                                                                              │
│ Line 69: steps = DEFAULT_PIPELINE  ← Uses the FRONTEND's list (MediaPipe)   │
│ Line 166: job_id = client.start_pipeline(album.id, steps, step_configs)     │
│                                                                              │
│ The frontend sends ["detect_faces", "score_face_eyes", ...] to the API      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP POST /api/albums/{id}/pipeline
                                    │ Body: { "steps": ["detect_faces", "score_face_eyes", ...] }
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ API ROUTER: sim_bench/api/routers/pipeline.py                                │
│                                                                              │
│ Receives the step list from frontend and passes to PipelineService          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SERVICE: sim_bench/api/services/pipeline_service.py                          │
│                                                                              │
│ Line 74-95: start_pipeline()                                                 │
│   - If steps is None → use backend DEFAULT_PIPELINE (InsightFace)           │
│   - If steps is provided → use the FRONTEND's list (MediaPipe!) ← PROBLEM   │
│                                                                              │
│ Line 127-162: execute_pipeline()                                             │
│   - Creates PipelineExecutor                                                 │
│   - Calls executor.execute(context, run.steps, config)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXECUTOR: sim_bench/pipeline/executor.py                                     │
│                                                                              │
│ Line 47-84: execute()                                                        │
│   - Line 69: steps = builder.build(step_names, auto_resolve=True)           │
│   - Resolves dependencies: "score_face_eyes" → depends_on ["detect_faces"]  │
│   - Loop through steps, call step.process() for each                        │
│                                                                              │
│ Line 127-154: _execute_step()                                                │
│   - Line 147: step.process(context, step_config)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP: sim_bench/pipeline/steps/score_face_eyes.py (MediaPipe version)        │
│                                                                              │
│ This is the MEDIAPIPE eye scoring step (NOT insightface_score_eyes)         │
│                                                                              │
│ Line 17-18: @register_step, class ScoreFaceEyesStep                          │
│   - name = "score_face_eyes"                                                 │
│   - depends_on = ["detect_faces"]                                            │
│                                                                              │
│ Line 110: process() → inherited from BaseStep                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BASE: sim_bench/pipeline/base.py                                             │
│                                                                              │
│ Line 101-113: process() - template method                                    │
│   - Checks if step uses template caching                                     │
│   - Line 110: self._process_with_cache(context, config)                     │
│                                                                              │
│ Line 127-205: _process_with_cache()                                          │
│   - Loads cached results or processes new items                              │
│   - Line 184: new_results = self._process_uncached(uncached_items, ...)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP: sim_bench/pipeline/steps/score_face_eyes.py                            │
│                                                                              │
│ Line 123-160: _process_uncached()                                            │
│   - Iterates over face images                                                │
│   - Line 147: eye_data = self._compute_eye_scores(face.image, config)       │
│                                                                              │
│ Line 61-96: _compute_eye_scores()                                            │
│   - Line 65: self._load_face_mesh()                                         │
│   - Line 69: results = self._face_mesh.process(image_np)  ← MEDIAPIPE CALL  │
│                                                                              │
│ Line 45-55: _load_face_mesh()                                                │
│   - import mediapipe as mp                                                   │
│   - mp.solutions.face_mesh.FaceMesh(...)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MEDIAPIPE: mediapipe/python/solutions/face_mesh.py                           │
│                                                                              │
│ Line 125: return super().process(input_data={'image': image})               │
│                                                                              │
│ → Calls into mediapipe/python/solution_base.py                              │
│ → Uses protobuf for serialization                                           │
│ → Calls GetPrototype() which doesn't exist in protobuf 4.x                  │
│                                                                              │
│ ERROR: AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Two Pipelines Explained

### MediaPipe Pipeline (Legacy)

Uses Google's MediaPipe library for face detection and analysis.

| Step Name | File | Description | Uses MediaPipe? |
|-----------|------|-------------|-----------------|
| `detect_faces` | `steps/detect_faces.py` | Face detection | ✅ Yes |
| `score_face_pose` | `steps/score_face_pose.py` | Head pose estimation | ✅ Yes (via SixDRepNet) |
| `score_face_eyes` | `steps/score_face_eyes.py` | Eye openness detection | ✅ Yes |
| `score_face_smile` | `steps/score_face_smile.py` | Smile detection | ✅ Yes |

### InsightFace Pipeline (Current)

Uses InsightFace library + YOLOv8 for detection and analysis. Does NOT use MediaPipe.

| Step Name | File | Description | Uses MediaPipe? |
|-----------|------|-------------|-----------------|
| `detect_persons` | `steps/detect_persons.py` | Person detection (YOLOv8-Pose) | ❌ No |
| `insightface_detect_faces` | `steps/insightface_detect_faces.py` | Face detection (SCRFD) | ❌ No |
| `insightface_score_expression` | `steps/insightface_score_expression.py` | Expression scoring | ❌ No |
| `insightface_score_eyes` | `steps/insightface_score_eyes.py` | Eye state scoring | ❌ No |
| `insightface_score_pose` | `steps/insightface_score_pose.py` | Face pose scoring | ❌ No |

---

## Step Dependencies (Why Steps Get Auto-Added)

The pipeline builder automatically resolves dependencies:

```
score_face_eyes
    └── depends_on: ["detect_faces"]
                        └── depends_on: ["discover_images"]

insightface_score_eyes
    └── depends_on: ["insightface_detect_faces"]
                        └── depends_on: ["discover_images", "detect_persons"]
```

If you include `score_face_eyes` in your pipeline, `detect_faces` is automatically added.

---

## What is MediaPipe?

**MediaPipe** is Google's open-source ML framework for:
- Face detection and 468-point face mesh
- Pose estimation (body landmarks)
- Hand tracking
- Object detection

In this codebase, it's used for:
1. **`detect_faces`** - Detect face bounding boxes
2. **`score_face_eyes`** - Use face mesh to compute Eye Aspect Ratio (EAR) for blink detection
3. **`score_face_smile`** - Use face mesh to detect smiling
4. **`score_face_pose`** - Work with SixDRepNet for head pose angles

---

## What is Protobuf?

**Protocol Buffers (protobuf)** is Google's binary serialization format:
- Like JSON but smaller and faster
- Used internally by MediaPipe to serialize ML model inputs/outputs
- MediaPipe was built for protobuf 3.x
- Protobuf 4.x removed `GetPrototype()` method that MediaPipe relies on

---

## Why the Error Happens

```
protobuf 4.x removed SymbolDatabase.GetPrototype()
       ↓
MediaPipe calls GetPrototype() at runtime
       ↓
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
```

---

## Fix Options

### Option 1: Update Frontend to Use InsightFace Pipeline (Recommended)

Edit `app/streamlit/components/pipeline_runner.py`:

```python
# Change DEFAULT_PIPELINE to use InsightFace steps:
DEFAULT_PIPELINE = [
    "discover_images",
    "score_iqa",
    "score_ava",
    "detect_persons",
    "insightface_detect_faces",
    "insightface_score_expression",
    "insightface_score_eyes",
    "insightface_score_pose",
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "extract_face_embeddings",
    "cluster_people",
    "cluster_by_identity",
    "select_best",
]
```

### Option 2: Don't Pass Steps from Frontend

Modify frontend to not send `steps` parameter, letting backend use its default:

```python
# In pipeline_runner.py, line ~166:
job_id = client.start_pipeline(album.id, steps=None, step_configs=step_configs)
#                                         ↑ Pass None to use backend default
```

### Option 3: Downgrade Protobuf (Quick Fix, Not Recommended)

```bash
pip install protobuf==3.20.3
```

This may cause issues with other packages that require protobuf 4.x.

---

## File Locations Summary

| Component | File Path | Purpose |
|-----------|-----------|---------|
| Frontend Pipeline Config | `app/streamlit/components/pipeline_runner.py:12-27` | Defines steps sent to API |
| Backend Pipeline Config | `sim_bench/api/services/pipeline_service.py:22-38` | Backend default (ignored when frontend sends steps) |
| YAML Pipeline Config | `configs/pipeline.yaml:14-28` | Reference config (not used at runtime) |
| MediaPipe Eye Scorer | `sim_bench/pipeline/steps/score_face_eyes.py` | Uses MediaPipe FaceMesh |
| InsightFace Eye Scorer | `sim_bench/pipeline/steps/insightface_score_eyes.py` | No MediaPipe |
| Pipeline Executor | `sim_bench/pipeline/executor.py` | Runs steps in order |
| Pipeline Builder | `sim_bench/pipeline/builder.py` | Resolves dependencies |
| Base Step | `sim_bench/pipeline/base.py` | Template method with caching |
| Step Registry | `sim_bench/pipeline/registry.py` | Maps step names to classes |
| All Steps Import | `sim_bench/pipeline/steps/all_steps.py` | Registers all steps |

---

## Visual: The Problem

```
                    ┌──────────────────────┐
                    │   Streamlit Frontend │
                    │                      │
                    │  DEFAULT_PIPELINE =  │
                    │  [..., score_face_   │
                    │   eyes, ...]         │ ← OUTDATED (MediaPipe)
                    └──────────┬───────────┘
                               │
                    Sends MediaPipe steps
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FastAPI Backend    │
                    │                      │
                    │  Has InsightFace     │
                    │  DEFAULT_PIPELINE    │ ← IGNORED because
                    │  but ignores it!     │   frontend sends steps
                    └──────────┬───────────┘
                               │
                    Executes MediaPipe steps
                               │
                               ▼
                    ┌──────────────────────┐
                    │   MediaPipe Library  │
                    │                      │
                    │  Incompatible with   │
                    │  protobuf 4.x        │
                    └──────────────────────┘
                               │
                               ▼
                         💥 ERROR 💥
```
