# Active vs Legacy Pipeline - Quick Reference

## 🎯 What You Need to Know

### ✅ ACTIVE (Used by API)

```
API Request
    ↓
DEFAULT_PIPELINE (hardcoded in pipeline_service.py)
    ↓
┌─────────────────────────────────────────────────────┐
│  1. discover_images                                 │
│  2. score_iqa                                       │
│  3. score_ava                                       │
│  4. detect_persons (YOLOv8)         ← NEW          │
│  5. insightface_detect_faces        ← NEW          │
│  6. insightface_score_expression    ← NEW          │
│  7. insightface_score_eyes          ← NEW          │
│  8. insightface_score_pose          ← NEW          │
│  9. filter_quality                                  │
│ 10. extract_scene_embedding                         │
│ 11. cluster_scenes                                  │
│ 12. extract_face_embeddings                         │
│ 13. cluster_people                                  │
│ 14. cluster_by_identity                             │
│ 15. select_best (composite scoring) ← REDESIGNED   │
└─────────────────────────────────────────────────────┘
```

**Key Components**:
- `sim_bench/pipeline/steps/detect_persons.py`
- `sim_bench/pipeline/steps/insightface_*.py`
- `sim_bench/pipeline/steps/select_best.py` ✨ (Your recent work!)
- `sim_bench/pipeline/scoring/quality_strategy.py`
- `sim_bench/pipeline/scoring/person_penalty.py`

---

### ⚠️ LEGACY (Not Used by API)

```
┌─────────────────────────────────────────────────────┐
│  face_pipeline/ module                              │
│  ├── pipeline.py (FacePipelineService)              │
│  ├── quality_scorer.py (FaceQualityScorer)          │
│  ├── crop_service.py                                │
│  └── pose_estimator.py                              │
└─────────────────────────────────────────────────────┘
    ↓
Used by OLD MediaPipe pipeline steps:
    ↓
┌─────────────────────────────────────────────────────┐
│  - detect_faces (MediaPipe)                         │
│  - score_face_quality                               │
│  - score_face_pose                                  │
│  - score_face_eyes                                  │
│  - score_face_smile                                 │
└─────────────────────────────────────────────────────┘
```

**Status**: Available but not used by default API pipeline.

---

## 📋 Component Comparison

| Feature | ACTIVE (InsightFace) | LEGACY (face_pipeline) |
|---------|----------------------|------------------------|
| **Person Detection** | YOLOv8-Pose ✅ | Not available |
| **Face Detection** | InsightFace SCRFD ✅ | MediaPipe |
| **Face Scoring** | Modular steps ✅ | FaceQualityScorer |
| **Architecture** | Modular pipeline steps ✅ | All-in-one service |
| **Scoring Logic** | Composite (quality + penalty) ✅ | Weighted face attributes |
| **Used By** | API default pipeline ✅ | Legacy code only |
| **Your Recent Work** | YES ✅ | NO |

---

## 🔍 How to Tell Which is Which

### File Paths

**ACTIVE**:
```
sim_bench/pipeline/steps/
├── detect_persons.py
├── insightface_detect_faces.py
├── insightface_score_*.py
├── select_best.py          ← You just redesigned this!
└── ...

sim_bench/pipeline/scoring/
├── quality_strategy.py     ← New quality scoring
└── person_penalty.py       ← New penalty computation
```

**LEGACY**:
```
sim_bench/face_pipeline/
├── pipeline.py             ← FacePipelineService (not used)
├── quality_scorer.py       ← FaceQualityScorer (not used)
├── crop_service.py
├── pose_estimator.py
└── types.py
```

### Code Patterns

**ACTIVE** - Modular steps:
```python
class InsightFaceDetectFacesStep(BaseStep):
    """Single-purpose pipeline step."""
    
    def process(self, context, config):
        # Do one thing
        # Write to context
        pass
```

**LEGACY** - All-in-one service:
```python
class FacePipelineService:
    """Complete face processing pipeline."""
    
    def process_album(self, image_paths):
        # Does everything: crop, score, embed, cluster
        # Returns complete result object
        pass
```

---

## 🎯 Answer to "Which is Being Used?"

**Q: When I run the API, which code executes?**

**A: ACTIVE (InsightFace) pipeline**

The flow is:
```
1. Start API: python -m uvicorn sim_bench.api.main:app
2. API receives: POST /api/v1/pipeline/run
3. PipelineService.start_pipeline()
   → Uses DEFAULT_PIPELINE (InsightFace steps)
4. Pipeline executes:
   → detect_persons ✅
   → insightface_detect_faces ✅
   → insightface_score_* ✅
   → select_best ✅ (your redesigned version!)
5. Results saved to database
```

**The face_pipeline/ module is NOT involved at all!**

---

## 📝 What You've Been Working On

### Your Recent Work: ✅ ACTIVE CODE

You redesigned `select_best.py` (ACTIVE) to use:
- `quality_strategy.py` - Image quality scoring
- `person_penalty.py` - Person/portrait penalties
- Composite scoring: `quality + penalty`

This is **exactly the right code** and is **actively used by the API**.

### What You Were Confused About: ⚠️ LEGACY CODE

- `face_pipeline/pipeline.py` - NOT used by API
- `face_pipeline/quality_scorer.py` - NOT used by API

These are legacy MediaPipe implementations that are no longer the default.

---

## 🚀 Running the System

### Start the Active Pipeline

```bash
# Terminal 1: Start API
python -m uvicorn sim_bench.api.main:app --reload --port 8000

# Terminal 2: Start Streamlit
streamlit run app/streamlit/main.py
```

This will use the **ACTIVE InsightFace pipeline** with your new composite scoring!

### Test Legacy Pipeline (Optional)

If you want to test the old MediaPipe pipeline:

```python
# Explicitly request MediaPipe steps
POST /api/v1/pipeline/run
{
  "album_id": "...",
  "steps": [
    "discover_images",
    "detect_faces",           # MediaPipe
    "score_face_pose",        # MediaPipe
    "score_face_eyes",        # MediaPipe
    "score_face_smile",       # MediaPipe
    "cluster_scenes",
    "select_best"
  ]
}
```

But the API defaults to the **InsightFace pipeline** if no steps specified.

---

## 📚 Further Reading

- `PIPELINE_ARCHITECTURE_CURRENT_STATE.md` - Detailed architecture explanation
- `SELECT_BEST_ARCHITECTURE.md` - Your new composite scoring design
- `SELECT_BEST_REDESIGN_IMPLEMENTATION.md` - Implementation guide

---

## ✅ Bottom Line

**You've been working on the RIGHT code!**

- ✅ Your work is in the ACTIVE pipeline
- ✅ It's used by the API by default
- ⚠️ The `face_pipeline/` stuff is legacy
- ⚠️ Don't worry about `FaceQualityScorer` - it's not used

The confusion came from having two parallel implementations in the codebase, but now you know:
- **What's active**: InsightFace steps + your new select_best
- **What's legacy**: face_pipeline module (ignore it!)

🎉 Keep working on `select_best.py` and the `pipeline/scoring/` modules - that's where the action is!
